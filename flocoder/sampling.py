import torch
import wandb
from scipy import integrate  # This is CPU-only
from .viz import save_img_grid
from .metrics import sinkhorn_loss, fid_score, g2rgb
from functools import partial


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def warp_time(t,            # time values to warp
              dt=None,      # optional derivative request
              s=.5):        # slope parameter: s=1 is linear, s<1 slower middle, s>1 slower ends
    """Parametric Time Warping: s = slope in the middle.
        s=1 is linear time, s < 1 goes slower near the middle, s>1 goes slower near the ends
        s = 1.5 gets very close to the "cosine schedule", i.e. (1-cos(pi*t))/2, i.e. sin^2(pi/2*x)"""
    if s<0 or s>1.5: raise ValueError(f"s={s} is out of bounds.")
    tw = 4*(1-s)*t**3 + 6*(s-1)*t**2 + (3-2*s)*t
    if dt:                           # warped time-step requested; use derivative
        return tw, dt * 12*(1-s)*t**2 + 12*(s-1)*t + (3-2*s)
    return tw


@torch.no_grad()
def rk4_step(f,     # function that takes (y,t) and returns dy/dt, i.e. velocity
             y,     # current location
             t,     # current t value
             dt):   # requested time step size
    k1 = f(y, t)
    k2 = f(y + dt*k1/2, t + dt/2)
    k3 = f(y + dt*k2/2, t + dt/2)
    k4 = f(y + dt*k3, t + dt)
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


@torch.no_grad()
def v_func_cfg(model,          # the flow model
               cond,           # conditioning (can be None)
               cfg_strength,   # classifier-free guidance strength
               x,              # current state
               t):             # current time
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype 
    t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
    v = model(x, t_vec * 999, cond)
    if cond is not None and cfg_strength is not None and cfg_strength != 0:
        v_uncond = model(x, t_vec * 999, None)
        v = v_uncond + cfg_strength * (v - v_uncond)
    return v


@torch.no_grad()
def generate_latents_rk4(model,          # the flow model 
                         shape,          # (batch_size, channels, height, width)
                         n_steps=50,     # integration steps
                         cond=None,      # conditioning labels/embeddings
                         cfg_strength=3.0): # classifier-free guidance strength
    """Generate latents using RK4 integration - this 'sampling' routine is primarily used for visualization."""
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype 
    initial_points = torch.randn(shape, device=device, dtype=dtype)
    current_points = initial_points.clone()
    ts = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)
    if warp_time: ts = warp_time(ts)
    v_func = partial(v_func_cfg, model, cond, cfg_strength)
    for i in range(len(ts)-1):
        current_points = rk4_step(v_func, current_points, ts[i], ts[i+1]-ts[i])
    nfe = n_steps * 4  # number of function evaluations
    return current_points, nfe


@torch.no_grad()
def generate_latents_rk45(model,            # the flow model
                          shape,            # (batch_size, channels, height, width) 
                          device,           # torch device
                          cond=None,        # conditioning labels/embeddings
                          cfg_strength=3.0, # classifier-free guidance strength  
                          eps=0.001):       # integration epsilon
    """Runge-Kutta method for integration. Original Source: Tadao Yamaoka"""
    rtol = atol = 1e-05
    model.eval()
    z0 = torch.randn(shape).to(device)
    
    def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        t_w = warp_time(t) 
        t_vec = torch.ones(shape[0], device=x.device) * t_w
        v = model(x, t_vec * 999, cond)                # if cond=None you get uncond
        if cond is not None and cfg_strength is not None and cfg_strength != 0:   # classifier-free guidance
            v_uncond = model(x, t_vec * 999, None)
            v = v_uncond + cfg_strength * (v - v_uncond) # cfg_strength ~= conditioning strength
        return to_flattened_numpy(v)

    solution = integrate.solve_ivp(   # TODO: we could just do regular rk4 and maybe have it faster; time it
        ode_func, (eps, 1), to_flattened_numpy(z0), rtol=rtol, atol=atol, method="RK45")
    nfe = solution.nfev
    x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)
    return x, nfe


@torch.no_grad()
def generate_latents(model,            # the flow model
                     shape,            # (batch_size, channels, height, width)
                     method='rk4',     # integration method: 'rk4' or 'rk45'
                     n_steps=50,       # number of steps (rk4 only)
                     cond=None,        # conditioning labels/embeddings
                     cfg_strength=3.0, # classifier-free guidance strength
                     device=None):     # torch device (auto-detected if None)
    """Generate latents using specified method"""
    if device is None: device = next(model.parameters()).device
    if method == "rk45":
        return generate_latents_rk45(model, shape, device, cond, cfg_strength)
    else:  # default to rk4
        return generate_latents_rk4(model, shape, n_steps, cond, cfg_strength)


def decode_latents(codec,          # the codec/autoencoder model
                   latents,        # latent tensors to decode
                   is_midi=False,  # whether this is MIDI data
                   keep_gray=False): # keep grayscale format for MIDI
    """Decode latents to images"""
    decoded = codec.decode(latents.to(next(codec.parameters()).device))
    return g2rgb(decoded, keep_gray=keep_gray) if is_midi else decoded


def compute_sample_metrics(pred_latents,   # predicted latent samples
                          target_latents,  # target latent samples (can be None)
                          codec,           # codec for decoding to pixel space
                          is_midi=False,   # whether this is MIDI data
                          keep_gray=False): # keep grayscale format for MIDI
    """Compute metrics between predicted and target samples"""
    if target_latents is None: return {}
    
    device = next(codec.parameters()).device
    batch_size = min(pred_latents.shape[0], target_latents.shape[0])
    
    # Latent space metrics
    sinkhorn_latent = sinkhorn_loss(target_latents[:batch_size], pred_latents[:batch_size], device=device)
    
    # Pixel space metrics - note: can't do FID on latents unless latents have 3 channels
    with torch.no_grad():  # TODO: this shouldn't be needed
        decoded_pred = decode_latents(codec, pred_latents[:batch_size], is_midi, keep_gray)
        decoded_target = decode_latents(codec, target_latents[:batch_size], is_midi, keep_gray)
    
    sinkhorn_px = sinkhorn_loss(decoded_target, decoded_pred, device=device)  # pixel space
    fid_px = fid_score(decoded_target, decoded_pred)  # pixel space
    
    return {
        'metrics/sinkhorn': sinkhorn_latent,
        'metrics/sinkhorn_px': sinkhorn_px, 
        'metrics/FID_px': fid_px
    }


@torch.no_grad()
def evaluate_model(model,            # the flow model to evaluate
                   codec,            # codec for decoding
                   target_latents=None,  # target samples for metrics (optional)
                   target_labels=None,   # labels for conditioning (optional)
                   batch_size=256,   # number of samples to generate
                   n_classes=0,      # number of classes for conditioning
                   latent_shape=(4,16,16), # shape of latent space
                   method='rk4',     # integration method
                   n_steps=50,       # number of integration steps
                   is_midi=False,    # whether this is MIDI data
                   keep_gray=False,  # keep grayscale format
                   cfg_strength=3.0): # classifier-free guidance strength
    """Generate samples and compute metrics for model evaluation"""
    model.eval(), codec.eval()
    
    # Handle conditioning
    cond = target_labels if n_classes > 0 else None
    if cond is None and n_classes > 0:  # grid where each column is a single class (10 columns)
        cond = torch.randint(n_classes, (10,)).repeat(batch_size // 10).to(next(model.parameters()).device)
    elif cond is not None:
        cond = cond[:batch_size]  # only grab what we can show
        # TODO: get indices to sort cond in ascending order, and sort cond using those indices
        # if target is not none: use the same indices to sort target
    
    # Generate samples
    shape = (batch_size,) + latent_shape
    pred_latents, nfe = generate_latents(model, shape, method, n_steps, cond, cfg_strength)
    
    # Compute metrics on generated outputs:
    metrics = compute_sample_metrics(pred_latents, target_latents, codec, is_midi, keep_gray)
    metrics['nfe'] = nfe
    
    model.train()
    return pred_latents, metrics


# Legacy function for backward compatibility - simplified interface
@torch.no_grad()
def sampler(model, codec, epoch, method='rk4', batch_size=256, tag="", 
            target=None, target_labels=None, n_classes=0, latent_shape=(4,16,16),
            is_midi=False, keep_gray=False, use_wandb=True, output_dir="output"):
    """Simplified sampler for training evaluation - use evaluate_model for more control"""
    
    # Generate samples and compute metrics
    pred_latents, metrics = evaluate_model(model, codec, target, target_labels, batch_size,
                                         n_classes, latent_shape, method, 50, 
                                         is_midi, keep_gray, 3.0)
    
    # Save outputs
    nfe = metrics.get('nfe', 50)
    save_img_grid(pred_latents.cpu(), epoch, nfe, tag=tag+method, use_wandb=use_wandb, output_dir=output_dir)
    
    decoded_images = decode_latents(codec, pred_latents, is_midi, keep_gray)
    save_img_grid(decoded_images.cpu(), epoch, nfe, tag="decoded_"+tag+method, 
                 use_wandb=use_wandb, output_dir=output_dir)
    
    # Log metrics
    if use_wandb and metrics:
        metrics['epoch'] = epoch
        wandb.log(metrics)
    
    return metrics

