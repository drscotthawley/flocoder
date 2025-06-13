import torch
import wandb
from scipy import integrate  # This is CPU-only
from .viz import save_img_grid
from .metrics import sinkhorn_loss, fid_score, g2rgb
from functools import partial
import time
import gc


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
             dt,    # requested time step size
             debug=False
             ):
    start = time.time()
    k1 = f(y, t)
    if debug: print(f"  k1: {(time.time()-start)*1000:.1f}ms")
    tpdto2 = t+dt/2
    start = time.time()
    k2 = f(y + dt*k1/2, tpdto2)
    if debug: print(f"  k2: {(time.time()-start)*1000:.1f}ms")
    start = time.time()
    k3 = f(y + dt*k2/2, tpdto2)
    if debug: print(f"  k3: {(time.time()-start)*1000:.1f}ms")
    start = time.time()
    k4 = f(y + dt*k3, t + dt)
    if debug: print(f"  k4: {(time.time()-start)*1000:.1f}ms")
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

@torch.no_grad()
def rk4_step_new(model, cond, cfg_strength, y, t, dt):
    model.eval()
    k1 = v_func_cfg(model, cond, cfg_strength, y, t).detach()
    tpdto2 = t+dt/2
    k2 = v_func_cfg(model, cond, cfg_strength, y + dt*k1/2, tpdto2).detach()
    k3 = v_func_cfg(model, cond, cfg_strength, y + dt*k2/2, tpdto2).detach()
    k4 = v_func_cfg(model, cond, cfg_strength, y + dt*k3, t + dt).detach()
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


@torch.no_grad()
def v_func_cfg(model,          # the flow model
               cond,           # conditioning (can be None)
               cfg_strength,   # classifier-free guidance strength
               t_vec_template, # pre-allocated tensor shape
               x,              # current state
               t,              # current time
               t_scale=999,     # 999 for naive embedding, 1.0 for sophisticated empedding
               debug=False
               ):
    start0 = time.time()
    device, dtype = x.device, x.dtype

    if debug: full_start = time.time()
    #t_vec = torch.full((x.shape[0],), t, device=device, dtype=dtype)
    #t_vec = torch.tensor(t, device=device, dtype=dtype).expand(x.shape[0])
    #t_vec = torch.full((x.shape[0],), t.item(), device=device, dtype=dtype)
    t_vec_template.fill_(t.item())
    t_vec = t_vec_template
    if debug: print(f"  t_vec creation time took {(time.time()-full_start)*1000:.1f}ms")

    if debug: start = time.time()
    #v = model(x, t_vec * t_scale, aug_cond=None, class_cond=cond) # for HDiT
    v = model(x, t_vec * t_scale, cond=cond)
    _ = v.sum().item()  # Force full computation and sync
    if debug: print(f"  v: model eval + sync time took {(time.time()-start)*1000:.1f}ms")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if cond is not None and cfg_strength is not None and cfg_strength != 0:
        if debug: print("v_func_cfg: calling model for uncond ")
        #v_uncond = model(x, t_vec * t_scale, aug_cond=None, class_cond=None)
        v_uncond = model(x, t_vec * t_scale, cond=None)
        v = v_uncond + cfg_strength * (v - v_uncond)

    if debug: print(f"  v_func_cfg time took {(time.time()-start0)*1000:.1f}ms")
    return v



@torch.no_grad()
def generate_latents_rk4(model,          # the flow model 
                         shape,          # (batch_size, channels, height, width)
                         n_steps=50,     # integration steps
                         cond=None,      # conditioning labels/embeddings
                         cfg_strength=3.0, # classifier-free guidance strength
                         debug=False,
                         ):
    """Generate latents using RK4 integration - this 'sampling' routine is primarily used for visualization."""
    gc.collect()
    torch.cuda.empty_cache()

    cond = None # todo remove this
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype 
    if debug: print("in generate_latents_rk4: device, dtype =",device, dtype)
    initial_points = torch.randn(shape, device=device, dtype=dtype)
    current_points = initial_points.clone()
    ts = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)
    t_vec_template = torch.zeros(shape[0], device=device, dtype=dtype)

    # Add this right before your RK4 loop in generate_latents_rk4:
    if debug: print("Testing tensor ops in training context:")
    test_tensor = torch.zeros(4, device=device)
    if debug: start = time.time()
    test_tensor.fill_(0.5)
    if debug: print(f"fill in training context: {(time.time()-start)*1000:.1f}ms")
    
    # Then proceed with your normal RK4 loop

    if warp_time: ts = warp_time(ts)
    v_func = partial(v_func_cfg, model, cond, cfg_strength, t_vec_template)
    for i in range(len(ts)-1):
        if debug: print(f"calling rk4_step, i = {i}")
        if debug: start = time.time()
        current_points = rk4_step(v_func, current_points, ts[i], ts[i+1]-ts[i])
        if debug: end = time.time()
        if debug: print(f"rk4_step took {(end-start)*1000:.1f}ms")
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
    gc.collect()
    torch.cuda.empty_cache()

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
                     debug=True,
                     device=None):     # torch device (auto-detected if None)
    """Generate latents using specified method"""
    if device is None: device = next(model.parameters()).device
    if method == "rk45":
        return generate_latents_rk45(model, shape, device, cond, cfg_strength)
    else:  # default to rk4
        if debug: print("generate_latents: calling rk4 integrator")
        return generate_latents_rk4(model, shape, n_steps, cond, cfg_strength)


def decode_latents(codec,          # the codec/autoencoder model
                   latents,        # latent tensors to decode
                   is_midi=False,  # whether this is MIDI data
                   keep_gray=False): # keep grayscale format for MIDI
    """Decode latents to images"""
    try:
        device = next(codec.parameters()).device
    except: 
        device = latents.device
    decoded = codec.decode(latents.to(device))
    return g2rgb(decoded, keep_gray=keep_gray) if is_midi else decoded


@torch.no_grad()
def sampler(model, codec, method='rk4', batch_size=256, n_steps=100,
            cond=None, n_classes=0, latent_shape=(4,16,16), cfg_strength=3.0,
            is_midi=False, keep_gray=False):
    """generates predicted latents and decodes them"""
    shape = (batch_size,) + latent_shape

    if cond is None and n_classes > 0:  # grid where each column is a single class (10 columns)
        cond = torch.randint(n_classes, (10,)).repeat(batch_size // 10).to(next(model.parameters()).device)
    elif cond is not None:
        cond = cond[:batch_size]  # only grab what we can show

    pred_latents, nfe = generate_latents(model, shape, method, n_steps, cond, cfg_strength)

    decoded_pred = decode_latents(codec, pred_latents, is_midi, keep_gray)
    return pred_latents, decoded_pred, nfe



@torch.no_grad()
def compute_sample_metrics(pred_latents,  target_latents,    # latent space
                           decoded_pred,   decoded_target):  # pixel space
    """Compute metrics between predicted and target samples"""
    batch_size = min(pred_latents.shape[0], target_latents.shape[0])
    print("compute_sample_metrics: using batch_size = ",batch_size)
    
    # Latent space metrics
    sinkhorn_latent = sinkhorn_loss(target_latents[:batch_size], pred_latents[:batch_size])
    
    sinkhorn_px = sinkhorn_loss(decoded_target, decoded_pred)  # pixel space
    fid_px = fid_score(decoded_target, decoded_pred)  # pixel space

    metrics = { 'metrics/sinkhorn': sinkhorn_latent, 'metrics/sinkhorn_px': sinkhorn_px, 'metrics/FID_px': fid_px }
    print("metrics =",metrics)
    return metrics


@torch.no_grad()
def evaluate_model(model,            # the flow model to evaluate
                   codec,            # codec for decoding
                   epoch,            # current epoch (For logging)
                   target_latents,   # target samples for metrics 
                   target_labels=None,   # labels for conditioning (optional)
                   batch_size=256,   # number of samples to generate
                   n_classes=0,      # number of classes for conditioning
                   latent_shape=(4,16,16), # shape of latent space
                   method='rk4',     # integration method
                   n_steps=100,       # number of integration steps
                   is_midi=False,    # whether this is MIDI data
                   keep_gray=False,  # keep grayscale format
                   tag="",           # extra naming tag that can be added
                   cb_tracker=None,  # extra work tracking codebook usage
                   cfg_strength=3.0, # classifier-free guidance strength
                   output_dir="./",
                   use_wandb=True,   # send outputs to wandb
                   pre_encoded=True, # note for how we're getting our data
                   debug=False):
    """Generate samples and compute metrics for model evaluation"""
    model.eval(), codec.eval()
    gc.collect()
    torch.cuda.empty_cache()

    latent_shape = target_latents.shape[-3:]
    shape = (batch_size,) + latent_shape
    if debug: print("evaluate_model: shape =",shape)
    
    pred_latents, decoded_pred, nfe = sampler(model, codec,method=method, batch_size=batch_size,
                                                 cond=target_labels, n_classes=0, latent_shape=latent_shape, cfg_strength=cfg_strength,
                                                 is_midi=is_midi, keep_gray=keep_gray)
    decoded_target = decode_latents(codec, target_latents[:batch_size], is_midi, keep_gray) 

    metrics = compute_sample_metrics(pred_latents, target_latents, decoded_pred, decoded_target)
    if cb_tracker is not None:
        if debug: print("\nevaluate_model: performing codebook analysis. target_latents.device =",target_latents.device) 
        # Track target latents
        z_compressed = target_latents.permute(0, 2, 3, 1).reshape(-1, target_latents.shape[1])
        z_q, indices, _ = codec.vq(z_compressed)
        cb_tracker.update_counts("val", indices)
        
        if debug: print("now tracking pred latents") 
        # Track predicted latents  
        z_compressed = pred_latents.permute(0, 2, 3, 1).reshape(-1, pred_latents.shape[1])
        z_q, indices, _ = codec.vq(z_compressed)
        cb_tracker.update_counts("gen", indices)

        if debug: print("now calling cb_tracker.analyze")
        cb_tracker.analyze(codec, epoch, use_wandb=use_wandb)
        if debug: print("back from analyze_codebooks")

    images = {'pred_latents':pred_latents, 'target_latents':target_latents, 'decoded_pred':decoded_pred, 'decoded_target':decoded_target}

    # Save outputs
    for key, val in images.items():
        #if debug: print(f"   evaluate_model: saving outputs: {key}") 
        save_img_grid(val.cpu(), epoch, nfe, tag=f"{tag}{key}_{method}_{nfe}", use_wandb=use_wandb, output_dir=output_dir)

    # Log metrics
    if use_wandb and metrics:
        metrics['epoch'] = epoch
        wandb.log(metrics)

    gc.collect()
    torch.cuda.empty_cache()

    model.train()
    return metrics
