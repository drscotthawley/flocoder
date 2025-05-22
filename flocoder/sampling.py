import torch
import wandb
from scipy import integrate  # This is CPU-only
from .viz import imshow, save_img_grid
from .metrics import sinkhorn_loss, fid_score


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def warp_time(t, dt=None, s=.5):
    """Parametric Time Warping: s = slope in the middle.
        s=1 is linear time, s < 1 goes slower near the middle, s>1 goes slower near the ends
        s = 1.5 gets very close to the "cosine schedule", i.e. (1-cos(pi*t))/2, i.e. sin^2(pi/2*x)"""
    if s<0 or s>1.5: raise ValueError(f"s={s} is out of bounds.")
    tw = 4*(1-s)*t**3 + 6*(s-1)*t**2 + (3-2*s)*t
    if dt:                           # warped time-step requested; use derivative
        return tw,  dt * 12*(1-s)*t**2 + 12*(s-1)*t + (3-2*s)
    return tw


def rk45_sampler(model, shape, device, eps=0.001, n_classes=0, cfg_scale=3.0):
    """Runge-Kutta '4.5' order method for integration. Source: Tadao Yamaoka"""
    rtol = atol = 1e-05
    model.eval()
    # Create a grid where each column is a single class (10 columns)
    cond = None # the conditioning signal to the model
    if n_classes > 0:
        cond = torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device)
    
    with torch.no_grad():
        # The rest remains the same
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            #t = warp_time(t) # TODO: we should enable this; not sure why it's off
            vec_t = torch.ones(shape[0], device=x.device) * t

            # classifier-free guidance
            v_cond = model(x, vec_t * 999, cond)
            v_uncond = model(x, vec_t * 999, None)
            velocity = v_uncond + cfg_scale * (v_cond - v_uncond) # cfg_scale ~= conditioning strength

            return to_flattened_numpy(velocity)

        # Rest of the implementation unchanged
        solution = integrate.solve_ivp(
            ode_func, (eps, 1), to_flattened_numpy(x),
            rtol=rtol, atol=atol, method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)
        model.train()
        return x, nfe



@torch.no_grad()
def sampler(model, codec, epoch, method, device, sample_N=None, batch_size=100, tag="", 
         images=None, use_wandb=True, output_dir="output", n_classes=0, 
         latent_shape=(4,16,16),
         target=None,  # target is in latent space, not pixel space
         ):
    # TODO: clean up:  through many quick additions, this whole thing has become janky AF
    """Evaluate model by generating samples with class conditioning"""
    model.eval()
    # Use a batch size that's a multiple of 10 to ensure proper grid layout
    # For a 10x10 grid, use batch_size = 100
    batch_size=100 # hard code this for the image display; ignore other batch size values
    shape = (batch_size,)+latent_shape

    if images is None: # TODO: rename, somehow, "images" may actually be latents
        if method == "euler":
            images, nfe = euler_sampler(model, shape=shape, sample_N=sample_N, device=device, n_classes=n_classes)
        elif method == "rk45":
            images, nfe = rk45_sampler(model, shape=shape, device=device, n_classes=n_classes)
    else:
        nfe=0

    decoded_images = codec.decode(images.to(device))
        
    save_img_grid(images.cpu(), epoch, method, nfe, tag=tag, use_wandb=use_wandb, output_dir=output_dir)
    save_img_grid(decoded_images.cpu(), epoch, method, nfe, tag=tag+"decoded_", use_wandb=use_wandb, output_dir=output_dir)

    # metrics on generated outputs:
    if target is not None:
        wandb_log_dict = {}
        sinkhorn = sinkhorn_loss(target[:batch_size], images, device=device)
        #fid = 0.0 # fid_score(target[:batch_size], images) # fid assumes 3 channels, which latents typically won't have
        with torch.no_grad():
            decoded_targ = codec.decode(target.to(device))
        sinkhorn_px = sinkhorn_loss(decoded_targ[:batch_size], decoded_images, device=device)  # pixel space
        fid_px = fid_score(decoded_targ[:batch_size], decoded_images)  # pixel space
        if use_wandb:
            wandb_log_dict = wandb_log_dict | {'epoch':epoch, 'metrics/sinkhorn': sinkhorn}  #, 'metrics/FID': fid }
            wandb_log_dict = wandb_log_dict | {'metrics/sinkhorn_px': sinkhorn_px, 'metrics/FID_px': fid_px }
            wandb.log(wandb_log_dict)

    return model.train()



