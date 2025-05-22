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


def rk45_sampler(model, 
                 shape, 
                 device, 
                 eps=0.001, 
                 cond=None, 
                 cfg_strength=3.0, #  do classifier-free-guidance if != 0 or None
                 ):
    """Runge-Kutta '4.5' order method for integration. Source: Tadao Yamaoka"""
    rtol = atol = 1e-05
    model.eval()

    
    with torch.no_grad():
        # The rest remains the same
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

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
            ode_func, (eps, 1), to_flattened_numpy(x),
            rtol=rtol, atol=atol, method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)
        model.train()
        return x, nfe



@torch.no_grad()
def sampler(model, codec, epoch, 
            method='rk45',                       # keep it on "rk45" for now
            device='cuda', 
            sample_N=None,                     # steps to use in integration. None=use default/adaptive
            batch_size=100,                    # number of (source) data points to integrate
            tag="", 
            images=None, 
            use_wandb=True, 
            output_dir="output", 
            n_classes=0, 
            latent_shape=(4,16,16),
            target=None,  # target is in latent space, not pixel space
            target_labels=None,  # labels for those targets, i.e. cond
            ):
    # TODO: clean up:  through many quick additions, this whole thing has become janky AF
    """Evaluate model by generating samples with class conditioning"""
    model.eval()
    # Use a batch size that's a multiple of 10 to ensure proper grid layout
    # For a 10x10 grid, use batch_size = 100
    batch_size=100 # hard code this for the image display; ignore other batch size values
    shape = (batch_size,)+latent_shape

    cond = target_labels
    if cond==None and n_classes > 0:  # grid where each column is a single class (10 columns)
        cond = torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device)
    elif cond is not None:
        cond = cond[:batch_size]  # only grab what we can show
        # TODO: get indices to sort cond in ascending order, and sort cond using those indices
        # if target is not none: use the same indices to sort target

    nfe = 0 
    if images is None: # TODO: rename, somehow, "images" may actually be latents
        tag = tag+method
        if method == "euler":
            images, nfe = euler_sampler(model, shape=shape, sample_N=sample_N, device=device, cond=cond)
        elif method == "rk45":
            images, nfe = rk45_sampler(model, shape=shape, device=device, cond=cond)

    save_img_grid(images.cpu(), epoch, nfe, tag=tag, use_wandb=use_wandb, output_dir=output_dir)
    decoded_images = codec.decode(images.to(device))
    save_img_grid(decoded_images.cpu(), epoch, nfe, tag="decoded_"+tag, use_wandb=use_wandb, output_dir=output_dir)

    # metrics on generated outputs:
    metrics_dict = {}
    if target is not None:
        sinkhorn = sinkhorn_loss(target[:batch_size], images, device=device)
        # note: can't do FID on latents unless latents have 3 channels  (or maybe just grab first 3 channels)
        with torch.no_grad():  # TODO: this shouldn't be needed
            decoded_targ = codec.decode(target.to(device))
        sinkhorn_px = sinkhorn_loss(decoded_targ[:batch_size], decoded_images, device=device)  # pixel space
        fid_px = fid_score(decoded_targ[:batch_size], decoded_images)  # pixel space

        metrics_dict = {'epoch':epoch, 'metrics/sinkhorn': sinkhorn, 'metrics/sinkhorn_px': sinkhorn_px, 'metrics/FID_px': fid_px }
        if use_wandb: wandb.log(metrics_dict)

    model.train()
    return metrics_dict 



