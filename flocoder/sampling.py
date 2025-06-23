import torch
import numpy as np
import wandb
from scipy import integrate  # This is CPU-only
from .viz import save_img_grid
from .general import print_vram
from .metrics import g2rgb, compute_sample_metrics
from functools import partial
import time
import gc
from torchvision.transforms import ToTensor
import random



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
    k1 = f(y, t)
    tpdto2 = t+dt/2
    k2 = f(y + dt*k1/2, tpdto2)
    k3 = f(y + dt*k2/2, tpdto2)
    k4 = f(y + dt*k3, t + dt)
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
    t_vec_template.fill_(t.item())
    t_vec = t_vec_template
    #v = model(x, t_vec * t_scale, aug_cond=None, class_cond=cond) # for HDiT
    v = model(x, t_vec * t_scale, cond=cond)
    _ = v.sum().item()  # Force full computation and sync before moving on
    if torch.cuda.is_available():  # just added to avoid killing vram
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if cond and cond.get('class_cond') is not None and cfg_strength:
        #v_uncond = model(x, t_vec * t_scale, aug_cond=None, class_cond=None) # for HDiT
        cond_no_class = cond.copy()
        cond_no_class['class_cond'] = None
        v_no_class = model(x, t_vec * t_scale, cond=cond_no_class)
        v = v_no_class + cfg_strength * (v - v_no_class)

    return v

@torch.no_grad()
def generate_latents_rk4(model,          # the flow model 
                         shape,          # (batch_size, channels, height, width)
                         n_steps=50,     # integration steps
                         cond=None,      # conditioning signals (dict)
                         cfg_strength=3.0, # classifier-free guidance strength
                         source=None,    # source data is noise unless otherwise specified
                         init_latents=None,  # "init image" usage which is not the same as conditional inpainting
                         init_strength=0.0, # integration will start at time 1-init_strength
                         jitter_strength=0, # 0.001, # to make it a bit diffusion-y
                         debug=False,
                         ):
    """Generate latents using RK4 integration - this 'sampling' routine is primarily used for visualization."""
    if debug: print("generate_latents_rk4:start, init_latents = ",None if init_latents is None else init_latents.shape)
    gc.collect()
    if torch.cuda.is_available:  # added to avoid CUDA OOM
        torch.cuda.empty_cache()   # try to free up vram

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype 
    if debug: print("in generate_latents_rk4: device, dtype =",device, dtype)

    
    current_points = source if source is not None else torch.randn(shape, device=device, dtype=dtype)   # starting source / randn noise
    if init_latents is None: # normal operation
        ts = torch.linspace(0, 1, n_steps, device=device, dtype=dtype)
        jitter_strength = 0
    else:  # inpainting
        # actually the following logic includes the no init case, but i like to keep it separate for debugging
        if debug: print(f"generate_latents_rk4: doing inpainting with init_latents.shape =",init_latents.shape) 
        current_points = (1 - init_strength) * current_points + init_strength * init_latents # start from linterp between noise & target
        n_steps = max(1, int(n_steps * (1.0 - init_strength)))                               # dont need as many steps
        ts = torch.linspace(init_strength, 1.0, n_steps, device=device, dtype=dtype)         # starting time = init_strength

    if warp_time: ts = warp_time(ts)
    t_vec_template = torch.zeros(shape[0], device=device, dtype=dtype)  # pre-allocate for mem-efficient time-coord broadcast
    v_func = partial(v_func_cfg, model, cond, cfg_strength, t_vec_template)

    if debug: print("generate_latents_rk4: starting integration from t =",ts[0],", n_steps =",n_steps)
    for i in range(len(ts)-1):
        current_points = rk4_step(v_func, current_points, ts[i], ts[i+1]-ts[i])
        if random.random() < 0.1 and jitter_strength > 0:  # occasional jitter
            current_points += torch.randn_like(current_points) * jitter_strength * (1-ts[i])  # Less noise as t→1

    nfe = n_steps * 4  # number of function evaluations, rk4 has 4
    return current_points, nfe




@torch.no_grad()
def generate_latents(model,            # the flow model
                     shape,            # (batch_size, channels, height, width)
                     method='rk4',     # integration method: 'rk4' or 'rk45'
                     n_steps=50,       # number of steps (rk4 only)
                     cond=None,        # conditioning signals (dict)
                     cfg_strength=3.0, # classifier-free guidance strength
                     device=None,     # torch device (auto-detected if None)
                     source=None,     # possible non-gaussian source data
                     init_latents=None,  # used for "init image" style integration
                     init_strength=0.0,
                     debug=False,
                     ):
    """Generate latents using specified method"""
    if device is None: device = next(model.parameters()).device
    if method == "rk45": # bad idea i removed this
        return generate_latents_rk45(model, shape, device, cond, cfg_strength)
    else:  # default to rk4
        if debug: print("generate_latents: calling rk4 integrator")
        return generate_latents_rk4(model, shape, n_steps, cond, cfg_strength, source=source, init_latents=init_latents, init_strength=init_strength)



def _decode_latents(codec,          # the codec/autoencoder model
                   latents,        # latent tensors to decode
                   is_midi=False,  # whether this is MIDI data
                   keep_gray=False,# keep grayscale format for MIDI
                   device=None,
                   debug=False): 
    """Decode latents to images"""
    if debug: print_vram("decode_latents: before decoding")
    if device is None:
        try:
            device = next(codec.parameters()).device
        except: 
            device = latents.device
    if debug: print("decode_latents: device = ",device)
    decoded = codec.decode(latents.to(device))
    if debug: print_vram("decode_latents: after decoding")
    return g2rgb(decoded, keep_gray=keep_gray) if is_midi else decoded


def decode_latents(codec, latents, is_midi=False, keep_gray=False, device=None, chunk_size=128, debug=False):
    """Decode latents in chunks to save VRAM"""
    chunks = []
    if debug: print_vram("decode_latents_chunked: start")
    for i in range(0, latents.shape[0], chunk_size):
        chunk = _decode_latents(codec, latents[i:i+chunk_size], is_midi=is_midi,
                               keep_gray=keep_gray, device=device, debug=debug)
        if debug: print_vram("decode_latents_chunked: after chunk")
        chunks.append(chunk.cpu())
        del chunk
        torch.cuda.empty_cache()
    if debug: print_vram("decode_latents_chunked: after loop, before result")
    result = torch.cat(chunks, dim=0).to(latents.device)
    if debug: print_vram("decode_latents_chunked: after result")
    return result


@torch.no_grad()
def sampler(model, codec, method='rk4', batch_size=256, n_steps=100,
            cond=None, n_classes=0, latent_shape=(4,16,16), cfg_strength=3.0,
            is_midi=False, keep_gray=False, device=None, 
            source=None,   # possible non-
            init_image=None,  # possible  "init image" 
            init_strength=0.0, # integration will start at time 1-init_strength
            debug=False,
            ):
    """generates predicted latents and decodes them"""
    if device is None: device = next(model.parameters()).device
    codec_device = next(codec.parameters()).device
    assert device==codec_device, f"sampler, device mismatch: device = {device}, but  codec_device {codec_device}"

    # code related to inpainting
    init_latents = None
    if init_image is not None: # Encode init image to latents
        if debug: print("sampler: encoding init_image") 
        if isinstance(init_image, str): init_image = Image.open(init_image) 
        # Convert PIL to tensor and encode
        #init_tensor = transforms.ToTensor()(init_image).unsqueeze(0).to(device)
        #init_tensor = torch.tensor(np.array(init_image)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        init_tensor = ToTensor()(init_image).unsqueeze(0).to(device)
        init_latents = codec.encode(init_tensor)
        if init_latents.shape[0] == 1 and batch_size > 1: # Expand to batch size if needed
            init_latents = init_latents.repeat(batch_size, 1, 1, 1)

    shape = (batch_size,) + latent_shape
    if source is not None: source = source[:batch_size]

    if cond.get('class_cond')  is None and n_classes > 0:  # grid where each column is a single class (10 columns)
        cond['class_cond'] = torch.randint(n_classes, (10,)).repeat(batch_size // 10).to(device)
    elif cond.get('class_cond') is not None:
        cond['class_cond'] = cond['class_cond'][:batch_size]  # grab only as many as we can analyze/evaluate/show

    if cond.get('mask_cond') is not None: cond['mask_cond'] = cond['mask_cond'][:batch_size]

    pred_latents, nfe = generate_latents(model, shape, method, n_steps, cond, cfg_strength, device=device, 
            source=source, init_latents=init_latents, init_strength=init_strength)

    if debug: print_vram("sampler: before decode_pred")
    decoded_pred = decode_latents(codec, pred_latents, is_midi, keep_gray, device=device)
    if debug: print_vram("sampler: after decode_pred")
    return pred_latents, decoded_pred, nfe



@torch.no_grad()
def evaluate_model(model,            # the flow model to evaluate
                   codec,            # codec for decoding
                   epoch,            # current epoch (For logging)
                   target_latents,   # target samples for metrics 
                   cond=None,        # conditioning signals, e.g. class labels
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
                   # TODO: ok we've got too many args at this point! 
                   use_wandb=True,   # send outputs to wandb
                   pre_encoded=True, # note for how we're getting our data
                   source=None,      # source data, if any. Otherwise will use randn noise 
                   mask_pixels=None, # only used for visualization/debugging
                   debug=False):
    """Generate samples and compute metrics for model evaluation"""
    model.eval(), codec.eval()
    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()

    latent_shape = target_latents.shape[-3:]
    shape = (batch_size,) + latent_shape
    if debug: 
        print("evaluate_model: shape =",shape)
        print_vram("before calling sampler")  
    pred_latents, decoded_pred, nfe = sampler(model, codec,method=method, batch_size=batch_size,
                                                 cond=cond, n_classes=0, latent_shape=latent_shape, cfg_strength=cfg_strength,
                                                 is_midi=is_midi, keep_gray=keep_gray, source=source,)

    print("Computing sample metrics...") 
    if debug: print_vram("after sampler, before decoding target")
    decoded_target = decode_latents(codec, target_latents[:batch_size], is_midi, keep_gray) 
    if debug: print_vram("after decoding target")
    metrics = compute_sample_metrics(pred_latents, target_latents, decoded_pred, decoded_target)

    if cb_tracker is not None:
        print("Performing codebook analysis. target_latents.device =",target_latents.device) 
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
    if source is not None: 
        decoded_source = decode_latents(codec, source[:batch_size], is_midi, keep_gray)
        images.update({'source_latents':source[:batch_size], 'decoded_source':decoded_source})  

    # Add mask visualization:
    if cond and isinstance(cond, dict):
        if 'mask_cond' in cond and cond['mask_cond'] is not None:
            images['mask_latents'] = cond['mask_cond'][:batch_size]
        if mask_pixels is not None:
            mask_pixels = mask_pixels[:batch_size].float()
            images['mask_pixels'] = mask_pixels

    # Save outputs
    for key, val in images.items():
        #if debug: print(f"   evaluate_model: saving outputs: {key}") 
        save_img_grid(val.cpu(), epoch, nfe, tag=f"{tag}{key}_{method}_{nfe}", use_wandb=use_wandb, output_dir=output_dir)

    if use_wandb and metrics:
        wandb_metrics = {f'metrics/{k}': v for k, v in metrics.items()}
        wandb_metrics['epoch'] = epoch
        wandb.log(wandb_metrics)

    gc.collect()
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()

    model.train()
    return metrics
