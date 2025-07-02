# This is a set of utilties that are all related by the task of training and operating a
# conditional inpainting flow model.
#
# The main velocity model in unet.py has been modified to support "mask_cond"
#
# Routines in this file will be called from elsewhere but support the following capabilities:
# - Encode the mask from pixel space into an embedding to use as a conditioning signal. 
# - When we pre-encode (training) data, we need it as "triplets":
#      1. encode the full image to latent space to serve as the target data
#      2. generate some (random) mask in pixel space  (don't try to encode the mask yet)
#      3. remove the pixels in the image "under" the mask, and encode to latent space to serve as the source

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import random
from PIL import Image
from torch.utils.data import IterableDataset


####################### implementation of CMU/Meta Algorithm 3 for inpainting ######
import torch

def dump_plot(Y_pred, Y):
    """Plot Y_pred vs Y to visualize linearity"""
    import matplotlib.pyplot as plt

    # Flatten and move to CPU for plotting
    y_pred_flat = Y_pred.detach().cpu().flatten()
    y_flat = Y.detach().cpu().flatten()

    # Subsample for plotting (too many points otherwise)
    indices = torch.randperm(len(y_flat))[:10000]  # random 10k points

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred_flat[indices], y_flat[indices], alpha=0.3, s=1)
    plt.xlabel("Y_pred (Linear Approximation)")
    plt.ylabel("Y (True Values)")
    plt.title("Linearity Check: Y_pred vs Y")
    plt.plot([y_flat.min(), y_flat.max()], [y_flat.min(), y_flat.max()], 'r--', alpha=0.8)  # perfect line
    plt.savefig("ypred_vs_y.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved linearity plot to ypred_vs_y.png")

def dump_raw_plot(y_L, x1_L):
    """Plot y_L vs x1_L to see raw relationship before linear approximation"""
    import matplotlib.pyplot as plt
    
    # Flatten and move to CPU for plotting
    y_flat = y_L.detach().cpu().flatten()
    x1_flat = x1_L.detach().cpu().flatten()
    
    # Subsample for plotting
    indices = torch.randperm(len(y_flat))[:10000]  # random 10k points
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x1_flat[indices], y_flat[indices], alpha=0.3, s=1)
    plt.xlabel("x1_L (Target/Clean)")
    plt.ylabel("y_L (Source/Masked)")
    plt.title("Raw Relationship: y_L vs x1_L")
    plt.savefig("yL_vs_x1L.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved raw relationship plot to yL_vs_x1L.png")


def approx_AL(source, target, debug=True):  # try to get the effect of pixel-wise mask but in latent space

    y_L, x1_L = source, target
    X = x1_L.view(x1_L.shape[0], -1)  # [2048, 256]
    Y = y_L.view(y_L.shape[0], -1)    # [2048, 256]

    # Solve: Y = X @ A_L.T  =>  A_L: [256, 256]
    A_L = torch.linalg.lstsq(X, Y).solution.T  # [256, 256]

    if debug:
        Y_pred = X @ A_L.T  # Batch matrix multiply
        mse = torch.mean((Y_pred - Y)**2)  # True MSE
        rel_error = torch.norm(Y_pred - Y) / torch.norm(Y)  # Relative error
        print(f"Reconstruction MSE: {mse:.6f}, Relative error: {rel_error:.3f}")

        if not hasattr(approx_AL, 'made_plot'):
            approx_AL.made_plot = True
            dump_plot(Y_pred, Y)
            dump_raw_plot(y_L, x1_L)

    return A_L



def algorithm3(v,           # velocity field output from pretrained model at (x, tp)
               x,           # current state x_t during ODE integration  
               t,           # initial time (fixed, for conditional OT path params)
               tp,          # current integration time t'
               y,           # noisy measurement (observed data)
               A,           # measurement matrix (1-mask for inpainting)
               sigma_y=0.05, # measurement noise std dev (0 for noiseless)
               gamma_t=1.0): # adaptive weights (1 = unadaptive)
    """Algorithm 3: Training-free approach to solve inverse problems via flows with pretrained vector field"""
    # Step 5: r²_t' = σ²_t' / (σ²_t' + α²_t')  
    # For conditional OT: α_t = t, σ_t = 1-t
    r_tp_sq = (1 - tp)**2 / (tp**2 + (1 - tp)**2)
    
    # Step 6: Convert vector field to x̂₁
    # x̂₁ = (α_t d ln(α_t/σ_t)/dt)⁻¹ (v̂ - d ln σ_t/dt x_t)
    # For conditional OT: d ln(α_t/σ_t)/dt = 1/(t(1-t)), d ln σ_t/dt = -1/(1-t)
    alpha_t, sigma_t = tp, 1 - tp
    d_ln_ratio_dt = 1 / (tp * (1 - tp))  # d ln(α_t/σ_t)/dt
    d_ln_sigma_dt = -1 / (1 - tp)        # d ln σ_t/dt
    
    coeff_inv = 1 / (alpha_t * d_ln_ratio_dt)  # (α_t d ln(α_t/σ_t)/dt)⁻¹
    x1_hat = coeff_inv * (v - d_ln_sigma_dt * x)
    
    # Step 7: ΠGDM correction 
    # g = (y - Ax̂₁)ᵀ (r²_t' AAᵀ + σ²_y I)⁻¹ A ∂x̂₁/∂x_t'
    residual = y - A @ x1_hat.flatten()  # flatten for matrix ops
    cov_matrix = r_tp_sq * (A @ A.T) + sigma_y**2 * torch.eye(A.shape[0], device=x.device)
    
    # Assuming ∂x̂₁/∂x_t' ≈ I (identity) for simplicity
    # In practice, this would require autograd for exact gradient
    g_flat = residual @ torch.linalg.solve(cov_matrix, A)
    g = g_flat.view_as(x)  # reshape back to image dimensions
    
    # Step 8: Correct unconditional vector field  
    # v̂_corrected = v̂ + σ²_t d ln(α_t/σ_t)/dt γ_t g
    correction_coeff = sigma_t**2 * d_ln_ratio_dt * gamma_t
    v_corrected = v + correction_coeff * g
    
    return v_corrected



########################  torch routines for mask encoding   ##################################

def mysigmoid(x, eps=0.01):
    # outputs from [-eps, 1+eps], may help avoid saturation while still basically being a sigmoid
    return F.sigmoid(x) * (1 + 2*eps) - eps


class DownsampleBlock_small(nn.Module):
    """ helper for MaskEncoder, below.
    shrinks by a factor of shrink_fac, includes concatenative skip connection of 'hard'/non-learnable
    interp or pooling"""
    def __init__(self, in_channels, out_channels, shrink_fac=4, mode='pool'):
        super().__init__()  
        self.conv = nn.Conv2d(in_channels, out_channels, shrink_fac, stride=shrink_fac)

        if mode == 'pool':
            self.hard_shrink = nn.AvgPool2d(kernel_size=shrink_fac, stride=shrink_fac)
        else:
            self.hard_shrink = partial(F.interpolate, scale_factor=1.0/shrink_fac, mode='bilinear')

    def forward(self, x):  # x is either pixel-space mask or at least the previous hard-shrunk mask is on channel zero
        mask = x[:, 0:1, :, :]
        skip = self.hard_shrink(mask)
        learned = F.silu(self.conv(x))
        return torch.cat([skip, learned], dim=1)


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shrink_fac=4, mode='pool'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, shrink_fac, stride=shrink_fac)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)  

        if mode == 'pool':
            self.hard_shrink = nn.AvgPool2d(kernel_size=shrink_fac, stride=shrink_fac)
        else:
            self.hard_shrink = partial(F.interpolate, scale_factor=1.0/shrink_fac, mode='bilinear')

    def forward(self, x):
        mask = x[:, 0:1, :, :]
        skip = self.hard_shrink(mask)
        learned = F.silu(self.conv1(x))
        learned = F.silu(self.conv2(learned))  
        return torch.cat([skip, learned], dim=1)


class MaskEncoder(nn.Module):
    """Long docstring! 
    Inpainting in latent space (where the space doesn't necessarily follow the spatial structure
    of pixel space) requires that the inpainting mask get encoded somehow from pixel space to...
    if not latent space itself then at least some embedding suitable for use as a conditioning signal
    when training (a mask-conditioned model).

    In this code, we do some aggressive learnable downsampling coupled with 
    standard pooling or interpolation to guard against model/mode collapse.  
    And these are joined via skip connections for efficient gradient propagation.  
    The final result has the same latent shape as our images, but this is not actually a requirement. 
    Just "something" that can get plugged in as a conditioning signal during training.

    We're not opting for any Attention here, and the spatial structure of the Conv operations 
    means we hope we don't need positional encoding. The overall hope is that as long as the mask encodings
    are 'sufficiently unique' that the model is able to learn using them, then it doesn't really
    matter.    ...At least for now! ;-) 

    How it gets used: 
    1. pass in pixel-space mask to MaskEncoder get out mask_latents 
    2. mask_latents get used as conditioning signal for model, combined with time via mask_cond_mlp projection
    3. source data needs noise added to function properly, we suggest this sample code: 
           mixed_source = source_latents + mask_latents * (noise - source_latents)
        Note: this is the same as this alternate (/more common?) formulation: 
           mixed_source = (1-mask_latents)*source_latents + mask_latents*noise
        Such equation(s) seem to assume mask_latents is on [0,1], outside of which may yield unexpected behavior, 
        yet the model may learn better if it has freedom to "color outside the lines".  YMMV.
        Use the kwarg "final_act" to enforce a domain, e.g sigmoid for [0,1]. but note that will also restrict gradients.
        Current default is SILU [-.27, infty], and we hope the model learns what to do ;-) 
    """
    def __init__(self, 
            output_channels=4,  
            shrink_fac=4,      # Shrink per DownSampleBlock of which there are two, so square this
            mode='pool',       # Mode for the hard-shrink.  !='pool' means use F.interpolate
            final_act=F.sigmoid # Activation before output. Intuitively we'd like mask_latents on [0,1] 
                               #   but I won't force it: sigmoid might be too "harsh" ( saturation/vanishing gradients )
                               #   SILU offers a bit o' freedom while keeping values from getting "really negative". 
                               #   You may replace SILU with sigmoid, None, etc, to change this
            ):
        # note that the 128's and 8's are not hard-coded, this just shrinks by a factor of shrink_fac^2
        super().__init__()
        self.layers = nn.Sequential(
            DownsampleBlock(1, 16,  shrink_fac, mode),    # 1 -> 17 channels
            DownsampleBlock(17, 32, shrink_fac, mode),    # 17 -> 33 channels
            nn.Conv2d(33, output_channels-1, 1)            # 33 -> output_channels(-1)
        )
        self.final_act = final_act

        # For the doubly-shrunk mask
        if mode == 'pool':
            self.double_shrink = nn.AvgPool2d(kernel_size=shrink_fac**2, stride=shrink_fac**2)
        else:
            self.double_shrink = partial(F.interpolate, scale_factor=1.0/(shrink_fac**2), mode='bilinear')


    def forward(self, mask_pixels):  # e.g. shape = [batch, 1, 128, 128]
        if mask_pixels.dtype in [torch.uint8, torch.int32, torch.int64, torch.bool]: 
            mask_pixels = mask_pixels.float()
        learned_features = self.layers(mask_pixels)  # e.g, shape of [batch, 4, 8, 8]
        if self.final_act is not None: 
            learned_features = self.final_act(learned_features)
        
        doubly_shrunk = self.double_shrink(mask_pixels)
        mask_latents = learned_features
        mask_latents = torch.cat([doubly_shrunk, learned_features], dim=1)   # "red"  channel will show doubly-shunk mask, other 3 channels are learned features
        return mask_latents


################### blending source and noise via mask 

def mask_blending(source, mask, noise=None):
    if noise is None: noise = torch.randn_like(source)
    source = source + mask*(noise - source)
    return source


###################   Data and data-augmentation routines   ##########################



def simulate_brush_stroke_cv2(size=(128,128), brush_size=5):
    import cv2 

    # caled by generate_mask, below
    mask = np.zeros(size)
    # Random walk with varying brush size
    x, y = np.random.randint(20, size[0]-20), np.random.randint(20, size[1]-20)
    stroke_length = np.random.randint(50, 200)
    for _ in range(stroke_length):
        # Add some randomness to brush movement
        dx, dy = np.random.normal(0, 2, 2)
        x, y = np.clip([x+dx, y+dy], [0, 0], [size[0]-1, size[1]-1])
        cv2.circle(mask, (int(x), int(y)), brush_size + np.random.randint(-2, 3), 1, -1)
    return mask



def simulate_brush_stroke(size=(128,128), num_strokes=1, brush_size=None, max_brush_size=15):
    mask = np.zeros(size)
    for s in range(num_strokes):
        bs = brush_size if brush_size is not None else np.random.randint(3, max_brush_size)
        x = np.random.randint(0, size[0]) 
        y = np.random.randint(size[1]//3, 2*size[1]//3)  # start around the middle 2/3 vertically
        stroke_length = np.random.randint(100, 300)
        direction = np.random.uniform(-np.pi/10, np.pi/10)
        if x > size[0]/2: direction += np.pi  # if starting in the left, head right, and vice versa
        dir_change_std = 0.04  # in radians. 
        for _ in range(stroke_length):
            direction += np.random.normal(0, dir_change_std)
            dx, dy = np.cos(direction) * 0.7, np.sin(direction) * 0.7
            new_x, new_y = x+dx, y+dy
            if new_x < 0 or new_x >= size[0] or new_y < 0 or new_y >= size[1]: break # stop when we go off the edge

            x, y = new_x, new_y
            #current_brush = max(1, bs + np.random.randint(-4, 4))
            current_brush = max(1, bs + np.random.randint(-bs//2, bs//2))  # Huge variation
            x_int, y_int, r = int(x), int(y), current_brush + 1
            y_min, y_max, x_min, x_max = max(0,y_int-r), min(size[0],y_int+r+1), max(0,x_int-r), min(size[1],x_int+r+1)
            yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
            mask[y_min:y_max, x_min:x_max][(xx - x_int)**2 + (yy - y_int)**2 <= current_brush**2] = 1
    return mask


def generate_rectangles(size=(128,128), max_size_ratio_x=0.8, max_size_ratio_y=0.3):
    mask = np.zeros(size)
    max_w, max_h = int(size[0] * max_size_ratio_x), int(size[1] * max_size_ratio_y)
   
    min_rects, max_rects = 2, 10   # most rects will "miss" the notes anyway, so more is good
    min_rect_dim = 3
    for _ in range(np.random.randint(min_rects, max_rects+1)):
        w =  np.random.randint(min_rect_dim, max_w)
        h =  np.random.randint(min_rect_dim, max_h)
        x = np.random.randint(0, size[0] - w)
        y = np.random.randint(0, size[1] - h)  # not much need to mask outsize the middle
        mask[x:min(size[0], x+w), y:min(size[1], y+h)] = 1
    return mask.T # .T bc numpy has x & y reversed


_status_msg = ''
def generate_mask(size=(128,128), 
        mask_type = '',  # can specify a mask algorithm name or else it'll be randomly chosen according to choices & p
        choices = ['total', 'brush', 'rectangles', 'noise', 'nothing'],  # names of different kinds of masks  to choose from
        p=        [ 0.4,    0.35,      0.15,         0.05,    0.05],      # probabilities for each kind of mask
        #p=        [ 0.01,    0.02,      0.87,         0.05,    0.05],      # probabilities for each kind of mask
        to_tensor=True, device='cpu', debug=False):
    """Ideally we want something that resembles human-drawn "brush strokes" with a circular cross section"""
    global _status_msg 

    if _status_msg == '':  # one-time status message for first call
        _status_msg = f"generate_mask: choices = {choices}, p={p}"
        print(_status_msg)   

    if mask_type == '':  
        mask_type = np.random.choice(choices, p=p)
    if debug: print("mask_type = ",mask_type)

    if mask_type == 'total': # all mask = unconditional generation
        mask = np.ones(size)
    elif mask_type == 'brush':
        min_strokes, max_strokes = 2, 5
        mask = simulate_brush_stroke(size, num_strokes=np.random.randint(min_strokes, max_strokes+1))
    elif mask_type == 'rectangles':
        mask = generate_rectangles(size)
    elif mask_type == 'noise': # no mask = no-op, no generation = boring, target=source
        mask = np.random.rand(*size) > 0.7
    elif mask_type == 'nothing': # no mask = no-op, no generation = boring
        mask = np.zeros(size)
    else: 
        raise ValueError(f"Unsupported mask_type: {mask_type}")

    if to_tensor: return torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return mask



def generate_mask_batch(size=(128,128), batch_size=1, unique_masks=False, to_tensor=True, **kwargs):
    #### NOTE: better to use InptainingDataset, below
    """Generate masks for batch processing"""
    if unique_masks and batch_size > 4:  # Parallelize for larger batches
        from multiprocessing import Pool
        from functools import partial
        with Pool() as pool:
            func = partial(generate_mask, size=size, to_tensor=False, **kwargs)
            masks = pool.map(func, [None] * batch_size)
        mask = np.stack(masks, axis=0)
    elif unique_masks:
        masks = [generate_mask(size, to_tensor=False, **kwargs) for _ in range(batch_size)]
        mask = np.stack(masks, axis=0)
    else:
        mask = np.tile(generate_mask(size, to_tensor=False, **kwargs)[None, ...], (batch_size, 1, 1))

    if to_tensor:
        device = kwargs.get('device', 'cpu')
        return torch.tensor(mask, dtype=torch.float32).unsqueeze(1).to(device)  # (batch_size, 1, H, W)
    return mask


# data:
def create_inpainting_triplet(full_image, codec, quantize=False):
    """Create (source_latents, mask_pixel, target_latents) triplet"""
    target_latents = codec.encode(full_image)

    mask_pixels = generate_mask(full_image.shape[-2:])  # 1x128x128

    incomplete_image = full_image * (1 - mask_pixels)  # Zero out masked regions
    source_latents = codec.encode(incomplete_image)
    if quantize: 
        source_latents, target_latents = codec.quantize(source_latents), codec.quantize(target_latents)

    return source_latents, mask_pixels, target_latents

## In preencode_data.py main loop, do something like this? 
#for batch in dataloader:
#    if random.random() < 0.5:  # 50% inpainting data
#        source_latents, mask_pixel, target_latents = create_inpainting_triplet(batch)
#        torch.save({
#            'source': source_latents,
#            'mask': mask_pixel,
#            'target': target_latents,
#            'type': 'inpainting'
#        }, save_path)
#    else:  # 50% standard generation data
#        target_latents = codec.encode(batch)
#        torch.save({
#            'target': target_latents,
#            'type': 'generation'
#        }, save_path)




class InpaintingDataset(IterableDataset):
    """Dataset that generates masks and creates masked images on-the-fly"""
    def __init__(self, base_dataset, mask_kwargs=None):
        self.base_dataset = base_dataset
        self.mask_kwargs = mask_kwargs or {}
        if hasattr(base_dataset, 'actual_len'): 
            print("InpaintingDataset: setting self.actual_len") 
            self.actual_len = base_dataset.actual_len
        else: 
            print("InpaintingDataset: base_datset has no attribute 'actual_len'")
    
    def __iter__(self):
        for item in self.base_dataset:
            if isinstance(item, tuple):
                full_image, label = item[0], item[1] if len(item) > 1 else 0
            else:
                full_image, label = item, 0
                
            # Generate mask for this specific item
            size = full_image.shape[-2:] if hasattr(full_image, 'shape') else (128, 128)
            mask_pixels = generate_mask(size=size, to_tensor=True, device=full_image.device, **self.mask_kwargs)
            
            # Create masked source image
            source_image = full_image * (1 - mask_pixels.squeeze())
            
            yield {
                'source_image': source_image,
                'mask_pixels': mask_pixels.squeeze(), 
                'target_image': full_image,
                'label': label
            }




#################################   testing    ########################################

# TODO move this to some kind of dedicated test suite

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from functools import partial

    # Mask Encoding: Test both modes 
    for mode in ['pool', 'bilinear']:
        print(f"\nTesting MaskEncoder with mode='{mode}'")

        # Create model
        encoder = MaskEncoder(output_channels=4, shrink_fac=4, mode=mode)

        # Create random binary mask (like real use case)
        mask = torch.randint(0, 2, (2, 1, 128, 128))  # batch=2, binary values
        print(f"Input mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Mask value range: {mask.min().item():.1f} to {mask.max().item():.1f}")

        # Forward pass
        output = encoder(mask)
        print(f"Output shape: {output.shape}")
        print(f"Output value range: {output.min().item():.3f} to {output.max().item():.3f}")

        # Verify expected downsampling: 128 -> 32 -> 8
        expected_spatial = 128 // (4 ** 2)  # shrink_fac^2
        assert output.shape == (2, 4, expected_spatial, expected_spatial), f"Expected (2,4,{expected_spatial},{expected_spatial}), got {output.shape}"

        print("✓ Test passed!")


    print("\nGenerating mask variations...")
    masks = []
    grid_size = 10 # number of images along one edge
    mask_size = 128
    for i in range(grid_size**2):  # number of images
        mask = generate_mask(size=(mask_size, mask_size), device='cpu', debug=True).squeeze().numpy()  # Remove batch/channel dims
        masks.append((mask * 255).astype(np.uint8))  # Convert to 0-255 range
    
    print("Packaging mask variations as an  grid")
    grid_img = np.zeros((grid_size * mask_size, grid_size * mask_size), dtype=np.uint8)
    
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            y_start, y_end = i * mask_size, (i + 1) * mask_size
            x_start, x_end = j * mask_size, (j + 1) * mask_size
            grid_img[y_start:y_end, x_start:x_end] = masks[idx]
    
    # Save as PNG
    filename = "../images/mask_gen_test.png"
    Image.fromarray(grid_img, mode='L').save(filename)
    print(f"✓ Saved mask grid to {filename}")
    print("All tests completed successfully!")

