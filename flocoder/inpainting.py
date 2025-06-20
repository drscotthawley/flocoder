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


import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import random
import cv2 



########################  torch routines for mask encoding   ##################################


class DownsampleBlock(nn.Module):
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

    def forward(self, x):  # x is either the pixel-space mask, or at least the previous hard-shrunk mask is on channel zero
        mask = x[:, 0:1, :, :]
        skip = self.hard_shrink(mask)
        learned = F.silu(self.conv(x))
        return torch.cat([skip, learned], dim=1)


class MaskEncoder(nn.Module):
    """Inpainting in latent space (where the space doesn't necessarily follow the spatial structure
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
    """
    def __init__(self, output_channels=4, shrink_fac=4, mode='pool'):
        # note that the 128's and 8's are not hard-coded, this just shrinks by a factor of shrink_fac^2
        super().__init__()
        self.layers = nn.Sequential(
            DownsampleBlock(1, 16,  shrink_fac, mode),    # 1 -> 17 channels
            DownsampleBlock(17, 32, shrink_fac, mode),    # 17 -> 33 channels
            nn.Conv2d(33, output_channels, 1)             # 33 -> 4 channels
        )

    def forward(self, mask):  # 1x128x128
        if mask.dtype in [torch.uint8, torch.int32, torch.int64, torch.bool]: mask = mask.float()
        return self.layers(mask)  # 4x8x8




###################   Data and data-augmentation routines   ##########################



def simulate_brush_stroke(size=(128,128), brush_size=5):
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


def generate_mask(size=(128,128), device='cpu'): 
    """Ideally we want something that resembles human-drawn "brush strokes" with a circular cross section"""
    if np.random.rand() < 0.7: 
        brush_size = np.random.randint(1, 10)
        mask = simulate_brush_stroke(size, brush_size)
    else:  # other kinds of denoising masks  for robustness
        # placeholder code stolen from https://github.com/annegnx/PnP-Flow/blob/main/demo/demo.ipynb
        mask = np.random.rand(size) > 0.7 # Create a random mask with 70% missing pixels

    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return mask_tensor


# data:
def create_inpainting_triplet(full_image):
    """Create (source_latents, mask_pixel, target_latents) triplet"""
    target_latents = codec.encode(full_image)

    mask_pixels = generate_mask(full_image.shape[-2:])  # 1x128x128

    incomplete_image = full_image * (1 - mask_pixels)  # Zero out masked regions
    source_latents = codec.encode(incomplete_image)

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





#################################   testing    ########################################

# TODO move this to some kind of dedicated test suite

if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from functools import partial

    # Test both modes
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

    print("\nAll tests completed successfully!")
