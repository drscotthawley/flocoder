import os
import torch 
from torchvision.utils import make_grid, save_image
import wandb
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image



#def denormalize(image_batch, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
def denormalize(image_batch, means=[0.5, 0.5, 0.5], stds=[0.5, 0.5, 0.5]):
    """undoes transforms normalization, use this before displaying output demo images"""
    # Create a deep copy to avoid modifying the original tensor
    image_batch = image_batch.clone().detach()

    # Ensure mean and std are on the same device as image_batch
    means = torch.tensor(means, device=image_batch.device)
    stds = torch.tensor(stds, device=image_batch.device)

    # For batched input with shape [B, C, H, W]
    # Reshape mean and std for proper broadcasting
    if image_batch.dim() == 4:
        means = means.view(1, 3, 1, 1)
        stds = stds.view(1, 3, 1, 1)

    return image_batch * stds + means # Apply inverse normalization

def imshow(img, filename):
    # rescales before saving
    imin, imax = img.min(), img.max()
    img = (img - imin) / (imax - imin) # rescale via max/min
    img = np.clip(img, 0, 1)
    npimg = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if npimg.shape[2] == 1:  # grayscale
        pil_img = Image.fromarray(npimg.squeeze(2), mode='L')
    else:  # RGB
        pil_img = Image.fromarray(npimg)
    pil_img.save(filename)


def save_img_grid(img, epoch, nfe, tag="", use_wandb=True, output_dir="output", show_batch_size=100):
    """Save image grid with consistent 10-column layout to match class conditioning"""
    filename = f"{tag}_epoch_{epoch + 1}_nfe_{nfe}.png"
    # Use nrow=10 to ensure grid columns match our class conditioning
    show_batch_size = min(show_batch_size, img.shape[0])
    img = img[:show_batch_size]
    img_grid = make_grid(img, nrow=10, normalize=True)
    file_path = os.path.join(output_dir, filename)
    #imshow(img_grid, file_path)
    save_image(img, file_path)
    name = f"demo/{tag}"
    if 'euler' in name: name = name + f"_nf{nfe}"
    if wandb.run is not None: wandb.log({name: wandb.Image(img_grid, caption=f"Epoch: {epoch}")})

