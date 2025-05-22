import os
import torch 
from torchvision.utils import make_grid
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



def viz_codebooks(model, epoch, no_wandb=False): # RVQ

    # Extract codebook vectors from all levels
    codebook_vectors = [codebook.detach().cpu().numpy() 
                       for codebook in model.vq.codebooks]
    
    # Create two figures - one for codebook visualizations and one for histograms
    # First figure: Codebook visualizations
    fig1, axs1 = plt.subplots(model.codebook_levels, 1, 
                             figsize=(16, 4*model.codebook_levels))
    if model.codebook_levels == 1:
        axs1 = [axs1]

    for level, vectors in enumerate(codebook_vectors):
        # Reshape the codebook vectors
        codebook_image = vectors  # Should already be in shape (num_embeddings, embedding_dim)
        
        # Plot codebook vectors
        axs1[level].imshow(codebook_image.T, aspect='auto', cmap='viridis')
        axs1[level].set_title(f'Codebook Level {level+1} Vectors')
        axs1[level].set_ylabel('Embedding Dimension')
        axs1[level].set_xlabel('Codebook Index')
        plt.colorbar(axs1[level].images[0], ax=axs1[level])

    plt.tight_layout()

    # Save the codebook visualization
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()
        wandb.log({
            'codebook/vectors': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - RVQ Codebook Vectors')
        })
    plt.close()

    # Second figure: Histograms
    fig2, axs2 = plt.subplots(model.codebook_levels, 2, 
                             figsize=(16, 4*model.codebook_levels))
    if model.codebook_levels == 1:
        axs2 = [axs2]

    for level, vectors in enumerate(codebook_vectors):
        # Compute magnitudes
        magnitudes = np.linalg.norm(vectors, axis=1)

        # Plot magnitude histogram
        axs2[level][0].hist(magnitudes, bins=12, color='blue', edgecolor='black')
        axs2[level][0].set_title(f'Level {level+1} - Histogram of Codebook Vector Magnitudes')
        axs2[level][0].set_xlabel('Magnitude')
        axs2[level][0].set_ylabel('Frequency')

        # Plot elements histogram
        axs2[level][1].hist(vectors.flatten(), bins=25, color='blue', edgecolor='black')
        axs2[level][1].set_title(f'Level {level+1} - Histogram of Codebook Vector Elements')
        axs2[level][1].set_xlabel('Element Value')
        axs2[level][1].set_ylabel('Frequency')

    plt.tight_layout()

    # Save the histogram visualization
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()
        wandb.log({
            'codebook/histograms': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - Histograms of RVQ Codebook Vectors')
        })

    plt.close()



def imshow(img, filename):
    imin, imax = img.min(), img.max()
    img = (img - imin) / (imax - imin) # rescale via max/min
    img = np.clip(img, 0, 1)
    npimg = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(npimg)
    pil_img.save(filename)


def save_img_grid(img, epoch, nfe, tag="", use_wandb=True, output_dir="output"):
    """Save image grid with consistent 10-column layout to match class conditioning"""
    filename = f"{tag}_epoch_{epoch + 1}_nfe_{nfe}.png"
    # Use nrow=10 to ensure grid columns match our class conditioning
    img_grid = make_grid(img, nrow=10)
    file_path = os.path.join(output_dir, filename)
    imshow(img_grid, file_path)
    name = f"demo/{tag}"
    if 'euler' in name: name = name + f"_nf{nfe}"
    if use_wandb: wandb.log({name: wandb.Image(file_path, caption=f"Epoch: {epoch}")})

