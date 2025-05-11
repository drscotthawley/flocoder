import torch 
from torchvision.utils import make_grid
import wandb
import matplotlib.pyplot as plt
import tempfile
import numpy as np


def denormalize(image_batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """undoes transforms normalization, use this before displaying output demo images"""
    # Create a deep copy to avoid modifying the original tensor
    image_batch = image_batch.clone().detach()
    
    # Ensure mean and std are on the same device as image_batch
    mean = torch.tensor(mean, device=image_batch.device)
    std = torch.tensor(std, device=image_batch.device)
    
    # For batched input with shape [B, C, H, W]
    # Reshape mean and std for proper broadcasting
    if image_batch.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    
    # Apply inverse normalization
    image_batch = image_batch * std + mean
    
    return image_batch




def viz_codebook(model, config, epoch):
    if config.no_wandb: return
    # Extract VQ codebook vectors
    codebook_vectors = model.vq.codebook.detach().cpu().numpy()
    
    # Reshape the codebook vectors to the desired shape
    codebook_image = codebook_vectors.reshape(config.vq_num_embeddings, config.vq_embedding_dim)
    
    # Create an image of the codebook vectors using matplotlib
    plt.figure(figsize=(16, 4))
    plt.imshow(codebook_image.T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('VQ Codebook Vectors')
    plt.ylabel('Embedding Dimension')
    plt.xlabel('Codebook Index')
    
    # Adjust layout to remove extra margins and whitespace
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()
        
        # Log the image to wandb
        wandb.log({
            'codebook/image': wandb.Image(tmpfile.name, caption=f'Epoch {epoch} - VQ Codebook Vectors')
        })
    
    plt.close()
    
    # Compute the magnitudes of the codebook vectors
    magnitudes = np.linalg.norm(codebook_vectors, axis=1)

    # Create a figure with one row and two columns for the histograms
    fig, axs = plt.subplots(1, 2, figsize=(16, 4))

    # Plot the histogram of magnitudes
    axs[0].hist(magnitudes, bins=50, color='blue', edgecolor='black')
    axs[0].set_title('Histogram of Codebook Vector Magnitudes')
    axs[0].set_xlabel('Magnitude')
    axs[0].set_ylabel('Frequency')

    # Plot the histogram of elements
    axs[1].hist(codebook_vectors.flatten(), bins=200, color='blue', edgecolor='black')
    axs[1].set_title('Histogram of Codebook Vector Elements')
    axs[1].set_xlabel('Element Value')
    axs[1].set_ylabel('Frequency')

    # Adjust layout to remove extra margins and whitespace
    plt.tight_layout()

    # Save the histogram image to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
        tmpfile.flush()

        # Log the histogram image to wandb
        wandb.log({
            'codebook/histograms': wandb.Image(tmpfile.name, caption=f'Epoch {epoch} - Histograms of Codebook Vectors')
        })

    plt.close()



def viz_codebooks(model, config, epoch): # RVQ
    if config.no_wandb: return

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

