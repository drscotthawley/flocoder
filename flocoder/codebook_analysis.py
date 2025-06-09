# flocoder/codebook_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from collections import defaultdict
import tempfile

def create_empty_counts(num_levels): # number of RVQ levels
    """Create empty count structures"""
    level_counts = [defaultdict(int) for _ in range(num_levels)]
    combinations = defaultdict(int)
    return level_counts, combinations

def update_usage_counts(indices,              # [batch, levels] tensor of codebook indices
                       train_level_counts,    # list of defaultdicts for train counts per level  
                       train_combinations,    # defaultdict for train combination counts
                       val_level_counts=None, # list of defaultdicts for val counts per level
                       val_combinations=None, # defaultdict for val combination counts
                       is_validation=False):  # whether this is validation data
    """Update usage counts. Returns updated count structures."""
    level_counts = val_level_counts if is_validation else train_level_counts
    combinations = val_combinations if is_validation else train_combinations
    indices_np = indices.detach().cpu().numpy()
    num_levels = len(level_counts)
    
    for level in range(num_levels): # track individual level usage
        level_indices = indices_np[:, level]
        for idx in level_indices: level_counts[level][idx] += 1
            
    for combo in indices_np: # track cross-level combinations
        combo_key = tuple(combo)
        combinations[combo_key] += 1
    
    return level_counts, combinations

def calc_usage_stats(train_level_counts, # list of defaultdicts for train counts per level
                    train_combinations,  # defaultdict for train combination counts
                    val_level_counts,    # list of defaultdicts for val counts per level  
                    val_combinations,    # defaultdict for val combination counts
                    codebook_size):      # size of each codebook level
    """Calculate usage statistics from count data"""
    stats = {}
    num_levels = len(train_level_counts)
    
    for level in range(num_levels): # per-level usage percentages
        train_used = len(train_level_counts[level])
        val_used = len(val_level_counts[level])
        stats[f'level_{level}_train_usage_pct'] = train_used / codebook_size * 100
        stats[f'level_{level}_val_usage_pct'] = val_used / codebook_size * 100
        
        train_set = set(train_level_counts[level].keys()) # overlap analysis
        val_set = set(val_level_counts[level].keys())
        val_only = val_set - train_set
        stats[f'level_{level}_val_only_count'] = len(val_only)
        stats[f'level_{level}_val_only_pct'] = len(val_only) / len(val_set) * 100 if val_set else 0
        
    train_combos = set(train_combinations.keys()) # combination stats
    val_combos = set(val_combinations.keys())
    val_only_combos = val_combos - train_combos
    stats['combo_train_count'] = len(train_combos)
    stats['combo_val_count'] = len(val_combos)
    stats['combo_val_only_count'] = len(val_only_combos)
    stats['combo_val_only_pct'] = len(val_only_combos) / len(val_combos) * 100 if val_combos else 0
    
    return stats

def plot_usage_histograms(train_level_counts, # list of defaultdicts for train counts per level
                         val_level_counts,    # list of defaultdicts for val counts per level
                         codebook_size,       # size of each codebook level
                         epoch,               # current epoch number
                         use_wandb=True):     # whether to log to wandb
    """Create overlaid histograms for each codebook level"""
    num_levels = len(train_level_counts)
    fig, axes = plt.subplots(num_levels, 1, figsize=(12, 4*num_levels))
    if num_levels == 1: axes = [axes]
    
    for level in range(num_levels):
        ax = axes[level]
        all_indices = range(codebook_size) # create histogram data
        train_freqs = [train_level_counts[level].get(i, 0) for i in all_indices]
        val_freqs = [val_level_counts[level].get(i, 0) for i in all_indices]
        
        ax.bar(all_indices, train_freqs, alpha=0.7, label='Train', color='blue') # plot overlaid histograms
        ax.bar(all_indices, val_freqs, alpha=0.7, label='Val', color='red')
        ax.set_title(f'Level {level} Codebook Usage (Epoch {epoch})')
        ax.set_xlabel('Codebook Index')
        ax.set_ylabel('Usage Count')
        ax.legend()
        
        val_only = set(val_level_counts[level].keys()) - set(train_level_counts[level].keys()) # highlight val-only indices
        if val_only:
            val_only_freqs = [val_freqs[i] if i in val_only else 0 for i in all_indices]
            ax.bar(all_indices, val_only_freqs, alpha=0.9, label='Val Only', color='orange', width=0.8)
    
    plt.tight_layout()
    
    if use_wandb:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight')
            wandb.log({'codebook/usage_histograms': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - Codebook Usage by Level')})
    plt.close()


def plot_combo_usage_map(train_combos,  # defaultdict for train combination counts
                           val_combos,      # defaultdict for val combination counts
                           epoch,           # current epoch number
                           codebook_size,   # size of each codebook level
                           use_wandb=True): # whether to log to wandb
    """Plot 2D usage map of codebook combinations with overlap visualization"""
    # Create matrix: 0=unused, 1=train_only, 2=val_only, 3=both
    combo_matrix = np.zeros((codebook_size, codebook_size), dtype=int)
    
    # Mark train combinations
    for combo in train_combos.keys():
        if len(combo) == 2:  # only for 2-level RVQ
            i, j = combo
            if 0 <= i < codebook_size and 0 <= j < codebook_size:
                combo_matrix[i, j] = 1  # train only
    
    # Mark val combinations (override train-only, set both)
    for combo in val_combos.keys():
        if len(combo) == 2:
            i, j = combo
            if 0 <= i < codebook_size and 0 <= j < codebook_size:
                if combo_matrix[i, j] == 1:
                    combo_matrix[i, j] = 3  # both train and val
                else:
                    combo_matrix[i, j] = 2  # val only
    
    # Create custom colormap: white=unused, blue=train, red=val, purple=both
    from matplotlib.colors import ListedColormap
    colors = ['white', 'blue', 'red', 'purple']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(combo_matrix.T, cmap=cmap, vmin=0, vmax=3, origin='lower') # transpose b/c numpy 
    
    ax.set_xlabel('Level 0 Codebook Index')
    ax.set_ylabel('Level 1 Codebook Index') 
    ax.set_title(f'Codebook Combinations (Epoch {epoch})')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Unused'),
        Patch(facecolor='blue', label='Train only'),
        Patch(facecolor='red', label='Val only'), 
        Patch(facecolor='purple', label='Both')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    plt.tight_layout()
    
    if use_wandb:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight')
            wandb.log({'codebook/combination_usage': wandb.Image(tmpfile.name,
                caption=f'Epoch {epoch} - Codebook Combinations (Blue=Train, Red=Val, Purple=Both)')})
    plt.close()


def viz_codebook_vectors(model,        # RVQ model with codebooks
                 epoch,        # current epoch number
                 no_wandb=False): # skip wandb logging if true
    """Visualize codebook vectors from RVQ model"""
    codebook_vectors = [codebook.detach().cpu().numpy() for codebook in model.vq.codebooks] # extract codebook vectors from all levels
    
    fig1, axs1 = plt.subplots(model.codebook_levels, 1, figsize=(16, 4*model.codebook_levels)) # first figure: codebook visualizations
    if model.codebook_levels == 1: axs1 = [axs1]

    for level, vectors in enumerate(codebook_vectors):
        codebook_image = vectors  # should already be in shape (num_embeddings, embedding_dim)
        axs1[level].imshow(codebook_image.T, aspect='auto', cmap='viridis') # plot codebook vectors
        axs1[level].set_title(f'Codebook Level {level+1} Vectors')
        axs1[level].set_ylabel('Embedding Dimension')
        axs1[level].set_xlabel('Codebook Index')
        plt.colorbar(axs1[level].images[0], ax=axs1[level])

    plt.tight_layout()
    if not no_wandb: # save the codebook visualization
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
            tmpfile.flush()
            wandb.log({'codebook/vectors': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - RVQ Codebook Vectors')})
    plt.close()

    fig2, axs2 = plt.subplots(model.codebook_levels, 2, figsize=(16, 4*model.codebook_levels)) # second figure: histograms
    if model.codebook_levels == 1: axs2 = [axs2]

    for level, vectors in enumerate(codebook_vectors):
        magnitudes = np.linalg.norm(vectors, axis=1) # compute magnitudes
        axs2[level][0].hist(magnitudes, bins=12, color='blue', edgecolor='black') # plot magnitude histogram
        axs2[level][0].set_title(f'Level {level+1} - Histogram of Codebook Vector Magnitudes')
        axs2[level][0].set_xlabel('Magnitude')
        axs2[level][0].set_ylabel('Frequency')
        axs2[level][1].hist(vectors.flatten(), bins=25, color='blue', edgecolor='black') # plot elements histogram
        axs2[level][1].set_title(f'Level {level+1} - Histogram of Codebook Vector Elements')
        axs2[level][1].set_xlabel('Element Value')
        axs2[level][1].set_ylabel('Frequency')

    plt.tight_layout()
    if not no_wandb: # save the histogram visualization
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
            tmpfile.flush()
            wandb.log({'codebook/histograms': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - Histograms of RVQ Codebook Vectors')})
    plt.close()

