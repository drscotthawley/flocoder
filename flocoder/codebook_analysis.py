# flocoder/codebook_analysis.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb
from collections import defaultdict
import tempfile

class CodebookUsageTracker:
    """Track and analyze codebook usage across datasets"""
    
    def __init__(self, num_levels, codebook_size):
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.datasets = {}  # {name: (level_counts, combinations)}
        self.reset_all()
    
    def reset_all(self): self.datasets = {}
    def reset_dataset(self, name): 
        if name in self.datasets: del self.datasets[name]
    
    def add_dataset(self, name):
        level_counts = [defaultdict(int) for _ in range(self.num_levels)]
        combinations = defaultdict(int)
        self.datasets[name] = (level_counts, combinations)
    
    def update_counts(self, name, indices):
        """Update usage counts for a specific dataset"""
        if name not in self.datasets: self.add_dataset(name)
        level_counts, combinations = self.datasets[name]
        
        for level in range(self.num_levels): # Keep on GPU for faster operations
            level_indices = indices[:, level]
            counts = torch.bincount(level_indices, minlength=max(level_indices.max().item() + 1, 
                                                                max(level_counts[level].keys()) + 1 if level_counts[level] else 1))
            nonzero_mask = counts > 0
            nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]
            for idx in nonzero_indices: level_counts[level][idx.item()] += counts[idx].item()
        
        unique_combos, combo_counts = torch.unique(indices, dim=0, return_counts=True) # GPU-accelerated combination counting
        unique_combos_cpu, combo_counts_cpu = unique_combos.cpu().numpy(), combo_counts.cpu().numpy()
        for combo, count in zip(unique_combos_cpu, combo_counts_cpu):
            combinations[tuple(combo)] += count
    
    def analyze(self, codec, epoch, use_wandb=True, debug=True):
        """Run complete codebook analysis on all tracked datasets"""
        if not self.datasets: return
        if debug: print(f"Running codebook analysis at epoch {epoch}")
        
        usage_stats = calc_usage_stats(self.datasets, self.codebook_size)
        if use_wandb: wandb.log({f'codebook/{k}': v for k, v in usage_stats.items()})
        plot_usage_histograms(self.datasets, self.codebook_size, epoch, use_wandb, debug)
        plot_combo_usage_map(self.datasets, epoch, self.codebook_size, use_wandb, debug)
        plot_zq_3d_scatter(self.datasets, codec, epoch, use_wandb, debug)
        
        for name, (level_counts, combinations) in self.datasets.items(): # frequency-based 3D scatter plots
            if debug: print(f"Creating frequency scatter for {name}")
            plot_zq_3d_frequency_scatter(name, level_counts, combinations, codec, epoch, 
                                       colormap='viridis', use_log=True, use_wandb=use_wandb, debug=debug)
        if debug: print("Codebook analysis completed")

def create_empty_counts(num_levels):
    """Create empty count structures"""
    level_counts = [defaultdict(int) for _ in range(num_levels)]
    combinations = defaultdict(int)
    return level_counts, combinations

def update_usage_counts(indices, level_counts, combinations):
    """Update usage counts with GPU acceleration. Returns updated count structures."""
    num_levels = len(level_counts)
    for level in range(num_levels):
        level_indices = indices[:, level]
        counts = torch.bincount(level_indices, minlength=max(level_indices.max().item() + 1, 
                                                            max(level_counts[level].keys()) + 1 if level_counts[level] else 1))
        nonzero_mask = counts > 0
        nonzero_indices = torch.nonzero(nonzero_mask, as_tuple=True)[0]
        for idx in nonzero_indices: level_counts[level][idx.item()] += counts[idx].item()
    
    unique_combos, combo_counts = torch.unique(indices, dim=0, return_counts=True)
    unique_combos_cpu, combo_counts_cpu = unique_combos.cpu().numpy(), combo_counts.cpu().numpy()
    for combo, count in zip(unique_combos_cpu, combo_counts_cpu):
        combinations[tuple(combo)] += count
    return level_counts, combinations

def calc_usage_stats(data_dict,      # {name: (level_counts, combinations)} for each dataset
                    codebook_size):  # size of each codebook level
    """Calculate usage statistics from count data"""
    stats = {}
    names = list(data_dict.keys())
    num_levels = len(data_dict[names[0]][0])
    
    for level in range(num_levels):
        for name, (level_counts, _) in data_dict.items():
            used = len(level_counts[level])
            stats[f'level_{level}_{name}_usage_pct'] = used / codebook_size * 100
        
        if len(names) == 2:
            name1, name2 = names
            set1, set2 = set(data_dict[name1][0][level].keys()), set(data_dict[name2][0][level].keys())
            val_only = set2 - set1
            stats[f'level_{level}_{name2}_only_count'] = len(val_only)
            stats[f'level_{level}_{name2}_only_pct'] = len(val_only) / len(set2) * 100 if set2 else 0
        
    if len(names) == 2:
        name1, name2 = names
        combos1, combos2 = set(data_dict[name1][1].keys()), set(data_dict[name2][1].keys())
        val_only_combos = combos2 - combos1
        stats[f'combo_{name1}_count'] = len(combos1)
        stats[f'combo_{name2}_count'] = len(combos2)
        stats[f'combo_{name2}_only_count'] = len(val_only_combos)
        stats[f'combo_{name2}_only_pct'] = len(val_only_combos) / len(combos2) * 100 if combos2 else 0
    return stats

def plot_usage_histograms(data_dict,      # {name: (level_counts, combinations)} for each dataset
                         codebook_size,   # size of each codebook level
                         epoch,           # current epoch number
                         use_wandb=True,  # whether to log to wandb
                         debug=False):    # whether to print debug info
    """Create overlaid histograms for each codebook level"""
    names = list(data_dict.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(names)]
    num_levels = len(data_dict[names[0]][0])
    
    fig, axes = plt.subplots(num_levels, 1, figsize=(12, 4*num_levels))
    if num_levels == 1: axes = [axes]
    
    for level in range(num_levels):
        ax = axes[level]
        all_indices = range(codebook_size)
        for i, (name, (level_counts, _)) in enumerate(data_dict.items()):
            freqs = [level_counts[level].get(idx, 0) for idx in all_indices]
            ax.bar(all_indices, freqs, alpha=0.7, label=name, color=colors[i])
        
        ax.set_title(f'Level {level} Codebook Usage (Epoch {epoch})')
        ax.set_xlabel('Codebook Index')
        ax.set_ylabel('Usage Count')
        if len(names) == 2:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[0], alpha=0.7, label=names[0]),
                Patch(facecolor=colors[1], alpha=0.7, label=names[1]),
                Patch(facecolor='purple', alpha=0.7, label='Overlap')
            ]
            ax.legend(handles=legend_elements)
        else:
            ax.legend()

    
    plt.tight_layout()
    if use_wandb:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight')
            wandb.log({'codebook/usage_histograms': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - Codebook Usage by Level')})
    plt.close()



def plot_combo_usage_map(data_dict,      # {name: (level_counts, combinations)} for each dataset
                         epoch,          # current epoch number
                         codebook_size,  # size of each codebook level
                         use_wandb=True, # whether to log to wandb
                         debug=False):   # whether to print debug info
    """Plot 3-panel codebook combinations: categorical usage + frequency maps"""
    names = list(data_dict.keys())
    if len(names) != 2: return # only works for pairs
    
    name1, name2 = names
    combos1, combos2 = data_dict[name1][1], data_dict[name2][1]
    
    combo_matrix = np.zeros((codebook_size, codebook_size), dtype=int) # 0=unused, 1=first_only, 2=second_only, 3=both
    freq_matrices = {name: np.zeros((codebook_size, codebook_size), dtype=float) for name in names}
    
    for combo, count in combos1.items(): # mark first dataset combinations
        if len(combo) == 2:
            i, j = combo
            if 0 <= i < codebook_size and 0 <= j < codebook_size:
                combo_matrix[i, j] = 1
                freq_matrices[name1][i, j] = count
    
    for combo, count in combos2.items(): # mark second dataset combinations
        if len(combo) == 2:
            i, j = combo
            if 0 <= i < codebook_size and 0 <= j < codebook_size:
                freq_matrices[name2][i, j] = count
                combo_matrix[i, j] = 3 if combo_matrix[i, j] == 1 else 2
    
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1.3, 1.3])
    axs = [fig.add_subplot(gs[i//3, i%3]) for i in range(6)]
    
    for ax in axs:
        ax.set_xlabel('Level 0 Codebook Index')
        ax.set_ylabel('Level 1 Codebook Index')
    
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    colors = ['white', 'blue', 'red', 'purple']
    cmap_cat = ListedColormap(colors)
    
    im1 = axs[0].imshow(combo_matrix.T, cmap=cmap_cat, vmin=0, vmax=3, origin='lower')
    axs[0].set_title('Usage Categories')
    
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Unused'),
        Patch(facecolor='blue', label=f'{name1} only'),
        Patch(facecolor='red', label=f'{name2} only'), 
        Patch(facecolor='purple', label='Both')
    ]
    axs[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
    unused_pct = (np.sum(combo_matrix == 0)) / combo_matrix.size * 100
    axs[0].text(1.02, 0.3, f'Unused = {unused_pct:.1f}%', transform=axs[0].transAxes, fontsize=10)
    
    colors_map = ['Blues', 'Reds', 'Greens', 'Oranges'] # frequency heatmaps
    for i, (name, freq_matrix) in enumerate(freq_matrices.items()):
        im = axs[i+1].imshow(freq_matrix.T, cmap=colors_map[i], origin='lower')
        axs[i+1].set_title(f'{name} Frequency')
        plt.colorbar(im, ax=axs[i+1], label='Usage Count', shrink=0.6)
        
        im = axs[i+4].imshow(np.log10(1+freq_matrix).T, cmap=colors_map[i], origin='lower')
        axs[i+4].set_title(f'{name} Frequency (Log)')
        plt.colorbar(im, ax=axs[i+4], label='log10(1 + Usage Count)', shrink=0.6)
    
    axs[3].set_visible(False)
    plt.suptitle(f'Codebook Combinations (Epoch {epoch})')
    plt.tight_layout()
    
    if use_wandb:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight')
            wandb.log({'codebook/combination_usage_map': wandb.Image(tmpfile.name,
                caption=f'Epoch {epoch} - Usage Categories + Frequencies')})
    plt.close()

def plot_zq_3d_scatter(data_dict,     # {name: (level_counts, combinations)} for each dataset
                       codec,        # codec model with vq.codebooks
                       epoch,        # current epoch for logging
                       use_wandb=True, # whether to log to wandb
                       debug=False): # whether to print debug info
    """Plot 3D scatter of quantized vectors in embedding space"""
    import plotly.graph_objects as go
    
    names = list(data_dict.keys())
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(names)]
    codebook_vectors = [codebook.detach().cpu().numpy() for codebook in codec.vq.codebooks]
    
    fig = go.Figure()
    all_points, overlap_points = [], []
    
    for i, (name, (_, combos)) in enumerate(data_dict.items()):
        points = []
        for combo, count in combos.items():
            if len(combo) == 2:
                i_idx, j_idx = combo
                if i_idx < len(codebook_vectors[0]) and j_idx < len(codebook_vectors[1]):
                    true_vector = codebook_vectors[0][i_idx] + codebook_vectors[1][j_idx]
                    points.append(true_vector)
        
        if points:
            points = np.array(points)
            if all_points:  # Check for overlaps
                overlap_mask = np.array([any(np.allclose(p, pp, atol=1e-6) for pp in all_points) for p in points])
                overlapping_points, unique_points = points[overlap_mask], points[~overlap_mask]
                if len(overlapping_points) > 0: overlap_points.extend(overlapping_points)
                if len(unique_points) > 0:
                    fig.add_trace(go.Scatter3d(x=unique_points[:,0], y=unique_points[:,1], z=unique_points[:,2],
                        mode='markers', marker=dict(color=colors[i], size=5, opacity=0.6), name=name,
                        hovertemplate=f'<b>{name}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'))
            else:  # First dataset
                fig.add_trace(go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', 
                    marker=dict(color=colors[i], size=5, opacity=0.6), name=name,
                    hovertemplate=f'<b>{name}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'))
            all_points.extend(points)
    
    if overlap_points: # Add "Both" trace
        overlap_points = np.array(overlap_points)
        fig.add_trace(go.Scatter3d(x=overlap_points[:,0], y=overlap_points[:,1], z=overlap_points[:,2],
            mode='markers', marker=dict(color='purple', size=5, opacity=0.8), name='Both',
            hovertemplate='<b>Both</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'))

    fig.update_layout(title=f'Quantized Vectors in 3D Space (Epoch {epoch})',
        scene=dict(xaxis_title='Embedding Dim 0', yaxis_title='Embedding Dim 1', zaxis_title='Embedding Dim 2',
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))), width=800, height=800, showlegend=True)

    if use_wandb: 
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            fig.write_html(tmp.name)
            wandb.log({'codebook/zq_3d_scatter': wandb.Html(tmp.name)})

def plot_zq_3d_frequency_scatter(name,        # name of dataset
                                 level_counts, # level counts for this dataset
                                 combinations, # combinations for this dataset  
                                 codec,        # codec model with vq.codebooks
                                 epoch,        # current epoch number
                                 colormap='viridis', # colormap to use
                                 use_log=True, # whether to use log scale
                                 use_wandb=True, # whether to log to wandb
                                 debug=False): # whether to print debug info
    """Plot 3D scatter with frequency-based coloring for a single dataset"""
    import plotly.graph_objects as go
    
    codebook_vectors = [codebook.detach().cpu().numpy() for codebook in codec.vq.codebooks]
    points, counts = [], []
    for combo, count in combinations.items():
        if len(combo) == 2:
            i, j = combo
            if i < len(codebook_vectors[0]) and j < len(codebook_vectors[1]):
                true_vector = codebook_vectors[0][i] + codebook_vectors[1][j]
                points.append(true_vector)
                counts.append(count)
    
    if not points: return
    
    points, counts = np.array(points), np.array(counts)
    color_values = np.log10(1 + counts) if use_log else counts
    color_label = f'log10(1 + Frequency)' if use_log else 'Frequency'
    
    fig = go.Figure(data=go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers',
        marker=dict(size=6, color=color_values, colorscale=colormap, colorbar=dict(title=color_label), opacity=0.8),
        text=[f'Combo: {combo}<br>Count: {count}' for combo, count in zip(combinations.keys(), counts)],
        hovertemplate=f'<b>{name}</b><br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<br>%{{text}}<extra></extra>'))
    
    log_suffix = '_log' if use_log else ''
    fig.update_layout(title=f'{name} Frequency in 3D Space (Epoch {epoch}){" - Log Scale" if use_log else ""}',
        scene=dict(xaxis_title='Embedding Dim 0', yaxis_title='Embedding Dim 1', zaxis_title='Embedding Dim 2',
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))), width=800, height=800)
    
    if use_wandb:
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            fig.write_html(tmp.name)
            wandb.log({f'codebook/{name}_3d_frequency_scatter{log_suffix}': wandb.Html(tmp.name)})

def viz_codebook_vectors(model,        # RVQ model with codebooks
                         epoch,        # current epoch number
                         no_wandb=False, # skip wandb logging if true
                         debug=False): # whether to print debug info
    """Visualize codebook vectors from RVQ model"""
    codebook_vectors = [codebook.detach().cpu().numpy() for codebook in model.vq.codebooks]
    
    fig1, axs1 = plt.subplots(model.codebook_levels, 1, figsize=(16, 4*model.codebook_levels))
    if model.codebook_levels == 1: axs1 = [axs1]

    for level, vectors in enumerate(codebook_vectors):
        axs1[level].imshow(vectors.T, aspect='auto', cmap='viridis')
        axs1[level].set_title(f'Codebook Level {level} Vectors')
        axs1[level].set_ylabel('Embedding Dimension')
        axs1[level].set_xlabel('Codebook Index')
        plt.colorbar(axs1[level].images[0], ax=axs1[level])

    plt.tight_layout()
    if not no_wandb:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
            wandb.log({'codebook/vectors': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - RVQ Codebook Vectors')})
    plt.close()

    fig2, axs2 = plt.subplots(model.codebook_levels, 2, figsize=(16, 4*model.codebook_levels))
    if model.codebook_levels == 1: axs2 = [axs2]

    for level, vectors in enumerate(codebook_vectors):
        magnitudes = np.linalg.norm(vectors, axis=1)
        axs2[level][0].hist(magnitudes, bins=12, color='blue', edgecolor='black')
        axs2[level][0].set_title(f'Level {level} - Histogram of Codebook Vector Magnitudes')
        axs2[level][0].set_xlabel('Magnitude')
        axs2[level][0].set_ylabel('Frequency')
        axs2[level][1].hist(vectors.flatten(), bins=25, color='blue', edgecolor='black')
        axs2[level][1].set_title(f'Level {level} - Histogram of Codebook Vector Elements')
        axs2[level][1].set_xlabel('Element Value')
        axs2[level][1].set_ylabel('Frequency')

    plt.tight_layout()
    if not no_wandb:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            plt.savefig(tmpfile.name, format='png', bbox_inches='tight', pad_inches=0)
            wandb.log({'codebook/histograms': wandb.Image(tmpfile.name, 
                caption=f'Epoch {epoch} - Histograms of RVQ Codebook Vectors')})
    plt.close()

def analyze_codebooks(data_dict,      # {name: (level_counts, combinations)} for each dataset
                     codebook_size,   # size of each codebook level
                     model,           # RVQ model with codebooks
                     epoch,           # current epoch number
                     use_wandb=True,  # whether to log to wandb
                     debug=False):    # whether to print debug info
    """Run complete codebook analysis: stats, histograms, usage maps, and 3D scatter"""
    if debug: print(f"Running legacy analyze_codebooks at epoch {epoch}")
    usage_stats = calc_usage_stats(data_dict, codebook_size)
    if use_wandb: wandb.log({f'codebook/{k}': v for k, v in usage_stats.items()})
    plot_usage_histograms(data_dict, codebook_size, epoch, use_wandb, debug)
    plot_combo_usage_map(data_dict, epoch, codebook_size, use_wandb, debug)
    plot_zq_3d_scatter(data_dict, model, epoch, use_wandb, debug)
    
    for name, (level_counts, combinations) in data_dict.items():
        if debug: print(f"Creating frequency scatter for {name}")
        plot_zq_3d_frequency_scatter(name, level_counts, combinations, model, epoch, 
                                   colormap='viridis', use_log=True, use_wandb=use_wandb, debug=debug)
    if debug: print("Legacy analyze_codebooks completed")

