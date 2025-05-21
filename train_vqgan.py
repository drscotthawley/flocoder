#!/usr/bin/env python3 

import matplotlib
matplotlib.use('Agg')
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # re. NATTEN's non-stop FutureWarnings
import math
from collections import defaultdict
import random 
import torch
torch.set_float32_matmul_precision('high') # did this just to check reconstruction error
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from flocoder.data import create_image_loaders
from flocoder.codecs import VQVAE, load_codec
from flocoder.viz import viz_codebooks, denormalize
from flocoder.metrics import *  # there's a lot
from flocoder.general import save_checkpoint, handle_config_path


class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
    "I added this scheduler to decay the base learning rate after T_0 epochs."
    def __init__(self, optimizer, T_0, T_mult=1,
                    eta_min=0, last_epoch=-1, verbose=False, decay=0.6):
        super().__init__(optimizer, T_0, T_mult=T_mult,
                            eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
        self.decay = decay
        self.initial_lrs = self.base_lrs
    
    def step(self, epoch=None):
        if epoch == None:
            if self.T_cur + 1 == self.T_i:
                if self.verbose:
                    print("multiplying base_lrs by {:.4f}".format(self.decay))
                self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    n = int(epoch / self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            else:
                n = 0
            
            self.base_lrs = [initial_lrs * (self.decay**n) for initial_lrs in self.initial_lrs]

        super().step(epoch)


def z_to_img(z_compressed, vq_model, debug=False):
    """utility routine to go from continuous z to output image"""
    # Reshape and prepare for VQ
    z_compressed = z_compressed.permute(0, 2, 3, 1)
    save_shape = z_compressed.shape
    z_compressed = z_compressed.reshape(-1, z_compressed.shape[-1])
            
    # Apply VQ
    z_q, indices, commit_loss = vq_model.vq(z_compressed)

    # Reshape back
    z_q = z_q.view(save_shape)
    if debug: print("z_q.shape =",z_q.shape)
    z_q = z_q.permute(0, 3, 1, 2)
    
    # Expand after VQ
    z_q = vq_model.expand(z_q)
    
    # Decode
    x_recon = vq_model.decoder(z_q, noise_strength=0.0)

    img = x_recon 
    return img 


def load_checkpoint_non_frozen(model, checkpoint):
    """
    Load only the frozen parameters from a checkpoint into the model.
    (If no parames were frozen, then this loads the entire model.)
    """
    checkpoint_state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    
    # Get only frozen parameters
    for name, param in model.named_parameters():
        if not param.requires_grad and name in checkpoint_state_dict:
            filtered_state_dict[name] = checkpoint_state_dict[name]
    
    # Include buffers for frozen parts
    for name, buffer in model.named_buffers():
        if (name.startswith('encoder.') or name.startswith('compress.')) and name in checkpoint_state_dict:
            filtered_state_dict[name] = checkpoint_state_dict[name]
    
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    print(f"Loaded {len(filtered_state_dict)} frozen parameters.")
    return model


def train_vqgan(cfg):
    print("train_vqgan: cfg =",cfg)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    # Data setup - access config values with appropriate defaults
    is_midi = hasattr(cfg, 'data') and ('pop909' in cfg.data.lower() or 'midi' in cfg.data.lower())
    batch_size = cfg.get('batch_size', 32)
    image_size = cfg.get('image_size', 128)
    data_path = cfg.get('data', None)
    num_workers = cfg.get('num_workers', 16)
    
    train_loader, val_loader = create_image_loaders(
        batch_size=batch_size, 
        image_size=image_size, 
        data_path=data_path, 
        num_workers=num_workers, 
        is_midi=is_midi
    )

    # Initialize model - either directly or using load_codec
    codec = VQVAE(
        in_channels=3,
        hidden_channels=cfg.get('hidden_channels', 256),
        num_downsamples=cfg.get('num_downsamples', 3),
        vq_num_embeddings=cfg.get('vq_num_embeddings', 512),
        internal_dim=cfg.get('internal_dim', 256),
        codebook_levels=cfg.get('codebook_levels', 4),
        vq_embedding_dim=cfg.get('vq_embedding_dim', 4),
        commitment_weight=cfg.get('commitment_weight', 0.25),
        use_checkpoint=not cfg.get('no_grad_ckpt', False),
    ).to(device)

    optimizer = optim.Adam(codec.parameters(), lr=cfg.get('learning_rate', 1e-4), weight_decay=1e-5)
    scheduler = None

    # vgg is used for perceptual loss part of adversarial training
    vgg = vgg16(weights='IMAGENET1K_V1').features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False    
    adv_loss = AdversarialLoss(device, use_checkpoint=not cfg.get('no_grad_ckpt', False)).to(device)
    d_optimizer = optim.Adam(adv_loss.discriminator.parameters(), 
                            lr=cfg.get('learning_rate', 1e-4) * 0.1, 
                            weight_decay=1e-5)

    # Resume from checkpoint if specified
    start_epoch = 0
    load_checkpoint = cfg.get('load_checkpoint', None)
    if load_checkpoint is not None:
        print(f"Loading checkpoint from {load_checkpoint}...")
        checkpoint = torch.load(load_checkpoint, map_location=device)
        codec = load_checkpoint_non_frozen(codec, checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print(f"No checkpoint specified (load_checkpoint=={load_checkpoint}). Starting from scratch")

    # Initialize wandb
    no_wandb = cfg.get('no_wandb', False)
    if not no_wandb:
        project_name = cfg.wandb.get('project_name', "vqgan-training")
        run_name = cfg.wandb.get('run_name', None)
        print("Got run name =",run_name)
        wandb.init(project=project_name, name=run_name, config=dict(cfg))

    # Training parameters
    epochs = cfg.get('epochs', 1000000)
    warmup_epochs = cfg.get('warmup_epochs', 15)
    
    # Lambda weight parameters with defaults
    lambda_adv = cfg.get('lambda_adv', 0.03)
    lambda_ce = cfg.get('lambda_ce', 2.0)
    lambda_l1 = cfg.get('lambda_l1', 0.2)
    lambda_mse = cfg.get('lambda_mse', 0.5)
    lambda_perc = cfg.get('lambda_perc', 1e-5)
    lambda_spec = cfg.get('lambda_spec', 2e-4)
    lambda_vq = cfg.get('lambda_vq', 0.25)
    
    # Set lambda values on cfg for compute_vqgan_losses
    cfg.lambda_adv = lambda_adv
    cfg.lambda_ce = lambda_ce
    cfg.lambda_l1 = lambda_l1
    cfg.lambda_mse = lambda_mse
    cfg.lambda_perc = lambda_perc
    cfg.lambda_spec = lambda_spec 
    cfg.lambda_vq = lambda_vq
    cfg.warmup_epochs = warmup_epochs

    # Main training loop
    for epoch in range(start_epoch, epochs):
        if epoch == warmup_epochs:
            print("*** WARMUP PERIOD FINISHED. Engaging adversarial training. ***")

        codec.train()
        epoch_losses = defaultdict(float)
        total_batches = 0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            source_imgs = batch[0].to(device)
            target_imgs = source_imgs
            
            noise_strength = 0.0 if epoch < warmup_epochs else 0.2 * min(1.0, (epoch - warmup_epochs) / 10)

            # Pre-warmup training
            if epoch < warmup_epochs:
                recon, vq_loss = codec(source_imgs)
  
                losses = compute_vqgan_losses(recon, target_imgs, vq_loss, vgg, 
                                        adv_loss=None, epoch=epoch, config=cfg)
                losses['total'] = get_total_vqgan_loss(losses, cfg)
                
                optimizer.zero_grad()
                losses['total'].backward()
                nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
                optimizer.step()

            # Post-warmup training
            else:
                # Train discriminator
                recon, vq_loss = codec(source_imgs, noise_strength=noise_strength)
                d_losses, d_stats_list, grad_stats_list = [], [], []
                
                for _ in range(1):
                    recon_detached = recon.detach()
                    d_loss, real_features = adv_loss.discriminator_loss(target_imgs, recon_detached)
                    d_stats = get_discriminator_stats(adv_loss, target_imgs, recon_detached)
                    d_stats_list.append(d_stats)

                    d_optimizer.zero_grad()
                    d_loss.backward()
                    nn.utils.clip_grad_norm_(adv_loss.discriminator.parameters(), max_norm=1.0)
                    grad_stats = get_gradient_stats(adv_loss.discriminator)
                    grad_stats_list.append(grad_stats)
                    d_losses.append(d_loss.item())
                    d_optimizer.step()

                # Train generator
                if batch_idx % 1 == 0:
                    recon, vq_loss = codec(source_imgs, noise_strength=noise_strength)
                    losses = compute_vqgan_losses(recon, target_imgs, vq_loss, vgg, 
                                            adv_loss=adv_loss, epoch=epoch, config=cfg)

                    losses['total'] = get_total_vqgan_loss(losses, cfg)

                    optimizer.zero_grad()
                    losses['total'].backward()
                    nn.utils.clip_grad_norm_(codec.parameters(), max_norm=1.0)
                    optimizer.step()

                    # Add discriminator losses
                    losses['d_loss'] = sum(d_losses) / len(d_losses)
                    avg_d_stats = {k: sum(d[k] for d in d_stats_list) / len(d_stats_list) 
                                 for k in d_stats_list[0]}
                    avg_grad_stats = {k: sum(d[k] for d in grad_stats_list) / len(grad_stats_list) 
                                    for k in grad_stats_list[0]}
                    losses.update(avg_d_stats)
                    losses.update(avg_grad_stats)

            # Update epoch losses and progress bar
            batch_losses = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
            total_batches += 1
            for k, v in batch_losses.items():
                epoch_losses[k] += v

            pbar.set_postfix({k: f'{v:.4g}' for k, v in batch_losses.items()})

            # Collect batch metrics for logging
            if not no_wandb:
                log_dict = {  'epoch': epoch,  **{f'batch/{k}_loss': v for k, v in batch_losses.items()} }
                wandb.log(log_dict)

        # Compute average training losses
        train_losses = {k: v / total_batches for k, v in epoch_losses.items()}

        # Validation phase
        with torch.no_grad():
            # TODO: maybe do some gpu vram garbage collection? 
            codec.eval()
            val_losses = defaultdict(float)
            val_total_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    source_imgs = batch[0].to(device)
                    target_imgs = source_imgs

                    # Basic validation forward pass
                    recon, vq_loss, dist_stats = codec(source_imgs, get_stats=True)
                    losses = compute_vqgan_losses(recon, target_imgs, vq_loss, vgg,
                                            adv_loss=adv_loss, epoch=epoch, config=cfg)
                    losses['total'] = get_total_vqgan_loss(losses, cfg)

                    # Update validation losses
                    batch_losses = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
                    val_total_batches += 1
                    for k, v in batch_losses.items():
                        val_losses[k] += v

                    # Log validation visualizations for first batch
                    if batch_idx == 0 and not no_wandb:
                        log_dict = {
                            'epoch': epoch,
                            'codebook/mean_distance': dist_stats['codebook_mean_dist'],
                            'codebook/max_distance': dist_stats['codebook_max_dist'],
                            **{f'validation/batch_{k}_loss': v for k, v in val_losses.items()},
                        }
                        if is_midi: 
                            note_metrics, metric_images = calculate_note_metrics(recon, target_imgs)
                            metric_grids = {k: make_grid(v[:8], nrow=8, normalize=True) 
                                    for k, v in metric_images.items()}
                            log_dict = log_dict | { 
                                **{f'note_metrics/{k}': v for k, v in note_metrics.items()},
                                **{f'metric_images/{k}': wandb.Image(v, caption=k) for k, v in metric_grids.items()}
                            }
                        
                        if epoch < 10 or epoch % max(1, int(epoch ** 0.5)) == 0:  # Add demo image visualization grid
                            nrow=8  # number of examples of each to show
                            orig = batch[0][:nrow].to(device)
                            recon = torch.clamp(recon[:nrow], orig.min(), orig.max())
                            if is_midi: orig, recon = g2rgb(orig), g2rgb(recon) # only for midi
                            viz_images = torch.cat([orig, recon])
                            caption = f'Epoch {epoch} - Top: Source, Bottom: Recon'
                            log_dict['demo/examples'] = wandb.Image(make_grid(viz_images, nrow=nrow, normalize=True), caption=caption)
                            wandb.log(log_dict)

        # Compute average validation losses
        val_losses = {k: v / val_total_batches for k, v in val_losses.items()}

        if epoch < 10 or epoch % max(1, int(epoch ** 0.5)) == 0:
            viz_codebooks(codec, cfg, epoch)

        # Log epoch metrics
        if not no_wandb:
            log_dict = {
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **{f'epoch/train_{k}_loss': v for k, v in train_losses.items()},
                **{f'epoch/val_{k}_loss': v for k, v in val_losses.items()}
            }
            wandb.log(log_dict)

        if (epoch + 1) % 250 == 0 and epoch > 0:
            save_checkpoint(codec, epoch, optimizer=optimizer)

        if scheduler:
            scheduler.step()

    return 'Training completed successfully.'


# Process any direct config file paths
handle_config_path()


@hydra.main(version_base="1.3", config_path="configs", config_name="flowers_sd")
def main(cfg):
    """Main entry point using Hydra."""
    print("cfg =",cfg)
    # Set torch options for better performance
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # Extract relevant config sections and handle various structure possibilities
    vqgan_cfg = None
    
    # Check for different possible config structures
    if hasattr(cfg, 'codec'):
        vqgan_cfg = cfg.codec
    elif hasattr(cfg, '_target_'):
        # Some Hydra configs use this structure
        vqgan_cfg = cfg
    else:
        # Assume the config is directly usable
        vqgan_cfg = cfg

    OmegaConf.set_struct(vqgan_cfg, False)
 
    # Handle the data path - check different possibilities
    if hasattr(cfg, 'data'):
        data_path = cfg.data
        # Add data to vqgan_cfg if it's a dictionary-like object
        if hasattr(vqgan_cfg, '__setattr__'):
            vqgan_cfg.data = data_path
        else:
            # If it's a regular dict
            vqgan_cfg = dict(vqgan_cfg)
            vqgan_cfg['data'] = data_path
    if hasattr(cfg, 'load_checkpoint'):
        vqgan_cfg.load_checkpoint = cfg.load_checkpoint
    
    # Print the available keys to help debug
    print(f"Available top-level config keys: {list(cfg.keys() if hasattr(cfg, 'keys') else dir(cfg))}")
    if vqgan_cfg is not None and hasattr(vqgan_cfg, 'keys'):
        print(f"VQGAN config keys: {list(vqgan_cfg.keys())}")
    
    # Pass the config to the training function
    train_vqgan(vqgan_cfg)
    return "Training complete"

if __name__ == "__main__":
    main()

