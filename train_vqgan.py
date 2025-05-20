#! /usr/bin/env python3 

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

from flocoder.data import create_image_loaders
from flocoder.vqvae import VQVAE
from flocoder.viz import viz_codebook, viz_codebooks, denormalize
from flocoder.metrics import *  # there's a lot
from flocoder.general import save_checkpoint



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







def train_vqgan(args):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    # Data setup
    is_midi = 'pop909' in args.data.lower() or 'midi' in args.data.lower()
    train_loader, val_loader = create_image_loaders(batch_size=args.batch_size, image_size=args.image_size, 
                                            data_path=args.data, num_workers=16, is_midi=is_midi)

    # Initialize models and losses
    codec = VQVAE(
        in_channels=3,
        hidden_channels=args.hidden_channels,
        num_downsamples=args.num_downsamples,
        vq_num_embeddings=args.vq_num_embeddings,
        internal_dim=args.internal_dim,
        codebook_levels=args.codebook_levels,
        vq_embedding_dim=args.vq_embedding_dim,
        commitment_weight=args.commitment_weight,
        use_checkpoint=not args.no_grad_ckpt,
    ).to(device)

    optimizer = optim.Adam(codec.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = None # optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.epochs)

    # vgg is used for perceptual loss part of adversarial training
    vgg = vgg16(weights='IMAGENET1K_V1').features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False    
    adv_loss = AdversarialLoss(device, use_checkpoint=not args.no_grad_ckpt).to(device)
    d_optimizer = optim.Adam(adv_loss.discriminator.parameters(), 
                            lr=args.learning_rate * 0.1, 
                            weight_decay=1e-5)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        # codec.load_state_dict(checkpoint['codec_state_dict'])
        codec = load_checkpoint_non_frozen(codec,checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")


    # Initialize wandb
    if not args.no_wandb:
        wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.warmup_epochs:
            print("*** WARMUP PERIOD FINISHED. Engaging adversarial training. ***")

        codec.train()
        epoch_losses = defaultdict(float)
        total_batches = 0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            source_imgs = batch[0].to(device)
            target_imgs = source_imgs
            
            noise_strength = 0.0 if epoch < args.warmup_epochs else 0.2 * min(1.0, (epoch - args.warmup_epochs) / 10)

            # Pre-warmup training
            if epoch < args.warmup_epochs:
                recon, vq_loss = codec(source_imgs)
  
                losses = compute_vqgan_losses(recon, target_imgs, vq_loss, vgg, 
                                        adv_loss=None, epoch=epoch, config=args)
                losses['total'] = get_total_vqgan_loss(losses, args)
                
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
                                            adv_loss=adv_loss, epoch=epoch, config=args)

                    losses['total'] = get_total_vqgan_loss(losses, args)

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
            if not args.no_wandb:
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
                                            adv_loss=adv_loss, epoch=epoch, config=args)
                    losses['total'] = get_total_vqgan_loss(losses, args)

                    # Update validation losses
                    batch_losses = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
                    val_total_batches += 1
                    for k, v in batch_losses.items():
                        val_losses[k] += v

                    # Log validation visualizations for first batch
                    if batch_idx == 0 and not args.no_wandb:
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
                            orig, recon = g2rgb(orig), g2rgb(recon)
                            viz_images = torch.cat([orig, recon])
                            caption = f'Epoch {epoch} - Top: Source, Bottom: Recon'
                            log_dict['demo/examples'] = wandb.Image( make_grid(viz_images, nrow=nrow, normalize=True), caption=caption)
                            wandb.log(log_dict)

        # Compute average validation losses
        val_losses = {k: v / val_total_batches for k, v in val_losses.items()}

        if epoch < 10 or epoch % max(1, int(epoch ** 0.5)) == 0:
            viz_codebooks(codec, args, epoch)

        # Log epoch metrics
        if not args.no_wandb:
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


    return 'Main finished.'


def parse_args_with_config():
    """
    This lets you specify args via a YAML config file and/or override those with command line args.
    i.e. CLI args take precedence 
    """
    import argparse
    import yaml 

    # First pass to check for config file
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None, help='path to config YAML file')
    args, _ = parser.parse_known_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config_full = yaml.safe_load(f)
        
        # grab only the "data" and the "vqgan" sections if the exists, and flatten the result 
        if 'vqgan' in yaml_config_full:
            yaml_config = yaml_config_full['vqgan']
        if 'data' in yaml_config_full:
            yaml_config["data"] = yaml_config_full['data'] 
        # Flatten any further nested structure, and treat config variable names as CLI args
        for section, params in yaml_config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    config[key.replace('_', '-')] = value
            else:
                config[section.replace('_', '-')] = params
    
    # Create full parser with config-based defaults
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add config option
    parser.add_argument('--config', type=str, default=None, help='path to config YAML file')
    
    # Training parameters with config-based defaults
    parser.add_argument('--batch-size', type=int, default=config.get('batch-size', 32), 
                       help='the batch size')
    parser.add_argument('--data', type=str, 
                       default=config.get('data', None), 
                       help='path to top-level-directory containing custom image data')
    parser.add_argument('--epochs', type=int, 
                       default=config.get('num-epochs', 1000000), 
                       help='number of epochs')
    parser.add_argument('--base-lr', type=float, 
                       default=config.get('learning-rate', 1e-4), 
                       help='base learning rate for batch size of 32')
    parser.add_argument('--image-size', type=int, 
                       default=config.get('image-size', 128), 
                       help='will rescale images to squares of (image-size, image-size)')
    parser.add_argument('--warmup-epochs', type=int, 
                       default=config.get('warmup-epochs', 15), 
                       help='number of epochs before enabling adversarial loss')
    parser.add_argument('--lambda-adv', type=float, 
                       default=config.get('lambda-adv', 0.03), 
                       help="regularization param for G part of adversarial loss")
    parser.add_argument('--lambda-ce', type=float, 
                       default=config.get('lambda-ce', 2.0), 
                       help="regularization param for cross-entropy loss")
    parser.add_argument('--lambda-l1', type=float, 
                       default=config.get('lambda-l1', 0.2), 
                       help="regularization param for L1/Huber loss")
    parser.add_argument('--lambda-mse', type=float, 
                       default=config.get('lambda-mse', 0.5), 
                       help="regularization param for MSE loss")
    parser.add_argument('--lambda-perc', type=float, 
                       default=config.get('lambda-perc', 1e-5), 
                       help="regularization param for perceptual loss")
    parser.add_argument('--lambda-spec', type=float, 
                       default=config.get('lambda-spec', 2e-4), 
                       help="regularization param for spectral loss")
    parser.add_argument('--lambda-vq', type=float, 
                       default=config.get('lambda-vq', 0.25), 
                       help="reg factor mult'd by VQ commitment loss")
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')

    # ae codec Model parameters
    parser.add_argument('--load-checkpoint', type=str, 
                       default=config.get('load-checkpoint', None), 
                       help='path to load checkpoint to resume training from')
    parser.add_argument('--hidden-channels', type=int, 
                       default=config.get('hidden-channels', 256))
    parser.add_argument('--num-downsamples', type=int, 
                       default=config.get('num-downsamples', 3), 
                       help='total downsampling is 2**[this]')
    parser.add_argument('--vq-num-embeddings', type=int, 
                       default=config.get('vq-num-embeddings', 32), 
                       help='aka codebook length')
    parser.add_argument('--internal-dim', type=int, 
                       default=config.get('internal-dim', 256), 
                       help='internal pre-vq emb dim before compression')
    parser.add_argument('--vq-embedding-dim', type=int, 
                       default=config.get('vq-embedding-dim', 4), 
                       help='(actual) dims of codebook vectors')
    parser.add_argument('--codebook-levels', type=int, 
                       default=config.get('codebook-levels', 4), 
                       help='number of RVQ levels')
    parser.add_argument('--commitment-weight', type=float, 
                       default=config.get('commitment-weight', 0.5), 
                       help='VQ commitment weight, aka quantization strength')
    parser.add_argument('--project-name', type=str, 
                       default=config.get('project-name', "vqgan-midi"), 
                       help='WandB project name')
    parser.add_argument('--run-name', type=str, 
                       default=config.get('run-name', None), 
                       help='WandB run name')
    parser.add_argument('--no-grad-ckpt', action='store_true', 
                       help='disable gradient checkpointing')
    
    # Parse args
    args = parser.parse_args()
    
    # Calculate learning rate based on batch size
    args.learning_rate = args.base_lr * math.sqrt(args.batch_size / 32)
    args.data = os.path.expanduser(args.data)
    
    return args




if __name__ == '__main__':
    args = parse_args_with_config()
    print("args = ",args)
    train_vqgan(args)
