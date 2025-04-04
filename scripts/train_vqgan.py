#! /usr/bin/env python3 
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # re. NATTEN's non-stop FutureWarnings
import torch
torch.set_float32_matmul_precision('high')
import matplotlib
matplotlib.use('Agg')
import math
import torch.nn as nn
import torch.optim as optim
#from torch_optimizer import Ranger
#from pytorch_ranger import Ranger  # this is from ranger.py
from torchvision.models import vgg16
from torchvision.utils import make_grid

import wandb
from tqdm.auto import tqdm
import argparse


from flocoder.data.dataloaders import create_image_loaders
from flocoder.models.vqvae import VQVAE
from flocoder.training.vqgan_losses import *
from flocoder.utils.viz import viz_codebook, viz_codebooks

#from geomloss import SamplesLoss
from collections import defaultdict


import random 

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CosineAnnealingWarmRestartsDecay(CosineAnnealingWarmRestarts):
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


def freeze_front_vqvae(vqvae, debug=False):
    # Freeze encoder
    for param in vqvae.encoder.parameters():
        param.requires_grad = False
    
    # Freeze compress layers
    for param in vqvae.compress.parameters():
        param.requires_grad = False
    
    # Verify the other parts are still trainable (optional)
    if debug:
        for name, param in vqvae.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")
    
    return vqvae



def load_non_frozen(model, checkpoint):
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



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters 
    parser.add_argument('--batch-size', type=int, default=32, help='the batch size') 
    parser.add_argument('--data', type=str, default=os.path.expanduser('~')+'/datasets/POP909_images', help='path to top-level-directory containing custom image data. If not specified, uses Flowers102')
    parser.add_argument('--data-fakez', type=str, default=None, help='Directory of generated (fake) pre-encoder latents (to align via distribution-based losses)')
    parser.add_argument('--epochs', type=int, default=1000000, help='number of epochs. (just let it keep training for hours/days/weeks/etc.)')
    #parser.add_argument('--epochs', type=int, default=300, help='number of epochs. (just let it keep training for hours/days/weeks/etc.)')
    parser.add_argument('--base-lr', type=float, default=1e-4, help='base learning rate for batch size of 32')
    parser.add_argument('--image-size', type=int, default=128, help='will rescale images to squares of (image-size, image-size)')
    parser.add_argument('--warmup-epochs', type=int, default=15, help='number of epochs before enabling adversarial loss')  
    parser.add_argument('--lambda-adv', type=float, default=0.03,  help="regularization param for G part of adversarial loss") 
    parser.add_argument('--lambda-ce',  type=float, default=2.0, help="regularization param for cross-entropy loss")
    parser.add_argument('--lambda-l1',  type=float, default=0.2, help="regularization param for L1/Huber loss")
    parser.add_argument('--lambda-mse', type=float, default=0.5, help="regularization param for MSE loss")
    parser.add_argument('--lambda-perc',type=float, default=1e-5, help="regularization param for perceptual loss")
    parser.add_argument('--lambda-sinkhorn',type=float, default=1e-5,  help="regularization param for sinkhorn loss")

    parser.add_argument('--lambda-spec',type=float, default=2e-4,  help="regularization param for spectral loss (1e-4='almost off')") # with lambda_spec=0, spec_loss serves as an independent metric
    parser.add_argument('--lambda-vq',  type=float, default=0.25, help="reg factor mult'd by VQ commitment loss")
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')

    # Model parameters
    parser.add_argument('--load-checkpoint', type=str, default=None, help='path to load checkpoint to resume training from')
    parser.add_argument('--hidden-channels', type=int, default=256)
    parser.add_argument('--num-downsamples'  , type=int, default=3, help='total downsampling is 2**[this]')
    parser.add_argument('--vq-num-embeddings', type=int, default=32, help='aka codebook length')
    parser.add_argument('--vq-embedding-dim' , type=int, default=256, help='pre-vq emb dim  before compression')
    parser.add_argument('--compressed-dim' , type=int, default=4, help='ACTUAL dims of codebook vectors')
    parser.add_argument('--codebook-levels'  , type=int, default=4, help='number of RVQ levels')
    parser.add_argument('--commitment-weight'  , type=float, default=0.5, help='VQ commitment weight, aka quantization strength')
    parser.add_argument('--project-name', type=str, default="vqgan-midi", help='WandB project name')
    parser.add_argument('--run-name', type=str, default=None, help='WandB run name')
    parser.add_argument('--no-grad-ckpt', action='store_true', help='disable gradient checkpointing (disabled uses more memory but faster)') 

    args = parser.parse_args()
    args.learning_rate = args.base_lr * math.sqrt(args.batch_size / 32)
    print("args = ",args)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    # Data setup
    train_loader, val_loader = create_image_loaders(batch_size=args.batch_size, image_size=args.image_size, 
                                            data_path=args.data, num_workers=16)
    fakez_train_dl = None
    fakez_val_dl = None
    sinkhorn_loss = None
    # if args.data_fakez is not None: # don't worry about this
    #     fakez_train_dl, fakez_val_dl = create_fakez_dataloaders(
    #         args.data_fakez, 
    #         batch_size=args.batch_size, 
    #         rand_length=(len(train_loader) * args.batch_size + len(val_loader) * args.batch_size)
    #     )
    #     print("len(fakez_train_dl), len(fakez_val_dl) =",len(fakez_train_dl), len(fakez_val_dl))
    #     #sinkhorn_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)#, backend="multiscale")

    # Initialize models and losses
    vgg = vgg16(weights='IMAGENET1K_V1').features[:16].to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False
    
    model = VQVAE(
        in_channels=3,
        hidden_channels=args.hidden_channels,
        num_downsamples=args.num_downsamples,
        vq_num_embeddings=args.vq_num_embeddings,
        vq_embedding_dim=args.vq_embedding_dim,
        codebook_levels=args.codebook_levels,
        compressed_dim=args.compressed_dim,
        commitment_weight=args.commitment_weight,
        use_checkpoint=not args.no_grad_ckpt,
    ).to(device)

    if args.data_fakez is not None:
        print("Freezing encoder and compress")
        model = freeze_front_vqvae(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = None # optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=args.epochs)
    
    adv_loss = AdversarialLoss(device, use_checkpoint=not args.no_grad_ckpt).to(device)
    d_optimizer = optim.Adam(adv_loss.discriminator.parameters(), 
                            lr=args.learning_rate * 0.1, 
                            weight_decay=1e-5)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        # model.load_state_dict(checkpoint['model_state_dict'])
        model = load_non_frozen(model,checkpoint)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")



    # Initialize wandb
    if not args.no_wandb:
        wandb.init(project=args.project_name, name=args.run_name)
        wandb.config.update(vars(args))

    os.makedirs('checkpoints', exist_ok=True)

    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        if epoch == args.warmup_epochs:
            print("*** WARMUP PERIOD FINISHED. Engaging adversarial training. ***")

        model.train()
        epoch_losses = defaultdict(float)
        total_batches = 0
        
        # Create fresh fakez training iterator at start of epoch
        if fakez_train_dl is not None:
            print(f"Creating new fakez train iterator. fakez_train_dl has {len(fakez_train_dl)} batches")
            fakez_train_iterator = iter(fakez_train_dl)

        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            source_imgs = batch[0].to(device)
            target_imgs = source_imgs
            
            # Get fakez batch, cycling through data if needed
            if fakez_train_dl is not None:
                try:
                    fakez_batch = next(fakez_train_iterator)
                except StopIteration:
                    print("Recreating fakez train iterator after StopIteration")
                    fakez_train_iterator = iter(fakez_train_dl)
                    try:
                        fakez_batch = next(fakez_train_iterator)
                    except StopIteration:
                        print("ERROR: New iterator also raised StopIteration!")
                        raise
            else:
                fakez_batch = None
            noise_strength = 0.0 if epoch < args.warmup_epochs else 0.1 * min(1.0, (epoch - args.warmup_epochs) / 10)

            # Pre-warmup training
            if epoch < args.warmup_epochs:
                recon, vq_loss = model(source_imgs)
                if fakez_batch is not None:
                    fakez = fakez_batch.to(device)
                    fakez_recon = z_to_img(fakez, model)
                    losses = compute_losses(recon, target_imgs, vq_loss, vgg, 
                                         adv_loss=None, epoch=epoch,  config=args,
                                         sinkhorn_loss=sinkhorn_loss, fakez_recon=fakez_recon)
                else:
                    losses = compute_losses(recon, target_imgs, vq_loss, vgg, 
                                         adv_loss=None, epoch=epoch, config=args)
                losses['total'] = get_total_loss(losses, args)
                
                optimizer.zero_grad()
                losses['total'].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Post-warmup training
            else:
                # Train discriminator
                recon, vq_loss = model(source_imgs, noise_strength=noise_strength)
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
                    recon, vq_loss = model(source_imgs, noise_strength=noise_strength)
                    if fakez_batch is not None:
                        fakez = fakez_batch.to(device)
                        fakez_recon = z_to_img(fakez, model)
                        losses = compute_losses(recon, target_imgs, vq_loss, vgg, 
                                             adv_loss=adv_loss, epoch=epoch, config=args,
                                             sinkhorn_loss=sinkhorn_loss, fakez_recon=fakez_recon)
                    else:
                        losses = compute_losses(recon, target_imgs, vq_loss, vgg, 
                                             adv_loss=adv_loss, epoch=epoch, config=args)

                    losses['total'] = get_total_loss(losses, args)

                    optimizer.zero_grad()
                    losses['total'].backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                log_dict = {
                    'epoch': epoch,
                    **{f'batch/{k}_loss': v for k, v in batch_losses.items()}
                }
                wandb.log(log_dict)

        # Compute average training losses
        train_losses = {k: v / total_batches for k, v in epoch_losses.items()}

        # Validation phase
        with torch.no_grad():
            # do some gpu vram garbage collection? 
            model.eval()
            val_losses = defaultdict(float)
            val_total_batches = 0

            # Create fresh fakez validation iterator
            if fakez_val_dl is not None:
                print(f"Creating new fakez val iterator. fakez_val_dl has {len(fakez_val_dl)} batches")
                fakez_val_iterator = iter(fakez_val_dl)

            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    source_imgs = batch[0].to(device)
                    target_imgs = source_imgs

                    # Get fakez validation batch
                    if fakez_val_dl is not None:
                        try:
                            fakez_batch = next(fakez_val_iterator)
                        except StopIteration:
                            print("Recreating fakez val iterator after StopIteration")
                            fakez_val_iterator = iter(fakez_val_dl)
                            try:
                                fakez_batch = next(fakez_val_iterator)
                            except StopIteration:
                                print("ERROR: New validation iterator also raised StopIteration!")
                                raise
                    else:
                        fakez_batch = None

                    # Basic validation forward pass
                    recon, vq_loss, dist_stats = model(source_imgs, get_stats=True)
                    if fakez_val_dl is not None:
                        with torch.no_grad(): # validation so we don't need grads
                            fakez_batch = next(iter(fakez_val_dl))
                            fakez = fakez_batch.to(device)  # OOM ERROR HAPPENS HERE
                            fakez_recon = z_to_img(fakez, model)
                            losses = compute_losses(recon, target_imgs, vq_loss, vgg,
                                                adv_loss=adv_loss, epoch=epoch, config=args,
                                                sinkhorn_loss=sinkhorn_loss, fakez_recon=fakez_recon)
                    else:
                        losses = compute_losses(recon, target_imgs, vq_loss, vgg,
                                            adv_loss=adv_loss, epoch=epoch, config=args)
                    losses['total'] = get_total_loss(losses, args)

                    # Update validation losses
                    batch_losses = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
                    val_total_batches += 1
                    for k, v in batch_losses.items():
                        val_losses[k] += v

                    # Log validation visualizations for first batch
                    if batch_idx == 0 and not args.no_wandb:
                        # Collect validation metrics
                        note_metrics, metric_images = calculate_note_metrics(recon, target_imgs)
                        metric_grids = {k: make_grid(v[:8], nrow=8, normalize=True) 
                                    for k, v in metric_images.items()}
                        
                        log_dict = {
                            'epoch': epoch,
                            'codebook/mean_distance': dist_stats['codebook_mean_dist'],
                            'codebook/max_distance': dist_stats['codebook_max_dist'],
                            **{f'note_metrics/{k}': v for k, v in note_metrics.items()},
                            **{f'validation/batch_{k}_loss': v for k, v in val_losses.items()},
                            **{f'metric_images/{k}': wandb.Image(v, caption=k) for k, v in metric_grids.items()}
                        }

                        # Add visualization grid
                        nrow=8  # number of examples of each to show
                        orig = batch[0][:nrow].to(device)
                        recon = torch.clamp(recon[:8], orig.min(), orig.max())
                        orig, recon = g2rgb(orig), g2rgb(recon)
                        
                        if fakez_val_dl is not None:
                            fakez_batch = next(iter(fakez_val_dl))
                            fakez = fakez_batch[:nrow].to(device)
                            fakez_recon = z_to_img(fakez, model)
                            fakez_recon = torch.clamp(fakez_recon, orig.min(), orig.max())
                            fakez_recon = g2rgb(fakez_recon)
                            viz_images = torch.cat([orig, recon, fakez_recon])
                            caption = f'Epoch {epoch} - Top: Source, Middle: Recon, Bottom: Decoded Fakez'
                        else:
                            viz_images = torch.cat([orig, recon])
                            caption = f'Epoch {epoch} - Top: Source, Bottom: Recon'
                        
                        log_dict['demo/examples'] = wandb.Image(
                            make_grid(viz_images, nrow=8, normalize=True),
                            caption=caption
                        )

                        wandb.log(log_dict)

        # Compute average validation losses
        val_losses = {k: v / val_total_batches for k, v in val_losses.items()}

        # Visualize codebooks periodically
        if epoch % 1 == 0:
            viz_codebooks(model, args, epoch)

        # Log epoch metrics
        if not args.no_wandb:
            log_dict = {
                'epoch': epoch,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **{f'epoch/train_{k}_loss': v for k, v in train_losses.items()},
                **{f'epoch/val_{k}_loss': v for k, v in val_losses.items()}
            }
            wandb.log(log_dict)

        # Save checkpoint
        if (epoch + 1) % 250 == 0 and epoch > 0:
            ckpt_path = f'checkpoints/model_epoch{epoch}.pt'
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)

        if scheduler:
            scheduler.step()
            
if __name__ == '__main__':
    main()
