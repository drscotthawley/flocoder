#! /usr/bin/env python3

## NOTICE: This code is a work in progress and may not run as-is.

import os
import re
import math 
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
#from scipy import integrate # this is CPU only ewww
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter  # SHH: I prefer wandb to tb
import wandb
from torchvision import datasets, transforms

from scipy import integrate # this is CPU-only ewww

from tqdm.auto import tqdm
#from dist_metrics import compare_distributions, make_histograms
from PIL import Image
from pathlib import Path

from flocoder.models.unet import Unet
#from flocoder.models.vqvae import VQVAE
from flocoder.data.datasets import PreEncodedDataset, InfiniteDataset
from flocoder.utils.general import save_checkpoint, keep_recent_files

import gc


# image_size = (16,16) # (64, 64) 
# channels = 4
# C, H, W = channels, image_size[0], image_size[1]
# batch_size=12288//3
# print("batch_size = ",batch_size)
# B = batch_size
# learning_rate = 1e-3
# num_epochs = 999999 # 1000  or just let it go til the disk fills with checkpoints TODO: keep only last few checkpoints
# eps = 0.001 # used in integration
# condition = False  # Enaable class-conditioning
# n_classes = None # updates in main()


class EMA:
    """Exponential Moving Average (EMA) for model parameters."""
    def __init__(self, model, decay=0.99, device=None):
        self.model = model
        self.decay = decay
        self.device = device if device is not None else 'cuda'
        self.shadow = {}
        self.backup = {}

        # Initialize on CPU
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.cpu().clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    # Move shadow to device, do computation, then back to CPU
                    new_average = (self.decay * self.shadow[name].to(self.device) +
                                 (1.0 - self.decay) * param.data)
                    self.shadow[name] = new_average.cpu().clone()

    def eval(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store current parameters and move them to CPU
                self.backup[name] = param.data.cpu().clone()
                # Move EMA parameters to device
                param.data = self.shadow[name].to(self.device)

    def train(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Restore original parameters from CPU to device
                param.data = self.backup[name].to(self.device)
                # Clear backup to free memory
                self.backup[name] = None


def euler_sampler(model, shape, sample_N, device, eps=0.001):
    """quick and dirty integration using Euler method; not very accurate"""
    model.eval()
    #cond = torch.arange(10).repeat(shape[0] // 10).to(device) if condition else None
    cond = torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device) # this is for a 10x10 grid of outputs, with one class per column
    with torch.no_grad():
        z0 = torch.randn(shape, device=device) # gaussian noise
        x = z0.detach().clone()

        dt = 1.0 / sample_N
        for i in range(sample_N):
            num_t = i / sample_N * (1 - eps) + eps
            t = torch.ones(shape[0], device=device) * num_t
            pred = model(x, t * 999, cond)

            x = x.detach().clone() + pred * dt
        nfe = sample_N # number of function evaluations
        return x.cpu(), nfe


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def rk45_sampler(model, shape, device, eps=0.001):
    """Runge-Kutta '4.5' order method for integration. Source: Tadoa Yomaoka"""
    rtol = atol = 1e-05
    model.eval()
    #cond = torch.arange(n_classes).repeat(shape[0] // n_classes).to(device) if condition else None 
    cond = None# torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device) # this is for a 10x10 grid of outputs, with one class per column
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model(x, vec_t * 999)#, cond)

            return to_flattened_numpy(drift)

        solution = integrate.solve_ivp(
            ode_func,
            (eps, 1),
            to_flattened_numpy(x),
            rtol=rtol,
            atol=atol,
            method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)
        model.train()
        return x, nfe


def imshow_old(img, filename):
    #metrics = compare_distributions(img, target_img)  # TODO: create target_img
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    npimg = img.permute(1, 2, 0).numpy()
    plt.imshow(npimg)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)

def imshow(img, filename):
    #img = img * 0.5 + 0.5
    imin, imax = img.min(), img.max()
    img = (img - imin) / (imax - imin) # rescale via max/min
    img = np.clip(img, 0, 1)
    npimg = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(npimg)
    pil_img.save(filename)


def save_img_grid(img, epoch, method, nfe, tag="", use_wandb=True, output_dir="output"):
    filename = f"{method}_epoch_{epoch + 1}_nfe_{nfe}.png"
    img_grid = torchvision.utils.make_grid(img, nrow=10)
    #print(f"img.shape = {img.shape}, img_grid.shape = {img_grid.shape}") 
    file_path = os.path.join(output_dir, filename)
    imshow(img_grid, file_path)
    name = f"demo/{tag}{method}"
    if 'euler' in name: name = name + f"_nf{nfe}"
    if use_wandb: wandb.log({name: wandb.Image(file_path, caption=f"Epoch: {epoch}")})


@torch.no_grad()
def eval(model, vae, epoch, method, device, sample_N=None, batch_size=100, tag="", images=None, use_wandb=True, output_dir="output"):
    model.eval()
    # saves sample generated images, unless images are passed through (target data)
    # TODO: Refactor. this is janky. images are not the same as latents. 

    shape = (batch_size, 4, 16, 16) # note we need to decode these too. 

    if images is None:
        if method == "euler":
            images, nfe = euler_sampler( model, shape=shape, sample_N=sample_N, device=device)
        elif method == "rk45":
            images, nfe = rk45_sampler(model, shape=shape, device=device)
    else:
        nfe=0
    #save_img_grid(images, f"{method}_epoch_{epoch + 1}_nfe_{nfe}.png")
    if vae.is_sd: 
        decoded_images = vae.decode(images.to(device)).sample
    else:
        decoded_images = vae.decode(images.to(device))
    save_img_grid(images.cpu(), epoch, method, nfe, tag=tag, use_wandb=use_wandb, output_dir=output_dir)
    save_img_grid(decoded_images.cpu(), epoch, method, nfe, tag=tag+"decoded_", use_wandb=use_wandb, output_dir=output_dir)
    return model.train()






def warp_time(t, dt=None, s=.5):
    """Parametric Time Warping: s = slope in the middle.
        s=1 is linear time, s < 1 goes slower near the middle, s>1 goes slower near the ends
        s = 1.5 gets very close to the "cosine schedule", i.e. (1-cos(pi*t))/2, i.e. sin^2(pi/2*x)"""
    if s<0 or s>1.5: raise ValueError(f"s={s} is out of bounds.")
    tw = 4*(1-s)*t**3 + 6*(s-1)*t**2 + (3-2*s)*t
    if dt:                           # warped time-step requested; use derivative
        return tw,  dt * 12*(1-s)*t**2 + 12*(s-1)*t + (3-2*s)
    return tw


def train_flow(args):
    global n_classes  # TODO: this is janky


    print(f"====================   data_path = {args.data}")

    dataset = PreEncodedDataset(args.data) # "/data/encoded-POP909")
    #dataset = InfiniteDataset(dataset)
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        #shuffle=True,
        num_workers=12,
        pin_memory=True
    )
    dataloader = train_dataloader # alias to avoid errors later

    n_classes = 0
    condition = False  # Enable class-conditioning
    C, H,W = 4, args.image_size, args.image_size  # TODO: get this from sample data 

    print(f"Configuring model for {n_classes} classes\n")
    output_dir = f"output_flowers-sd-{H}x{W}" # TODO: fix this.
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=",device)

    print("creating (VQ)VAE model...")

    # Initialize (VQ)VAE - we'll only use the decoder for viz & eval, since our data is pre-encoded
    if 'SD' in args.vqgan_checkpoint or 'stable-diffusion' in args.vqgan_checkpoint:
        try:
            from diffusers.models import AutoencoderKL
        except ImportError:
            raise ImportError("To use SD VAE, you need to install diffusers. Try: pip install diffusers")

        print(f"Loading (VQ)VAE checkpoint from HuggingFace Diffusers")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)
        vae.is_sd = True
        vae.scaling_factor = 0.18215  # SD's standard scaling factor
        if args.image_size % 8 != 0:
            print(f"Warning: SD VAE works best with image sizes divisible by 8. Current size: {args.image_size}")
    else:
        from flocoder.models.vqvae import VQVAE
        vqvae = VQVAE(
            in_channels=3,
            hidden_channels=args.hidden_channels,
            num_downsamples=args.num_downsamples,
            vq_num_embeddings=args.vq_num_embeddings,
            internal_dim=args.internal_dim,
            vq_embedding_dim=args.vq_embedding_dim,
            codebook_levels=args.codebook_levels,
            use_checkpoint=not args.no_grad_ckpt,# this refers to gradient checkpointing
            no_natten=False,
        ).eval().to(device)

        # TODO: consistent use of either VQVAE or VQGAN
        vqgan_ckpt_file =args.vqgan_checkpoint #  "midi_vqvae_latest.pt"
        if not os.path.exists(vqgan_ckpt_file):
            raise FileNotFoundError(f"VQVAE checkpoint file {vqgan_ckpt_file} not found.")
        else:
            print(f"Loading VQVAE checkpoint from {vqgan_ckpt_file}")
        checkpoint = torch.load(vqgan_ckpt_file, map_location=device, weights_only=True)
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae.to(device)
        #vae.device = device
        print("(vq)vae checkpoints loaded")

    print("(vq)vae model ready")

    model = Unet(
        dim=H,
        channels=C,
        dim_mults=(1, 2, 4),
        condition=condition,
        n_classes=n_classes,
        #use_checkpoint=True,
    )
    model.to(device)
    #model_cp_file = "model_epoch_2300.pt"
    #checkpoint = torch.load(model_cp_file, map_location=device, weights_only=True)
    #model.load_state_dict(checkpoint)
    #print("model checkpoint loaded")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    use_wandb = not args.no_wandb
    if use_wandb: wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    ema = EMA(model, decay=0.999, device=device)

    rolling_avg_loss, alpha = None, 0.99 
    eps = 0.001 # used in integration
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{args.epochs}:')
        for batch, cond in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            z0 = torch.randn_like(batch)
            t = torch.rand(batch.shape[0], device=device) * (1 - eps) + eps
            t = warp_time(t)          

            t_expand = t.view(-1, 1, 1, 1).repeat(
                1, batch.shape[1], batch.shape[2], batch.shape[3]
            )
            #print("t.shape, t_expand.shape, batch.shape = ",t.shape, t_expand.shape, batch.shape)
            perturbed_data = t_expand * batch + (1 - t_expand) * z0
            target = batch - z0 # velocity

            score = model(
                perturbed_data, t * 999, cond.to(device) if condition else None
            )

            losses = torch.square(score - target)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)

            loss.backward()

            # for training stability : adaptive LR and gradient clipping
            if rolling_avg_loss is None:  
                rolling_avg_loss = loss.item()
            else:
                rolling_avg_loss = alpha * rolling_avg_loss + (1 - alpha) * loss.item()
                if loss.item() > 3 * rolling_avg_loss:  # Loss spike detected
                    optimizer.param_groups[0]['lr'] *= 0.5  # Halve the learning rate
            if use_wandb:
                wandb.log({
                    "Loss/train": loss.item(),
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch })
            pbar.set_postfix({"Loss/train":f"{loss.item():.4g}"})
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # gradient clipping for stability

            optimizer.step()
            ema.update()


        if (epoch < 10 and epoch %2==0 and epoch>0) or (epoch >= 10 and ((epoch + 1) % 10 == 0)): # evals every 1 at beginning, then every 20 later
            print("Generating sample outputs...")
            images = batch[:100].cpu()
            batch, score, target = None, None, None
            gc.collect()  # force clearing of GPU memory cache
            torch.cuda.empty_cache()
            eval(model, vae, epoch, "target_data", device, tag="", images=images, use_wandb=use_wandb, output_dir=output_dir)
            eval(model, vae, epoch, "rk45", device, tag="", use_wandb=use_wandb, output_dir=output_dir)
            ema.eval()
            eval(model, vae, epoch, "rk45", device, tag="ema_", use_wandb=use_wandb, output_dir=output_dir)
            ema.train()

        if (epoch + 1) % 25 == 0: # checkpoint every 100
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flow_", ckpt_dir=f"checkpoints")
            
            #torch.save( model.state_dict(), os.path.join(directory, f"model_epoch_{epoch + 1}.pt"),)
            # Save EMA weights
            ema.eval()  # Switch to EMA weights
            #torch.save( model.state_dict(),  # Now contains EMA weights
            #    os.path.join(directory, f"model_ema_epoch_{epoch + 1}.pt"))
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flowema_", ckpt_dir=f"checkpoints")

            ema.train()  # Switch back to regular weights
            #keep_recent_files(10, directory=directory, pattern="*.pt") # occasionally clean up the disk
            keep_recent_files(100, directory=output_dir, pattern="*.png") # not too many image outputs



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
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config_full = yaml.safe_load(f)
        
        config = {}  # Initialize config
        
        # Add global data if it exists
        if 'data' in yaml_config_full:
            config["data"] = yaml_config_full['data']
        
        # Process sections
        sections_to_process = ['vqgan', 'preencoding', 'flow'] # note later items overwrite earlier ones
        for section in sections_to_process:
            if section in yaml_config_full:
                section_params = yaml_config_full[section]
                
                # Flatten the nested structure
                for key, value in section_params.items():
                    if isinstance(value, dict):
                        # This is where the flattening happens for nested dicts
                        for subkey, subvalue in value.items():
                            config[subkey.replace('_', '-')] = subvalue
                    else:
                        config[key.replace('_', '-')] = value
        
        print("Flattened config from file:", config)
    
    # Create full parser with config-based defaults
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add config option
    parser.add_argument('--config', type=str, default=None, help='path to config YAML file')
    
    # Training parameters with config-based defaults
    parser.add_argument('--batch-size', type=int, default=config.get('batch-size', 32), 
                       help='the batch size')
    parser.add_argument('--data', type=str, 
                       default=config.get('data', None)+"_encoded", 
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
    parser.add_argument('--load-checkpoint', type=str, 
                       default=config.get('load-checkpoint', None), 
                       help='path to load checkpoint to resume training from')
    
    # vq parameters
    parser.add_argument('--vqgan-checkpoint', type=str, 
                       default=config.get('vqgan-checkpoint', None), 
                       help='path to load vqvgan checkpoint to resume training from')
    # TODO: vqgan model params should be included in checkpoint file. 
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
                       help='pre-vq emb dim before compression')
    parser.add_argument('--vq-embedding-dim', type=int, 
                       default=config.get('vq-embedding-dim', 4), 
                       help='(actual) dims of codebook vectors')
    parser.add_argument('--codebook-levels', type=int, 
                       default=config.get('codebook-levels', 4), 
                       help='number of RVQ levels')
    parser.add_argument('--commitment-weight', type=float, 
                       default=config.get('commitment-weight', 0.5), 
                       help='VQ commitment weight, aka quantization strength')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging')
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
    train_flow(args)