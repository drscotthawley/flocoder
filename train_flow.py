#! /usr/bin/env python3

## NOTICE: This code is a work in progress and may not run as-is.

import os
import re
import math 
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import wandb
from torchvision import datasets, transforms
from scipy import integrate  # This is CPU-only
from tqdm.auto import tqdm
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
import hydra
from omegaconf import DictConfig

from flocoder.unet import Unet, MRUnet
from flocoder.codecs import load_codec
from flocoder.data import PreEncodedDataset, InfiniteDataset
from flocoder.general import save_checkpoint, keep_recent_files, handle_config_path

import gc


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


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def rk45_sampler(model, shape, device, eps=0.001, n_classes=0, cfg_scale=3.0):
    """Runge-Kutta '4.5' order method for integration. Source: Tadao Yamaoka"""
    rtol = atol = 1e-05
    model.eval()
    # Create a grid where each column is a single class (10 columns)
    cond = None # the conditioning signal to the model
    if n_classes > 0:
        cond = torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device)
    
    with torch.no_grad():
        # The rest remains the same
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t

            # classifier-free guidance
            v_cond = model(x, vec_t * 999, cond)
            v_uncond = model(x, vec_t * 999, None)
            velocity = v_uncond + cfg_scale * (v_cond - v_uncond) # cfg_scale ~= conditioning strength

            return to_flattened_numpy(velocity)

        # Rest of the implementation unchanged
        solution = integrate.solve_ivp(
            ode_func, (eps, 1), to_flattened_numpy(x),
            rtol=rtol, atol=atol, method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)
        model.train()
        return x, nfe


def imshow(img, filename):
    imin, imax = img.min(), img.max()
    img = (img - imin) / (imax - imin) # rescale via max/min
    img = np.clip(img, 0, 1)
    npimg = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(npimg)
    pil_img.save(filename)


def save_img_grid(img, epoch, method, nfe, tag="", use_wandb=True, output_dir="output"):
    """Save image grid with consistent 10-column layout to match class conditioning"""
    filename = f"{method}_epoch_{epoch + 1}_nfe_{nfe}.png"
    # Use nrow=10 to ensure grid columns match our class conditioning
    img_grid = torchvision.utils.make_grid(img, nrow=10)
    file_path = os.path.join(output_dir, filename)
    imshow(img_grid, file_path)
    name = f"demo/{tag}{method}"
    if 'euler' in name: name = name + f"_nf{nfe}"
    if use_wandb: wandb.log({name: wandb.Image(file_path, caption=f"Epoch: {epoch}")})


@torch.no_grad()
def eval(model, codec, epoch, method, device, sample_N=None, batch_size=100, tag="", 
         images=None, use_wandb=True, output_dir="output", n_classes=0, latent_shape=(4,16,16)):
    """Evaluate model by generating samples with class conditioning"""
    model.eval()
    # Use a batch size that's a multiple of 10 to ensure proper grid layout
    # For a 10x10 grid, use batch_size = 100
    batch_size=100 # hard code this for the image display; ignore other batch size values
    shape = (batch_size)+latent_shape

    if images is None:
        if method == "euler":
            images, nfe = euler_sampler(model, shape=shape, sample_N=sample_N, device=device, n_classes=n_classes)
        elif method == "rk45":
            images, nfe = rk45_sampler(model, shape=shape, device=device, n_classes=n_classes)
    else:
        nfe=0

    decoded_images = codec.decode(images.to(device))
        
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



def train_flow(cfg):
    # Extract parameters from config
    data_path = f"{cfg.data}_encoded_{cfg.codec.choice}"
    batch_size = cfg.training.batch_size
    n_classes = cfg.model.n_classes
    condition = cfg.model.condition
    lambda_lowres = cfg.training.get('lambda_lowres', 0.1)
    learning_rate = cfg.training.learning_rate
    epochs = cfg.training.num_epochs
    project_name = cfg.wandb.project_name
    run_name = cfg.wandb.get('run_name', None)
    no_wandb = cfg.get('no_wandb', False)
    
    print(f"data_path = {data_path}")

    dataset = PreEncodedDataset(data_path) 
    sample_item, _ = dataset[0]
    latent_shape = sample_item.shape
    C, H, W = latent_shape
    print(f"Detected latent dimensions: C={C}, H={H}, W={W}")
    
    # If n_classes is not specifically set but the dataset has this info, use it
    if n_classes <= 0 and hasattr(dataset, 'n_classes'):
        n_classes = dataset.n_classes
        print(f"Using dataset-provided n_classes: {n_classes}")
    
    # If we're using conditioning but don't have class info, warn and disable
    if condition and (n_classes is None or n_classes <= 0):
        print("Warning: Conditioning requested but no class information available. Disabling conditioning.")
        condition = False
        n_classes = 0

    print(f"Configuring model for {n_classes} classes, conditioning = {condition}\n")
    #C, H, W = 4, 16, 16 # TODO: read this from data!!

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
    )
    dataloader = train_dataloader # alias to avoid errors later

    output_dir = f"output_{data_path.split('/')[-1]}-{H}x{W}" 
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    codec = load_codec(cfg, device) # Load codec for inference/evaluation

    # Create flow model
    model = Unet(
        dim=H,
        channels=C,
        dim_mults=(1, 2, 4),
        condition=condition,
        n_classes=n_classes,
    )
    model.to(device)
    print(f"Model conditioning enabled: {hasattr(model, 'condition') and model.condition}")
    print(f"Model has cond_mlp: {hasattr(model, 'cond_mlp')}")

    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    use_wandb = not no_wandb
    
    if use_wandb: 
        wandb.init(project=project_name, name=run_name, config=dict(cfg))

    ema = EMA(model, decay=0.999, device=device)

    rolling_avg_loss, alpha = None, 0.99 
    eps = 0.001 # used in integration
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{epochs}:')
        for batch, cond in pbar:
            batch, cond = batch.to(device), cond.to(device)
            if random.random() < 0.1: cond = None # for classifier-free guidance: turn off cond signal sometimes
            target = batch # alias

            optimizer.zero_grad()

            source = torch.randn_like(batch)
            t = torch.rand(batch.shape[0], device=device) * (1 - eps) + eps
            t = warp_time(t)          

            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1 - t_expand) * source

            v_guess = target - source  #batch - z0
            v_model = model(perturbed_data, t * 999, cond)
            loss = loss_fn(v_model, v_guess)
        
            if hasattr(model,'shrinker'):  # if multiresolution is available
                lowres_v_guess = model.shrinker(v_guess)
                lowres_loss = loss_fn(model.bottleneck_target_hook, lowres_v_guess) # compare with hook
                loss = loss + lambda_lowres * lowres_loss

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
                    "epoch": epoch 
                })
            
            pbar.set_postfix({"Loss/train":f"{loss.item():.4g}"})
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # gradient clipping for stability

            optimizer.step()
            ema.update()

        if (epoch < 10 and epoch % 1 == 0) or (epoch >= 10 and ((epoch + 1) % 10 == 0)): 
            # evals more frequently at beginning, then every 10 epochs later
            print("Generating sample outputs...")
            images = batch[:100].cpu()
            batch, score, target = None, None, None
            gc.collect()  # force clearing of GPU memory cache
            torch.cuda.empty_cache()
            eval_kwargs = {'use_wandb': use_wandb, 'output_dir': output_dir, 'n_classes': n_classes, 
               'latent_shape': latent_shape, 'batch_size': 100}
            eval(model, codec, epoch, "target_data", device, tag="", images=images, **eval_kwargs)
            eval(model, codec, epoch, "rk45", device, tag="", **eval_kwargs)
            ema.eval()
            eval(model, codec, epoch, "rk45", device, tag="ema_", **eval_kwargs)
            ema.train()

        if (epoch + 1) % 25 == 0: # checkpoint every 25 epochs
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flow_", ckpt_dir=f"checkpoints")
            ema.eval()  # Switch to EMA weights
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flowema_", ckpt_dir=f"checkpoints")
            ema.train()  # Switch back to regular weights
            keep_recent_files(100, directory=output_dir, pattern="*.png") # not too many image outputs


handle_config_path()
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers")
def main(cfg: DictConfig) -> None:
    print("Config keys:", list(cfg.keys()))     
    train_flow(cfg)


if __name__ == "__main__":
    main()
