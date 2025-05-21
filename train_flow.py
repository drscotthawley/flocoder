#! /usr/bin/env python3

import os
import random
import gc
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm
import hydra

from flocoder.unet import Unet, MRUnet
from flocoder.codecs import load_codec
from flocoder.data import PreEncodedDataset, InfiniteDataset
from flocoder.general import save_checkpoint, keep_recent_files, handle_config_path
from flocoder.sampling import sampler, warp_time


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


def train_flow(cfg):
    # Extract parameters from config
    data_path = f"{cfg.data}_encoded_{cfg.codec.choice}"
    batch_size = cfg.flow.batch_size
    n_classes = cfg.model.n_classes
    condition = cfg.model.condition
    lambda_lowres = cfg.flow.get('lambda_lowres', 0.1)
    learning_rate = cfg.flow.learning_rate
    epochs = cfg.flow.epochs
    project_name = cfg.project_name
    run_name = cfg.get('run_name', None)
    no_wandb = cfg.get('no_wandb', False)
    
    print(f"data_path = {data_path}")

    dataset = PreEncodedDataset(data_path) 
    sample_item, _ = dataset[0]
    latent_shape = tuple(sample_item.shape)
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
            sampler(model, codec, epoch, "target_data", device, tag="", images=images, **eval_kwargs)
            sampler(model, codec, epoch, "rk45", device, tag="", **eval_kwargs)
            ema.eval()
            sampler(model, codec, epoch, "rk45", device, tag="ema_", **eval_kwargs)
            ema.train()

        if (epoch + 1) % 25 == 0: # checkpoint every 25 epochs
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flow_", ckpt_dir=f"checkpoints")
            ema.eval()  # Switch to EMA weights
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flowema_", ckpt_dir=f"checkpoints")
            ema.train()  # Switch back to regular weights
            keep_recent_files(100, directory=output_dir, pattern="*.png") # not too many image outputs


handle_config_path()
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers")
def main(cfg):
    print("Config:", cfg)     
    train_flow(cfg)


if __name__ == "__main__":
    main()
