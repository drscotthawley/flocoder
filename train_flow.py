#! /usr/bin/env python3

import os
import sys
import random
import gc
import itertools
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm
import hydra
from omegaconf import OmegaConf

from natten import use_fused_na
use_fused_na()


from flocoder.unet import Unet, MRUnet
from flocoder.codecs import setup_codec
from flocoder.data import PreEncodedDataset, InfiniteDataset, create_image_loaders
from flocoder.general import save_checkpoint, keep_recent_files, handle_config_path, ldcfg, CosineAnnealingWarmRestartsDecay
from flocoder.sampling import sampler, warp_time, evaluate_model
from flocoder.hdit import ImageTransformerDenoiserModelV2, LevelSpec, MappingSpec, GlobalAttentionSpec
from flocoder.codebook_analysis import CodebookUsageTracker


print("Imports successful!") 


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


def train_flow(config):
    # Extract parameters from config
    data_path = f"{config.data}_encoded_{config.codec.choice}"
    batch_size = ldcfg(config,'batch_size')
    n_classes = ldcfg(config.flow.unet,'n_classes', 0,verbose=True)
    if n_classes == 0: 
        condition = False
    else: 
        condition = ldcfg(config,'condition',False)
    learning_rate = ldcfg(config,'learning_rate')
    epochs = ldcfg(config,'epochs')
    project_name = ldcfg(config,'project_name')
    run_name = ldcfg(config, 'run_name')
    no_wandb = ldcfg(config,'no_wandb', False)
    lambda_lowres = config.flow.get('lambda_lowres', 0.1) # might not have it
    is_midi = any(x in data_path.lower() for x in ['pop909', 'midi'])
    keep_gray = ldcfg(config.codec,'in_channels',3)==1
    print("is_midi, keep_gray =",is_midi, keep_gray)

    # If we're using conditioning but don't have class info, warn and disable
    if condition and (n_classes is None or n_classes <= 0):
        print("Warning: Conditioning requested but no class information available. Disabling conditioning.")
        condition = False
        n_classes = 0
    
    print(f"data_path = {data_path}")
    train_path, val_path = f"{data_path}/train",  f"{data_path}/val"

    print(f"Loading train data from: {train_path}")
    pre_encoded=True 
    if pre_encoded:
        train_dataset = PreEncodedDataset(train_path, n_classes=n_classes)
        train_dataloader = DataLoader( dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, 
                pin_memory=True, persistent_workers=True,)
        dataset, dataloader = train_dataset, train_dataloader # aliases to avoid errors later
    
        print(f"Loading val data from: {val_path}")
        val_dataset = PreEncodedDataset(val_path, n_classes=n_classes)
        val_dataloader = DataLoader( dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=12, 
                pin_memory=True, persistent_workers=True,)

        sample_item, _ = dataset[0]
    else: 
        image_size = ldcfg(config, 'image_size', 128)
        num_workers = ldcfg(config,'num_workers', 16)
        is_midi = True
        train_dataloader, val_dataloader = create_image_loaders( batch_size=batch_size, image_size=image_size, 
                data_path=data_path, num_workers=num_workers, is_midi=is_midi, config=config)
        batch = next(iter(train_dataloader))
        source_imgs, _, target_imgs, _b = batch
        sample_item = source_imgs[0]

    latent_shape = tuple(sample_item.shape)
    C, H, W = latent_shape
    print(f"Detected latent dimensions: C={C}, H={H}, W={W}")

    print(f"Configuring model for {n_classes} classes, conditioning = {condition}\n")
    #C, H, W = 4, 16, 16 # TODO: read this from data!!

    output_dir = f"output_{data_path.split('/')[-1]}-{H}x{W}" 
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    codec = setup_codec(config, device).eval() # Load codec for inference/evaluation
    # variables for tracking codebook usage
    cb_levels = ldcfg(config, 'codebook_levels', 4)
    cb_size = ldcfg(config, 'vq_num_embeddings', 32)
    cb_tracker = CodebookUsageTracker(num_levels=cb_levels, codebook_size=cb_size)

    # Create flow model 
    dim_mults = ldcfg(config,'dim_mults', [1,2,4,4], verbose=True)
    if pre_encoded: # usually we want this
        model = Unet( dim=H, channels=C, dim_mults=dim_mults, condition=condition, n_classes=n_classes,).to(device)
    else:  # added to test HDiT
        levels = [
            LevelSpec(depth=2, width=256, d_ff=768, self_attn=GlobalAttentionSpec(d_head=64), dropout=0.0),
            LevelSpec(depth=4, width=512, d_ff=1536, self_attn=GlobalAttentionSpec(d_head=64), dropout=0.0),
        ]
        mapping = MappingSpec(depth=2, width=256, d_ff=768, dropout=0.0)
        print("model is ImageTransformerDenoiserModelV2")
        model = ImageTransformerDenoiserModelV2(
            levels=levels,  # List of LevelSpec objects
            mapping=mapping,  # MappingSpec object
            in_channels=C,
            out_channels=C,
            patch_size=(4, 4),  # or (2, 2) for smaller images
            num_classes=n_classes if n_classes else 0,
            mapping_cond_dim=0  # unless you have extra conditioning
        )
    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total/1e9:.1f}B" if total >= 1e9 else f"Model params: {total/1e6:.1f}M")
    print(f"Model conditioning enabled: {hasattr(model, 'condition') and model.condition}")
    print(f"Model has cond_mlp: {hasattr(model, 'cond_mlp')}")
    print("")


    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=50, T_mult=2, decay=0.6) # previously T_mult=1, decay=1
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=200, T_mult=2, decay=0.3) 

    use_wandb = not no_wandb
    if use_wandb: 
        wandb.init(project=project_name, name=run_name, config=dict(config))

    ema = EMA(model, decay=0.999, device=device)

    rolling_avg_loss, alpha = None, 0.99 
    eps = 0.001 # used in integration
    for epoch in range(epochs):
        model.train()
    
        #short_loader = itertools.islice(train_dataloader, 10)  # Only take 10 batches  ; #temp for testing. import itertools
        #pbar = tqdm(short_loader, desc=f'Epoch {epoch}/{epochs}:', mininterval=0.25)
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{epochs}:', mininterval=0.25)
        for batch in pbar:
            if pre_encoded: 
                batch, cond = batch
            else: 
                _, _a, batch, cond = batch 

            batch, cond = batch.to(device), cond.to(device)
            cond=None # TODO: this is hard-coded for MIDI data right now. fix this and make it adaptive, re. using config.conditioned
            if random.random() < 0.1: cond = None # for classifier-free guidance: turn off cond signal sometimes
            target = batch # alias

            optimizer.zero_grad()

            source = torch.randn_like(batch)
            t = torch.rand(batch.shape[0], device=device) * (1 - eps) + eps
            t = warp_time(t)          

            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            x = t_expand * target + (1 - t_expand) * source

            v_guess = target - source    # constant velocity
            t_scale = 999 if not pre_encoded else 1
            #v_model = model(x, t * t_scale, aug_cond=None, class_cond=cond) for use with HDiT 
            v_model = model(x, t * t_scale, cond=cond)
            loss = loss_fn(v_model, v_guess)
        
            if hasattr(model,'shrinker'):  # optional: if multiresolution UNet is available
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
            
            pbar.set_postfix({"Loss/train":f"{loss.item():.4g}"}, refresh=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping for stability

            optimizer.step()
            ema.update()

        # Evals / Metrics / Viz
        if (epoch < 20 ) or (epoch >= 20 and ((epoch + 1) % 10 == 0)):
            model.eval()
            val_batch = next(iter(val_dataloader))  # Fixed missing parenthesis
            if len(val_batch)>2: 
                _, _a, val_batch, val_cond = val_batch
            else:
                val_batch, val_cond = val_batch

            val_batch, val_cond = val_batch.to(device), val_cond.to(device)

            # evals more frequently at beginning, then every 10 epochs later
            print("Generating sample outputs...")
            eval_kwargs = {
                'method':'rk4', 'use_wandb': use_wandb, 'output_dir': output_dir,
                'n_classes': n_classes, 'latent_shape': latent_shape, 'batch_size': 768,
                'is_midi':is_midi, 'keep_gray':keep_gray,
                'pre_encoded': pre_encoded
            }
            metrics = evaluate_model( model, codec, epoch, val_batch, target_labels=val_cond, tag="", cb_tracker=cb_tracker, **eval_kwargs)

            if epoch>50: # no point doing ema earlier
                ema.eval()
                evaluate_model(model, codec, epoch, val_batch, target_labels=val_cond, tag="ema_", **eval_kwargs)
                ema.train()
            model.train()

            if epoch % 5 == 0: cb_tracker.reset_all()  # accumulate codebook usage info over 5 epochs

        # Checkpoints
        if (epoch + 1) % 25 == 0: # checkpoint every 25 epochs
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flow_", ckpt_dir=f"checkpoints", config=config)
            ema.eval()  # Switch to EMA weights
            save_checkpoint(model, epoch=epoch, optimizer=optimizer, keep=5, prefix="flowema_", ckpt_dir=f"checkpoints", config=config)
            ema.train()  # Switch back to regular weights
            keep_recent_files(100, directory=output_dir, pattern="*.png") # not too many image outputs

        if False and epoch % 50 == 0 and epoch > 0:  # cheap batch size increaser. TODO: try https://github.com/ancestor-mithril/bs-scheduler
            batch_size = min(batch_size * 2, 12000) # increase batch size
            print("Setting batch size to",batch_size,", remaking DataLoader.")
            train_dataloader = DataLoader( dataset=dataset,
                batch_size=batch_size,  # Use the updated value
                shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True,)

        scheduler.step()
    # end of epoch loop


handle_config_path()
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers")
def main(config):
    OmegaConf.set_struct(config, False)  # make config mutable
    print("Config:", config)     
    train_flow(config)


if __name__ == "__main__":
    main()
