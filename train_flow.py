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
import numpy as np

from flocoder.unet import Unet
#from flocoder.unet import MRUnet # didn't help
from flocoder.codecs import setup_codec
from flocoder.data import PreEncodedDataset, InfiniteDataset, create_image_loaders
from flocoder.general import save_checkpoint, keep_recent_files, handle_config_path, ldcfg, CosineAnnealingWarmRestartsDecay
from flocoder.sampling import sampler, warp_time, evaluate_model
from flocoder.hdit import ImageTransformerDenoiserModelV2, LevelSpec, MappingSpec, GlobalAttentionSpec
from flocoder.codebook_analysis import CodebookUsageTracker
from flocoder.inpainting import MaskEncoder


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


def otf_gen_aug_indices(mask, p_ones=0.3, p_zeros=0.02):
    batch_size = mask.shape[0]
    n_ones = int(p_ones * batch_size)
    n_zeros = int(p_zeros * batch_size)

    ones_indices = np.random.choice(batch_size, n_ones, replace=False).tolist()
    remaining_indices = [i for i in range(batch_size) if i not in ones_indices]
    zeros_indices = np.random.choice(remaining_indices, min(n_zeros, len(remaining_indices)), replace=False).tolist()

    all_indices = set(range(batch_size))
    normal_indices = list(all_indices - set(ones_indices) - set(zeros_indices))

    return ones_indices, zeros_indices, normal_indices



def batch_to_data(batch, device, pre_encoded=True, mask_encoder=None,
         epoch=None, 
         curriculum_epochs=10,    # for the early epochs, we just do unconditional gen which is easier for the model to learn than mask geometries
         extend_epochs=20,
         ): 
    """main routine that unpacks dataloader batch (in train or val sets) to usable data
       also does some mask encoding... whatever data-prep ops are likely to be the same for train and val sets
    """
    mask, mask_pixels = None, None
    if pre_encoded:
        data, class_cond = batch
        if isinstance(data, dict):
            target = data['target_latents'].to(device)
            class_cond = class_cond.to(device)
            if mask_encoder is not None:
                mask_pixels = data['mask_pixels'].float() # use/save this for later debugging, but float makes it more versatile than bool
                if len(mask_pixels.shape) < 4: mask_pixels = mask_pixels.unsqueeze(1)  # add channel dim if needed (and it's typically needed)
                mask = mask_pixels.to(device)
                source = data['source_latents'].to(device)
        else:
            target, class_cond = data.to(device), class_cond.to(device)
    else:
        _, _a, target, class_cond = batch
        target, class_cond = target.to(device), class_cond.to(device)

    noise = torch.randn_like(target)  # randn noise
    if mask is None:  # no inpainting mask, start from noise
        source = noise
    elif mask_pixels is not None and mask_encoder is not None:  # for conditional inpainting
        # we can do curriculum learning and/or on-the-fly data augmentation by overwriting the mask & source info...
        p_ones, p_zeros = 0.3, 0.02
        if epoch <= curriculum_epochs:    # curriculum learning: start with unconditional generation
            #return noise, target, class_cond, None, None  # simple curriculum: hard switch

            p_ones, p_zeros = (curriculum_epochs - (epoch-1))/curriculum_epochs, 0.0    # ramp down the probility of all-ones as epochs increase; no zeros for CL
        elif epoch <= extend_epochs:  # extended transition
            # smooth transition from curriculum to final values
            progress = (epoch - curriculum_epochs) / (extend_epochs - curriculum_epochs)
            p_ones = 0.1 + (0.3 - 0.1) * progress  # 0.1 -> 0.3
            p_zeros = 0.02 * progress  # 0 -> 0.02

        oi, zi, ni = otf_gen_aug_indices(mask, p_ones, p_zeros)
        if len(oi) > 0: # ones indices  - uncond generation
            mask_pixels[oi], source[oi] = 1, noise[oi]  # for the 1's, we use pytorch's broadcasting of scalars. (note: mask[oi] already=1) 
        if len(zi) > 0: # zeros indices  - no inpainting/gen
            mask_pixels[zi], source[zi] = 0, target[zi]
        mask = mask_encoder(mask_pixels.to(device))
        source = source + mask*(noise - source)  # blending equation 
    else: 
        assert False, "Unintended edge case"

    return source, target, class_cond, mask, mask_pixels   # note we're never using pure/original source from dataset, always mixed



def train_flow(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("train_flow: device =", device)

    # Extract parameters from config
    data_path = config.data
    if not 'encoded' in data_path: 
        data_path = f"{data_path}_encoded_{config.codec.choice}"
    print("train_flow: data_path =",data_path)
    batch_size = ldcfg(config,'batch_size')
    n_classes = ldcfg(config.flow.unet,'n_classes', 0,verbose=True)
    if n_classes == 0: 
        class_condition = False
    else: 
        class_condition = ldcfg(config,'condition',False)
    learning_rate = ldcfg(config,'learning_rate')
    epochs = ldcfg(config,'epochs')
    project_name = ldcfg(config,'project_name')
    run_name = ldcfg(config, 'run_name')
    no_wandb = ldcfg(config,'no_wandb', False)
    lambda_lowres = config.flow.get('lambda_lowres', 0.1) # might not have it
    is_midi = any(x in data_path.lower() for x in ['pop909', 'midi'])
    keep_gray = ldcfg(config.codec,'in_channels',3)==1
    print("is_midi, keep_gray =",is_midi, keep_gray)

    print(f"data_path = {data_path}")
    train_path, val_path = f"{data_path}/train",  f"{data_path}/val"

    print(f"Loading train data from: {train_path}")
    pre_encoded=True   # TODO: don't hard-code this, move to config
    mask_encoder = None
    if pre_encoded:  # main use case
        train_dataset = PreEncodedDataset(train_path, n_classes=n_classes)
        train_dataloader = DataLoader( dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, 
                pin_memory=True, persistent_workers=True,)
        dataset, dataloader = train_dataset, train_dataloader # aliases to avoid errors later
    
        print(f"Loading val data from: {val_path}")
        val_dataset = PreEncodedDataset(val_path, n_classes=n_classes)
        val_dataloader = DataLoader( dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=12, 
                pin_memory=True, persistent_workers=True,)

        # Inspect the data we'll be loading
        print("\nInspecting sample data...")
        batch = next(iter(train_dataloader))  # will be a list of something
        batch_type = type(batch)
        print("batch_type = ",batch_type) 
        print("len(batch) = ",len(batch)) 
        batch_data_type = type(batch[0])
        print("batch_data_type = ",batch_data_type) 
        if batch_data_type == dict: 
            print("batch[0].keys() = ",batch[0].keys())
            if 'mask_pixels' in batch[0].keys(): 
                mask_encoder = MaskEncoder().to(device)
                mask_pixels = batch[0]['mask_pixels']
                print("we're inpainting")
            sample_item = batch[0]['target_latents'][0]
            print("sample_item.shape = ",sample_item.shape)
        elif batch_data_type == tuple: 
            print("we've got a (standard) tuple")
            sample_item, _ = batch[0]  
            print("sample_item.shape = ",sample_item.shape)
    else:   # not recommended unless codec is a no-op
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
    print(f"Detected latent dimensions: C={C}, H={H}, W={W}\n")


    output_dir = f"output_{data_path.split('/')[-1]}-{H}x{W}" 
    os.makedirs(output_dir, exist_ok=True)

    codec = setup_codec(config, device).eval() # Load codec for inference/evaluation
    # variables for tracking codebook usage
    cb_levels = ldcfg(config, 'codebook_levels', 4)
    cb_size = ldcfg(config, 'vq_num_embeddings', 32)
    cb_tracker = CodebookUsageTracker(num_levels=cb_levels, codebook_size=cb_size)

    inpainting = mask_encoder is not None
    print(f"Configuring model for {n_classes} classes, class_condition = {class_condition}, mask_cond={mask_encoder is not None}\n")
    # Create flow model 
    dim_mults = ldcfg(config,'dim_mults', [1,2,4,4], verbose=True)
    if pre_encoded: # usually we want this
        model = Unet( dim=H, channels=C, dim_mults=dim_mults, n_classes=n_classes, mask_cond=inpainting).to(device)
    else:  # added this to test HDiT
        levels = [ LevelSpec(depth=2, width=256, d_ff=768, self_attn=GlobalAttentionSpec(d_head=64), dropout=0.0),
                   LevelSpec(depth=4, width=512, d_ff=1536, self_attn=GlobalAttentionSpec(d_head=64), dropout=0.0), ]
        mapping = MappingSpec(depth=2, width=256, d_ff=768, dropout=0.0)
        print("model is ImageTransformerDenoiserModelV2")
        model = ImageTransformerDenoiserModelV2( levels=levels,  mapping=mapping, in_channels=C, out_channels=C,
            patch_size=(4, 4),  # or (2, 2) for smaller images
            num_classes=n_classes if n_classes else 0, mapping_cond_dim=0)  # unless you have extra conditioning

    model = model.to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total/1e9:.1f}B" if total >= 1e9 else f"Model params: {total/1e6:.1f}M")
    print(f"Model has class_cond_mlp: {hasattr(model, 'class_cond_mlp')}")
    print(f"Model has mask_fusion_conv: {hasattr(model, 'mask_fusion_conv')}")
    print("")


    loss_fn = torch.nn.MSELoss()
    if mask_encoder is not None:
        mask_encoder_lr = learning_rate*0.1  # slower mask enc learning for training stability
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': learning_rate},
            {'params': mask_encoder.parameters(), 'lr': mask_encoder_lr}
        ])
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=50, T_mult=2, decay=0.6) # previously T_mult=1, decay=1
    scheduler = CosineAnnealingWarmRestartsDecay(optimizer, T_0=30, T_mult=2, decay=0.3) 

    use_wandb = not no_wandb
    if use_wandb: 
        wandb.init(project=project_name, name=run_name, config=dict(config))

    model.mask_encoder = mask_encoder # lazy: attach mask_encoder as an attribute to model, so EMA & checkpointing codes can stay the same ;-) 
    ema = EMA(model, decay=0.999, device=device)

    rolling_avg_loss, alpha = None, 0.99 
    eps = 0.001 # used in integration
    for epoch in range(1, epochs+1):
        model.train()
    
        #short_loader = itertools.islice(train_dataloader, 10)  # Only take 10 batches  ; #temp for testing. import itertools
        #pbar = tqdm(short_loader, desc=f'Epoch {epoch}/{epochs}:', mininterval=0.25)
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{epochs}:', mininterval=0.25)
        mask_pixels, mask_cond, source = None, None, None  # mask_cond = mask_latents
        for batch in pbar:

            source, target, class_cond, mask_cond, mask_pixels = batch_to_data(batch, device, pre_encoded, mask_encoder, epoch=epoch)
            class_cond=None # TODO: this is hard-coded for MIDI data right now. fix this and make it adaptive, re. using config.conditioned
            if random.random() < 0.1: # for classifier-free guidance: turn off class_cond signal sometimes
                class_cond = None 

            optimizer.zero_grad()

            t = torch.rand(target.shape[0], device=device) * (1 - eps) + eps # see above for small eps, eg 0.001
            t = warp_time(t)          

            t_expand = t.view(-1, 1, 1, 1).repeat(1, target.shape[1], target.shape[2], target.shape[3])
            x = (1 - t_expand) * source +  t_expand * target  # linterp between source & target = constant velocity
            v_guess = target - source    # constant velocity
            t_scale = 999 if pre_encoded else 1
            #v_model = model(x, t * t_scale, aug_cond=None, class_cond=cond) for use with HDiT 
            v_model = model(x, t * t_scale, cond={'class_cond': class_cond, 'mask_cond': mask_cond})
            loss = loss_fn(v_model, v_guess)
        
            if hasattr(model,'shrinker'):  # optional: if multiresolution UNet is available
                lowres_v_guess = model.shrinker(v_guess)
                lowres_loss = loss_fn(model.bottleneck_target_hook, lowres_v_guess) # compare with hook
                loss = loss + lambda_lowres * lowres_loss

            loss.backward()

            # for training stability : adaptive LR and gradient clipping
            #  actually comment out adaptive LR as it conflicts with the LR scheduler in weird ways that mess up Adam
            #if rolling_avg_loss is None:  
            #    rolling_avg_loss = loss.item()
            #else:
            #    rolling_avg_loss = alpha * rolling_avg_loss + (1 - alpha) * loss.item()
            #    if loss.item() > 3 * rolling_avg_loss:  # Loss spike detected
            #        optimizer.param_groups[0]['lr'] *= 0.5  # Halve the learning rate
            
            if use_wandb:
                wandb.log({
                    "Loss/train": loss.item(),
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch 
                })
            
            pbar.set_postfix({"Loss/train":f"{loss.item():.4g}"}, refresh=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping for stability
            if mask_encoder is not None:
                torch.nn.utils.clip_grad_norm_(mask_encoder.parameters(), max_norm=0.5)  # bit lower for stability

            optimizer.step()
            ema.update()

        #---- evals / metrics / viz
        if ((epoch < 20 ) and (epoch % 1==0)) or (epoch >= 20 and (epoch % 10 == 0)):
            model.eval()
            if mask_encoder is not None: mask_encoder.eval()

            batch = next(iter(val_dataloader))  # Fixed missing parenthesis
            source, target, class_cond, mask_cond, mask_pixels = batch_to_data(batch, device, pre_encoded, mask_encoder, epoch=epoch)

            print("Generating sample outputs...")
            eval_batch_size = target.shape[0]  # full batch now that we're chunking 768 # to avoid OOM
            eval_kwargs = {
                'method':'rk4', 'use_wandb': use_wandb, 'output_dir': output_dir,
                'n_classes': n_classes, 'latent_shape': latent_shape, 'batch_size': eval_batch_size,
                'cond':{'class_cond': class_cond, 'mask_cond':mask_cond}, 
                'is_midi':is_midi, 'keep_gray':keep_gray, 'pre_encoded':pre_encoded, 
                'source':source, 'mask_pixels':mask_pixels,
            }
            metrics = evaluate_model( model, codec, epoch, target, tag="", cb_tracker=cb_tracker, **eval_kwargs)

            if epoch>50: # no point doing ema earlier
                model.mask_encoder = mask_encoder # lazy: attach mask_encoder as an attribute to model so I don't have to write more code
                ema.eval()
                evaluate_model(model, codec, epoch, target, tag="ema_", cb_tracker=cb_tracker, **eval_kwargs)
                ema.train()
            model.train()
            if mask_encoder is not None: mask_encoder.train()

            if epoch % 5 == 0: cb_tracker.reset_all()  # accumulate codebook usage info over 5 epochs

        # Checkpoints
        if epoch % 25 == 0: # checkpoint every 25 epochs
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
    print("\nScript invoked via:\n", " ".join(sys.argv),"\n")
    print("cwd is",os.getcwd())
    main()
