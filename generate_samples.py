#! /usr/bin/env python3

import os, sys, random, torch, gc
from pathlib import Path
import re

import hydra
from omegaconf import OmegaConf, DictConfig

from flocoder.unet import Unet, MRUnet
from flocoder.codecs import load_codec
from flocoder.sampling import sampler, rk45_sampler
from flocoder.general import handle_config_path
from flocoder.viz import save_img_grid, imshow


@torch.no_grad() # eval only 
def generate_samples(unet_path, config, output_dir=None, n_samples=10, cfg_strength=3.0, device=None, method="rk45", save_latents=False):
    """Generate samples from a trained unet. Returns paths to generated images."""
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if output_dir is None: output_dir = f"output_{Path(unet_path).stem}"  # Set default output directory
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    if not unet_path.endswith(".pt"): unet_path = f"checkpoints/{unet_path}.pt"  # Handle checkpoint path
    checkpoint = torch.load(unet_path, map_location=device)  # Load checkpoint

    # Determine unet architecture from checkpoint
    # Get dimensions that match the checkpoint
    state_dict = checkpoint['model_state_dict']
    init_conv = state_dict['init_conv.weight']
    C = init_conv.shape[1]  # Input channels from checkpoint
    H = W = 16  # Default downsampled size
    
    # Use condition and n_classes from config
    n_classes = config.unet.get('n_classes', 0)
    condition = config.flow.unet.get('condition', False)
    
    print(f"Creating unet with: channels={C}, dim={H}, condition={condition}, n_classes={n_classes}")
    
    # Create unet with parameters matching the checkpoint
    unet_cls = Unet #if config.get('use_mrunet', False) else Unet
    print(f"Setting up unet class: {unet_cls}")

    # for reference:
    # class Unet(nn.Module):
    # def __init__(
    #     self,
    #     dim,
    #     dim_mults=(1, 2, 4, 8),
    #     channels=3,
    #     resnet_block_groups=4,
    #     condition=False,
    #     n_classes=10,
    #     use_checkpoint=False,
    # ):
    #unet = unet_cls(dim=H, channels=C, condition=condition, n_classes=n_classes).to(device)
    unet = Unet(
        dim=H,
        channels=C,
        dim_mults=(1, 2, 4),
        condition=condition,
        n_classes=n_classes,
    ).to(device)
    
    print(f"Loading unet checkpoint from {unet_path}")
    try:
        # First try loading with strict=False to allow parameter mismatches
        unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded checkpoint with non-strict parameter matching")
    except Exception as e:
        print(f"Error loading unet: {e}")
        raise RuntimeError("Failed to load unet checkpoint")
        
    unet.eval()
    
    # Load codec based on config choice - similar to how it's done in train_flow.py
    if hasattr(config, 'codec') and hasattr(config.codec, 'choice'):
        codec_choice = config.codec.choice
        print(f"Using codec: {codec_choice}")
    else:
        codec_choice = "vqgan"  # Default
        print(f"No codec choice found in config. Using default: {codec_choice}")
    
    # Load the codec
    codec = load_codec(config, device)  # This function handles the codec choice internally
    
    # For sampling, we need to know the latent shape
    if codec_choice == "sd":
        # SD VAE uses 4x downsampling with 4 channels
        latent_shape = (4, config.get('image_size', 128) // 8, config.get('image_size', 128) // 8)
    elif codec_choice == "resize":
        # Resize codec preserves channel count and just scales dimensions
        latent_shape = (3, config.get('image_size', 32), config.get('image_size', 32))
    else:  # vqgan
        # VQGAN typically does 2^num_downsamples reduction with internal_dim channels
        ds_factor = 2 ** config.get('num_downsamples', 3)
        latent_shape = (config.unet.get('vq_embedding_dim', 4), 
                       config.get('image_size', 128) // ds_factor, 
                       config.get('image_size', 128) // ds_factor)
    
    print(f"Using latent shape: {latent_shape}")
    
    
    batch_size = 10 if n_samples < 10 else 100  # sampler expects multiples of 10
    all_filenames, samples_left = [], max(n_samples, 10)
    
    while samples_left > 0:
        curr_batch_size = min(batch_size, samples_left)
        curr_shape = (curr_batch_size, *latent_shape)
        
        # Generate samples with selected method
        print(f"calling sampler with curr_shape = {curr_shape}")
        if method == "rk45": 
            images, nfe = rk45_sampler(unet, curr_shape, device, n_classes=n_classes)
        else: 
            # Use existing sampler function for other methods
            images = sampler(unet, codec, 0, method, device, batch_size=curr_batch_size, 
                          n_classes=n_classes, latent_shape=latent_shape, cfg_strength=cfg_strength)
            nfe = 100  # Default NFE value for non-RK45 methods
        
        decoded_images = codec.decode(images.to(device))  # Decode to real images
        tag = f"batch_{len(all_filenames)}_"  # Tag for filenames
        
        # Save latents if requested
        latent_files = []
        if save_latents:
            save_img_grid(images.cpu(), 0, method, nfe, tag=tag+"latent_", use_wandb=False, output_dir=output_dir)
            latent_files = [os.path.join(output_dir, f"{tag}latent_{method}_epoch_1_nfe_{nfe}.png")]
            # Save individual latent images
            for i in range(min(len(images), 100)):
                img_filename = os.path.join(output_dir, f"{tag}latent_{method}_epoch_1_nfe_{nfe}_{i:03d}.png")
                imshow(images.cpu()[i], img_filename)
                latent_files.append(img_filename)
        
        # Save decoded images
        save_img_grid(decoded_images.cpu(), 0, method, nfe, tag=tag, use_wandb=False, output_dir=output_dir)
        decoded_files = [os.path.join(output_dir, f"{tag}{method}_epoch_1_nfe_{nfe}.png")]
        
        # Save individual decoded images
        for i in range(min(len(decoded_images), 100)):
            img_filename = os.path.join(output_dir, f"{tag}{method}_epoch_1_nfe_{nfe}_{i:03d}.png")
            imshow(decoded_images.cpu()[i], img_filename)
            decoded_files.append(img_filename)
        
        all_filenames.extend(decoded_files)
        if save_latents: all_filenames.extend(latent_files)
        
        samples_left -= curr_batch_size
        
        # Clear memory
        images, decoded_images = None, None
        gc.collect(); torch.cuda.empty_cache()
    
    # Only return individual images, not the grid image
    individual_images = [f for f in all_filenames if f.endswith('.png') and '_nfe_' in f and not f.endswith('_nfe_.png')]
    return individual_images[1:n_samples+1]


handle_config_path()  # allow for full path in --config-name
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers_sd")
def main(config: DictConfig) -> None:
    """Entry point for the sample generation script."""
    print(f"Config: {OmegaConf.to_yaml(config)}")
    
    # Extract parameters from the config
    unet_path = config.get("unet_path", "flow_99")
    n_samples = config.get("n_samples", 100)
    method = config.get("method", "rk45")
    output_dir = config.get("output_dir", None)
    device = config.get("device", None)
    save_latents = config.get("save_latents", False) 
    use_gradio = config.get("use_gradio", False)
    
    # Handle gradio interface if requested
    if use_gradio:
        n_samples = 4
        import gradio as gr
        
        # Create simple Gradio interface
        with gr.Blocks(title="Flow unet Sampler") as app:
            title = gr.Markdown(f"# flocoder: Latent Flow Sampler")
            with gr.Row():
                with gr.Column():
                    unet_input = gr.Textbox(value=unet_path, label="unet Path")
                    samples_input = gr.Slider(minimum=1, maximum=n_samples, value=n_samples, step=1, label="Samples")
                    cfg_input = gr.Slider(minimum=-3, maximum=15.0, value=3.0, step=0.1, label="CFG Strength")
                    method_input = gr.Radio(choices=["rk45"], value=method, label="Method")
                    generate_btn = gr.Button("Generate")
                
                output_gallery = gr.Gallery()
            
            def show_samples(unet_path_ui, n_samples_ui, cfg_strength_ui, method_ui):
                # This function receives inputs from the Gradio UI and calls the main generate_samples function
                # 'config' is available in this scope from the main function
                print(f"show_samples: Generating samples with: unet_path={unet_path_ui}, n_samples={n_samples_ui}, cfg_strength={cfg_strength_ui}, method={method_ui}")
                files = generate_samples(
                    unet_path=unet_path_ui,
                    config=config, # Pass the original config object
                    n_samples=int(n_samples_ui),
                    method=method_ui,
                    cfg_strength=cfg_strength_ui, # Pass the CFG strength from the UI
                    output_dir=None # Ensure output_dir is passed, though None triggers default
                )
                # Return only up to n_samples images for the gallery
                print(f"Generated {len(files)} files, retuning only {n_samples_ui} images")
                return files[:int(n_samples_ui)]
            
            generate_btn.click(show_samples, [unet_input, samples_input, cfg_input, method_input], output_gallery)
        
        app.launch(share=False)  # Launch the app
    else:
        # Normal operation - generate samples using the provided config
        filenames = generate_samples(unet_path, config, output_dir=output_dir, n_samples=n_samples,
                                  device=device, method=method, save_latents=save_latents)
        print(f"Generated {len(filenames)} files")
        print(f"Grid images: {[f for f in filenames if '_nfe_' in f]}")

if __name__ == "__main__":
    main()

