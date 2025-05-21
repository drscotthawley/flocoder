#! /usr/bin/env python3

import os, sys, random, torch, gc
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig

from flocoder.unet import Unet, MRUnet
from flocoder.codecs import load_codec
from flocoder.sampling import sampler, rk45_sampler
from flocoder.general import handle_config_path
from flocoder.viz import save_img_grid, imshow

def generate_samples(model_path, config, output_dir=None, n_samples=100, device=None, method="rk45", save_latents=False):
    """Generate samples from a trained model. Returns paths to generated images."""
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = config
    
    if output_dir is None: output_dir = f"output_{Path(model_path).stem}"  # Set default output directory
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    if not model_path.endswith(".pt"): model_path = f"checkpoints/{model_path}.pt"  # Handle checkpoint path
    checkpoint = torch.load(model_path, map_location=device)  # Load checkpoint

    # Determine model architecture from checkpoint
    # Get dimensions that match the checkpoint
    state_dict = checkpoint['model_state_dict']
    init_conv = state_dict['init_conv.weight']
    C = init_conv.shape[1]  # Input channels from checkpoint
    H = W = 16  # Default downsampled size
    
    # Use condition and n_classes from config
    n_classes = cfg.model.get('n_classes', 0)
    condition = cfg.model.get('condition', False)
    
    print(f"Creating model with: channels={C}, dim={H}, condition={condition}, n_classes={n_classes}")
    
    # Create model with parameters matching the checkpoint
    model_cls = MRUnet if cfg.get('use_mrunet', False) else Unet
    model = model_cls(dim=H, channels=C, condition=condition, n_classes=n_classes).to(device)
    
    try:
        # First try loading with strict=False to allow parameter mismatches
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded checkpoint with non-strict parameter matching")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load model checkpoint")
        
    model.eval()
    
    # Load codec based on config choice - similar to how it's done in train_flow.py
    if hasattr(cfg, 'codec') and hasattr(cfg.codec, 'choice'):
        codec_choice = cfg.codec.choice
        print(f"Using codec: {codec_choice}")
    else:
        codec_choice = "vqgan"  # Default
        print(f"No codec choice found in config. Using default: {codec_choice}")
    
    # Load the codec
    codec = load_codec(cfg, device)  # This function handles the codec choice internally
    
    # For sampling, we need to know the latent shape
    if codec_choice == "sd":
        # SD VAE uses 4x downsampling with 4 channels
        latent_shape = (4, cfg.get('image_size', 128) // 8, cfg.get('image_size', 128) // 8)
    elif codec_choice == "resize":
        # Resize codec preserves channel count and just scales dimensions
        latent_shape = (3, cfg.get('image_size', 32), cfg.get('image_size', 32))
    else:  # vqgan
        # VQGAN typically does 2^num_downsamples reduction with internal_dim channels
        ds_factor = 2 ** cfg.get('num_downsamples', 3)
        latent_shape = (cfg.model.get('vq_embedding_dim', 4), 
                       cfg.get('image_size', 128) // ds_factor, 
                       cfg.get('image_size', 128) // ds_factor)
    
    print(f"Using latent shape: {latent_shape}")
    
    batch_size = min(n_samples, 100)  # Use appropriate batch size
    all_filenames, samples_left = [], n_samples
    
    while samples_left > 0:
        curr_batch_size = min(batch_size, samples_left)
        curr_shape = (curr_batch_size, *latent_shape)
        
        # Generate samples with selected method
        if method == "rk45": 
            images, nfe = rk45_sampler(model, curr_shape, device, n_classes=n_classes)
        else: 
            # Use existing sampler function for other methods
            images = sampler(model, codec, 0, method, device, batch_size=curr_batch_size, 
                          n_classes=n_classes, latent_shape=latent_shape)
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
    
    return all_filenames

handle_config_path()  # allow for full path in --config-name
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers_sd")
def main(cfg: DictConfig) -> None:
    """Entry point for the sample generation script."""
    print(f"Config: {OmegaConf.to_yaml(cfg)}")
    
    # Extract parameters from the config
    model_path = cfg.get("model_path", "flow_99")
    n_samples = cfg.get("n_samples", 100)
    method = cfg.get("method", "rk45")
    output_dir = cfg.get("output_dir", None)
    device = cfg.get("device", None)
    save_latents = cfg.get("save_latents", False) 
    use_gradio = cfg.get("use_gradio", False)
    
    # Handle gradio interface if requested
    if use_gradio:
        import gradio as gr
        
        # Create simple Gradio interface
        with gr.Blocks(title="Flow Model Sampler") as app:
            with gr.Row():
                with gr.Column():
                    model_input = gr.Textbox(value=model_path, label="Model Path")
                    samples_input = gr.Slider(minimum=1, maximum=100, value=n_samples, step=1, label="Samples")
                    method_input = gr.Radio(choices=["rk45", "euler"], value=method, label="Method")
                    generate_btn = gr.Button("Generate")
                
                output_gallery = gr.Gallery()
            
            def show_samples(model, n_samples, method):
                files = generate_samples(model, cfg, n_samples=int(n_samples), method=method)
                return [f for f in files if "_nfe_" in f and not any(f.endswith(f"_{i:03d}.png") for i in range(100))]
            
            generate_btn.click(show_samples, [model_input, samples_input, method_input], output_gallery)
        
        app.launch(share=False)  # Launch the app
    else:
        # Normal operation - generate samples using the provided config
        filenames = generate_samples(model_path, cfg, output_dir=output_dir, n_samples=n_samples,
                                  device=device, method=method, save_latents=save_latents)
        print(f"Generated {len(filenames)} files")
        print(f"Grid images: {[f for f in filenames if '_nfe_' in f]}")

if __name__ == "__main__":
    main()

