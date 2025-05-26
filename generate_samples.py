#!/usr/bin/env python3

import os, gc, torch, time
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from PIL import Image


# NATTEN initialization - keep at top since it prints diagnostics
import natten
from natten import has_half
print(f"Natten version {natten.__version__}, checking half precision:")
print(has_half())  # Default torch cuda device
print(has_half(0)) # cuda:0
from natten import use_fused_na
use_fused_na()

from flocoder.unet import Unet
from flocoder.codecs import load_codec
from flocoder.sampling import generate_latents, decode_latents
from flocoder.general import handle_config_path, ldcfg
from flocoder.viz import save_img_grid, imshow
from flocoder.metrics import g2rgb
from flocoder.pianoroll import img_file_2_midi_file, square_to_rect



# module-level globals
_codec = None
_unet = None
_config = None


@torch.no_grad()
def load_models_once(unet_path,    # path to saved unet checkpoint
                     config,       # config object with model parameters
                     device=None,  # torch device to load models on
                     use_half=False): # whether to use half precision
    """Load models only if not already loaded or path changed"""
    global _codec, _unet, _config
    
    if _codec is None or _unet is None or _config != config:
        print("Loading models...")
        _config = config
        if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load codec based on config choice - similar to how it's done in train_flow.py
        _codec = load_codec(config, device).eval()
        
        # Handle checkpoint path
        if not unet_path.endswith(".pt"): 
            unet_path = f"checkpoints/{unet_path}.pt"
        
        checkpoint = torch.load(unet_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # Infer model parameters from checkpoint
        init_conv = state_dict['init_conv.weight']
        C = init_conv.shape[1]  # Input channels from checkpoint
        H = W = 16  # Default downsampled size
        n_classes = ldcfg(config, 'n_classes', 0, supply_defaults=True)
        condition = ldcfg(config, 'condition', False)
        dim_mults = ldcfg(config, 'dim_mults', [1,2,4,4])
        
        unet = Unet(dim=H, channels=C, dim_mults=dim_mults, 
                   condition=condition, n_classes=n_classes).to(device)
        
        try:
            # First try loading with strict=False to allow parameter mismatches
            unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded checkpoint with non-strict parameter matching")
        except Exception as e:
            print(f"Error loading unet: {e}")
            raise RuntimeError("Failed to load unet checkpoint")
        
        _unet = unet.eval()
    else:
        print("load_models_once: we've already loaded. Skipping.")

    if not use_half: return _codec, _unet

    for param in _codec.parameters():
        param.data = param.data.half()
    for buffer in _codec.buffers():
        buffer.data = buffer.data.half()
    return _codec.half(), _unet


def load_models(unet_path,    # path to saved unet checkpoint
                config,       # config object with model parameters
                device):      # torch device to load models on
    """Load codec and unet models - wrapper around load_models_once"""
    return load_models_once(unet_path, config, device)


def infer_latent_shape(config): # config object with codec parameters
    """Determine latent shape from config"""
    codec_choice = ldcfg(config.codec, 'choice')
    image_size = ldcfg(config, 'image_size', 128)
    
    if codec_choice == "sd":
        # SD VAE uses 4x downsampling with 4 channels
        return (4, image_size//8, image_size//8)
    elif codec_choice == "resize":
        # Resize codec preserves channel count and just scales dimensions
        return (3, config.get('image_size', 32), config.get('image_size', 32))
    elif codec_choice == 'vqgan':
        # VQGAN typically does 2^num_downsamples reduction with internal_dim channels
        ds_factor = 2 ** ldcfg(config, 'num_downsamples', 3)
        return (ldcfg(config, 'vq_embedding_dim', 4), image_size//ds_factor, image_size//ds_factor)
    else:
        raise ValueError(f"Invalid codec_choice = {codec_choice}")


@torch.no_grad()
def generate_batch(unet,           # the flow model
                   codec,          # the codec for decoding
                   latent_shape,   # shape of latent space
                   method,         # integration method
                   n_steps,        # number of integration steps
                   cfg_strength,   # classifier-free guidance strength
                   device,         # torch device
                   curr_batch_size): # current batch size to use
    """Generate one batch of samples"""
    shape = (curr_batch_size,) + latent_shape
    
    start_time = time.time()
    print(f"\n====> HERE WE GO: Calling sampler with method = {method}, shape = {shape}")
    latents, nfe = generate_latents(unet, shape, method, n_steps, cond=None, cfg_strength=cfg_strength)
    mid_time = time.time()
    print(f"\n<======  BACK FROM FLOW SAMPLER in {mid_time-start_time:.2f} seconds")
    
    print("CALLING DECODER... ====>")
    dtype = next(codec.parameters()).dtype
    images = decode_latents(codec, latents.to(dtype).to(device))  # Decode to real images
    end_time = time.time()
    print(f"<======== BACK FROM DECODER in {end_time-mid_time:.2f} seconds.")
    print(f"Total time for flow+decode: {end_time - start_time:.2f} seconds")
    
    return images, latents, nfe


def save_sample_batch(images,        # decoded images to save
                      latents,       # latent codes to save (optional)
                      output_dir,    # directory to save in
                      tag,           # filename tag/prefix
                      method,        # method name for filename
                      nfe,           # number of function evaluations
                      save_latents=False): # whether to save latent images too
    """Save batch of samples and return filenames"""
    
    # Save grids
    save_img_grid(images.cpu(), 0, nfe, tag=f"{tag}{method}", use_wandb=False, output_dir=output_dir)
    if save_latents:
        save_img_grid(latents.cpu(), 0, nfe, tag=f"{tag}latent_{method}", use_wandb=False, output_dir=output_dir)
    
    # Save individual files
    filenames = []
    for i in range(min(len(images), 100)):
        img_filename = os.path.join(output_dir, f"{tag}{method}_epoch_1_nfe_{nfe}_{i:03d}.png")
        img = images.cpu()[i]
        if img.shape[0] == 1: img = g2rgb(img, keep_gray=True)[0]
        imshow(img, img_filename)
        filenames.append(img_filename)
    
    if save_latents:
        for i in range(min(len(latents), 100)):
            latent_filename = os.path.join(output_dir, f"{tag}latent_{method}_epoch_1_nfe_{nfe}_{i:03d}.png")
            latent = latents.cpu()[i]
            if latent.shape[0] == 1: latent = g2rgb(latent, keep_gray=True)
            imshow(latent, latent_filename)
            filenames.append(latent_filename)
    
    return filenames


@torch.no_grad()
def generate_samples(unet_path,       # path to trained unet checkpoint
                     config,          # config object
                     output_dir=None, # output directory (auto-generated if None)
                     n_samples=10,    # number of samples to generate
                     cfg_strength=3.0, # classifier-free guidance strength
                     device=None,     # torch device (auto-detected if None)
                     method="rk4",    # integration method
                     n_steps=10,      # number of integration steps
                     save_latents=False): # save latent images too
    """Generate samples from trained unet. Returns paths to generated images."""
    
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Set default output directory
    if output_dir is None: 
        output_dir = f"output_{Path(unet_path).stem}"
    print(f"output_dir = {output_dir}")
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    os.makedirs("output", exist_ok=True)  # Ensure output directory exists
    
    try:
        codec, unet = load_models_once(unet_path, config, device)
        codec, unet = codec.to(device), unet.to(device)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    
    latent_shape = infer_latent_shape(config)
    
    print(f"Using codec: {ldcfg(config.codec, 'choice')}")
    print(f"Using latent shape: {latent_shape}")
    
    batch_size = 10 if n_samples < 10 else 100  # sampler expects multiples of 10
    all_filenames = []
    samples_left = max(n_samples, 10)
    
    while samples_left > 0:
        curr_batch_size = min(batch_size, samples_left)
        print("batch_size, samples_left, curr_batch_size =", batch_size, samples_left, curr_batch_size)
        
        # Generate batch
        images, latents, nfe = generate_batch(unet, codec, latent_shape, method, n_steps, cfg_strength, device, curr_batch_size)
        
        print("Pushing to the web interface.")
        # Save batch  
        tag = f"batch_{len(all_filenames)}_"  # Tag for filenames
        files = save_sample_batch(images[:curr_batch_size], latents[:curr_batch_size], 
                                output_dir, tag, method, nfe, save_latents)
        all_filenames.extend(files)
        
        samples_left -= curr_batch_size
        
        # Clear memory
        images, latents = None, None
        gc.collect()
        torch.cuda.empty_cache()
    
    # Only return individual images, not the grid image
    individual_images = [f for f in all_filenames if f.endswith('.png') and '_nfe_' in f and not f.endswith('_nfe_.png')]
    return individual_images[:n_samples]


def create_gradio_interface(config,  # config object for interface
                           codec,   # pre-loaded codec model
                           unet):   # pre-loaded unet model
    """Create gradio interface for interactive generation"""
    import gradio as gr
    from midi_player import MIDIPlayer
    from midi_player.stylers import dark
    
    # Create simple Gradio interface
    with gr.Blocks(title="Flow Sampler") as app:
        title = gr.Markdown("# flocoder: Latent Flow Sampler")
        
        with gr.Row():
            with gr.Column():
                unet_input = gr.Textbox(value="flow_99", label="UNet Path")
                samples_input = gr.Slider(1, 20, value=4, step=1, label="Samples")
                cfg_input = gr.Slider(-3, 15.0, value=3.0, step=0.1, label="CFG Strength")
                method_input = gr.Radio(["rk4", "rk45"], value="rk4", label="Method")
                steps_input = gr.Slider(1, 100, value=10, step=1, label="Steps")
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                output_gallery = gr.Gallery()
                midi_players = [gr.HTML(label=f"MIDI Player {i+1}") for i in range(4)]
        
        def show_samples(unet_path_ui, n_samples_ui, cfg_strength_ui, method_ui, steps_ui):
            # nested function called by gradio button clicker, below
            # Models are already loaded, so we can generate directly without reloading
            print(f"show_samples: Generating samples with: n_samples={n_samples_ui}, cfg_strength={cfg_strength_ui}, method={method_ui}")
            
            # Use the pre-loaded models for generation
            latent_shape = infer_latent_shape(config)
            device = next(unet.parameters()).device
            
            # Generate samples directly
            all_filenames = []
            samples_left = int(n_samples_ui)
            batch_size = 10 if samples_left < 10 else 100
            
            while samples_left > 0:
                curr_batch_size = min(batch_size, samples_left)
                images, latents, nfe = generate_batch(unet, codec, latent_shape, method_ui, steps_ui, cfg_strength_ui, device, curr_batch_size)
                
                tag = f"gradio_batch_{len(all_filenames)}_"
                files = save_sample_batch(images[:curr_batch_size], latents[:curr_batch_size], 
                                        "output", tag, method_ui, nfe, False)
                all_filenames.extend(files)
                samples_left -= curr_batch_size
                
                # Clear memory
                del images, latents
                gc.collect()
                torch.cuda.empty_cache()
            
            # Return only up to n_samples images for the gallery
            individual_images = [f for f in all_filenames if f.endswith('.png') and '_nfe_' in f and not f.endswith('_nfe_.png')]
            result_images = individual_images[:int(n_samples_ui)]
            print(f"Generated {len(result_images)} images for gradio display")
            # Format images for gallery (needs tuples of (image_path, caption))
            gallery_items = [(img_path, f"Sample {i+1}") for i, img_path in enumerate(result_images)]


            # Convert images to MIDI and create player HTML
            midi_htmls = []
            for i, img_file in enumerate(result_images):
                if i >= 4:  # Only create MIDI for first 4 samples
                    break
                try:
                    # Convert image to rectangular format if needed (for MIDI conversion)
                    img = Image.open(img_file)
                    if False and img.size[0] == img.size[1]:  # square image, convert to rect
                        img_rect = square_to_rect(img)
                        rect_file = img_file.replace('.png', '_rect.png')
                        img_rect.save(rect_file)
                        midi_file = img_file_2_midi_file(rect_file, require_onsets=False)
                    else:
                        midi_file = img_file_2_midi_file(img_file, require_onsets=False)
                    
                    # Create MIDI player HTML
                    srcdoc = MIDIPlayer(midi_file, 300, styler=dark).html
                    srcdoc = srcdoc.replace("\"", "'")
                    html = f'''<iframe srcdoc="{srcdoc}" height="300" width="100%" title="MIDI Player {i+1}"></iframe>'''
                    midi_htmls.append(html)
                except Exception as e:
                    print(f"Error creating MIDI for {img_file}: {e}")
                    midi_htmls.append(f"<p>Error creating MIDI player for sample {i+1}</p>")
            
            # Pad with empty HTML if we have fewer than 4 samples
            while len(midi_htmls) < 4:
                midi_htmls.append("")
            
            print(f"Generated {len(result_images)} files for gradio display")
            return result_images, *midi_htmls
        
        generate_btn.click(show_samples, 
                          [unet_input, samples_input, cfg_input, method_input, steps_input], 
                          [output_gallery]+midi_players)
    
    return app


handle_config_path()  # allow for full path in --config-name
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers_sd")
def main(config: DictConfig) -> None:
    """Entry point for sample generation script"""
    print(f"Config: {OmegaConf.to_yaml(config)}")
    
    # Extract parameters from the config - these will likely just yield the defaults
    unet_path = config.get("unet_path", "flow_99")
    n_samples = config.get("n_samples", 100)
    method = config.get("method", "rk4")
    output_dir = config.get("output_dir", None)
    save_latents = config.get("save_latents", False)
    use_gradio = config.get("use_gradio", False)
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    if use_gradio:
        codec, unet = load_models_once(unet_path, config, device=device)
        app = create_gradio_interface(config, codec, unet)
        app.launch(share=False)  # Launch the app
    else:
        # Normal operation - generate samples using the provided config
        filenames = generate_samples(unet_path, config, output_dir=output_dir, 
                                   n_samples=n_samples, device=device, method=method, 
                                   save_latents=save_latents)
        print(f"Generated {len(filenames)} files")
        print(f"Grid images: {[f for f in filenames if '_nfe_' in f]}")


if __name__ == "__main__":
    main()

