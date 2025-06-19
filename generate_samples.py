#!/usr/bin/env python3

import os, gc, torch, time
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from PIL import Image

from flocoder.unet import Unet
from flocoder.codecs import setup_codec
from flocoder.sampling import sampler, decode_latents  # Use existing functions
from flocoder.general import handle_config_path, ldcfg
from flocoder.viz import save_img_grid, imshow
from flocoder.metrics import g2rgb
from flocoder.pianoroll import img_file_2_midi_file, square_to_rect


import subprocess

def midi_to_audio(midi_file, audio_file=None):
    if audio_file is None:
        audio_file = midi_file.replace('.mid', '.wav')
    try:
        subprocess.run(['timidity', midi_file, '-Ow', '-o', audio_file],
                      check=True, capture_output=True)
        return audio_file
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def debug_actual_latent_shape(codec, config):
    """Debug function to determine actual latent shape from codec"""
    device = next(codec.parameters()).device
    # Create a dummy image to encode - use 1 channel if codec expects grayscale
    in_channels = ldcfg(config.codec, 'in_channels', 3)
    dummy_img = torch.randn(1, in_channels, 128, 128).to(device)

    print(f"Codec choice: {ldcfg(config.codec, 'choice')}")
    print(f"Input image shape: {dummy_img.shape}")

    try:
        if hasattr(codec, 'encode'):
            latent = codec.encode(dummy_img)
            print(f"Encoded latent shape: {latent.shape}")
            return latent.shape[1:]  # Remove batch dimension
        else:
            print("Codec has no encode method")
            return None
    except Exception as e:
        print(f"Error encoding dummy image: {e}")
        return None


# module-level globals
_codec = None
_vmodel = None
_vmodel_path = None
_config = None

@torch.no_grad()
def load_models_once(vmodel_path, config, device=None, use_half=False):
    """Load models only if not already loaded or path changed"""
    global _codec, _vmodel, _config, _vmodel_path
    
    if _codec is None or _vmodel is None or _config != config or vmodel_path != _vmodel_path: 
        print("Loading models...")
        _config, _vmodel_path = config, vmodel_path
        if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading codec...")
        _codec = setup_codec(config, device).eval()
        print("checking codec") 
        debug_actual_latent_shape(_codec, config)
        
        
        if not vmodel_path.endswith(".pt"): vmodel_path = f"checkpoints/{vmodel_path}.pt"
        print("Loading velocity model from",vmodel_path)
        checkpoint = torch.load(vmodel_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        print(f"Velocity model init_conv.weight shape: {state_dict['init_conv.weight'].shape}")
        print(f"This should be [output_dim, 4, 1, 1] for 4-channel input")
        # Check if config was saved with checkpoint
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            print(f"Saved config codec in_channels: {ldcfg(saved_config.codec, 'in_channels', 'not found')}")
        else:
            print("No config saved in checkpoint")

        
        # Infer model parameters from checkpoint
        init_conv = state_dict['init_conv.weight']
        C = init_conv.shape[1]  # Input channels from checkpoint
        #H = W = 16  # Default downsampled size TODO: don't let this be hard-coded
        H = W = init_conv.shape[0]  # we're assuming square images
        print("C, H, W =", C, H, W)
        n_classes = ldcfg(config.flow.unet, 'n_classes', 0) if hasattr(config, 'flow') and hasattr(config.flow, 'unet') else 0
        condition = n_classes > 0
        dim_mults = ldcfg(config.flow, 'dim_mults', [1,2,4,4]) if hasattr(config, 'flow') else [1,2,4,4]
        
        vmodel = Unet(dim=H, channels=C, dim_mults=dim_mults, 
                   condition=condition, n_classes=n_classes).to(device)
        
        try:
            vmodel.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Loaded checkpoint with non-strict parameter matching")
        except Exception as e:
            print(f"Error loading vmodel: {e}")
            raise RuntimeError("Failed to load vmodel checkpoint")
        
        _vmodel = vmodel.eval()
    else:
        print("load_models_once: we've already loaded. Skipping.")

    if use_half: 
        for param in _codec.parameters(): param.data = param.data.half()
        for buffer in _codec.buffers(): buffer.data = buffer.data.half()
        return _codec.half(), _vmodel
    return _codec, _vmodel

def load_models(vmodel_path, config, device):
    """Load codec and vmodel models - wrapper around load_models_once"""
    return load_models_once(vmodel_path, config, device)

def infer_latent_shape(config, debug=True):
    """Determine latent shape from config"""
    codec_choice = ldcfg(config.codec, 'choice')
    image_size = ldcfg(config, 'image_size', 128)
    
    if codec_choice == "sd": 
        return (4, image_size//8, image_size//8)
    elif codec_choice == "noop": 
        return (3, image_size, image_size)
    elif codec_choice == "resize": 
        return (3, config.get('image_size', 32), config.get('image_size', 32))
    elif codec_choice == 'vqgan':
        ds_factor = 2 ** ldcfg(config.codec, 'num_downsamples', 3)
        return (ldcfg(config.codec, 'vq_embedding_dim', 4), image_size//ds_factor, image_size//ds_factor)
    else: raise ValueError(f"Invalid codec_choice = {codec_choice}")

@torch.no_grad()
def generate_batch(vmodel, codec, latent_shape, method, n_steps, cfg_strength, device, curr_batch_size, is_midi=False, keep_gray=False):
    """Generate one batch of samples using existing sampler function"""
    codec, vmodel = codec.to(device), vmodel.to(device)
    
    start_time = time.time()
    print(f"\n====> HERE WE GO: Calling sampler with method = {method}, shape = {(curr_batch_size,) + latent_shape}")
    
    # Use the existing sampler function from sampling.py
    print("generate_batch: calling sampler, device=",device)
    pred_latents, decoded_pred, nfe = sampler(
        model=vmodel, codec=codec, method=method, batch_size=curr_batch_size,
        n_steps=n_steps, cond=None, n_classes=0, latent_shape=latent_shape, 
        cfg_strength=cfg_strength, is_midi=is_midi, keep_gray=keep_gray
    )
    
    end_time = time.time()
    print(f"<======== Total time for flow+decode: {end_time - start_time:.2f} seconds")
    
    return decoded_pred, pred_latents, nfe

def save_sample_batch(images, latents, output_dir, tag, method, nfe, save_latents=False):
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
def generate_samples(vmodel_path, config, output_dir=None, n_samples=10, cfg_strength=3.0, 
                     device=None, method="rk4", n_steps=10, save_latents=False):
    """Generate samples from trained vmodel. Returns paths to generated images."""
    
    device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("generate_samples: running with device =",device)
    
    if output_dir is None: output_dir = f"output_{Path(vmodel_path).stem}"
    print(f"output_dir = {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    try:
        codec, vmodel = load_models_once(vmodel_path, config, device)
        codec, vmodel = codec.to(device), vmodel.to(device)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    
    latent_shape = infer_latent_shape(config)
    is_midi = any(x in str(config.data).lower() for x in ['pop909', 'midi'])
    keep_gray = ldcfg(config.codec, 'in_channels', 3) == 1
    
    print(f"Using codec: {ldcfg(config.codec, 'choice')}")
    print(f"Using latent shape: {latent_shape}")
    print(f"is_midi: {is_midi}, keep_gray: {keep_gray}")
    
    batch_size = 10 if n_samples < 10 else 100
    all_filenames = []
    samples_left = max(n_samples, 10)
    
    while samples_left > 0:
        curr_batch_size = min(batch_size, samples_left)
        print("batch_size, samples_left, curr_batch_size =", batch_size, samples_left, curr_batch_size)
        
        # Generate batch
        images, latents, nfe = generate_batch(vmodel, codec, latent_shape, method, n_steps, 
                                            cfg_strength, device, curr_batch_size, is_midi, keep_gray)
        
        print("Saving samples...")
        tag = f"batch_{len(all_filenames)}_"
        files = save_sample_batch(images[:curr_batch_size], latents[:curr_batch_size], 
                                output_dir, tag, method, nfe, save_latents)
        all_filenames.extend(files)
        
        samples_left -= curr_batch_size
        
        # Clear memory
        images, latents = None, None
        gc.collect()
        torch.cuda.empty_cache()
    
    individual_images = [f for f in all_filenames if f.endswith('.png') and '_nfe_' in f and not f.endswith('_nfe_.png')]
    return individual_images[:n_samples]

def create_gradio_interface(config, codec, vmodel):
    """Create gradio interface for interactive generation"""
    import gradio as gr
    from midi_player import MIDIPlayer
    from midi_player.stylers import dark
    
    with gr.Blocks(title="Flow Sampler") as app:
        title = gr.Markdown("# flocoder: Latent Flow Sampler")
        
        with gr.Row():
            with gr.Column():
                vmodel_input = gr.Radio(["latent", "HDiT"], value="latent", label="Model")
                checkpoint_input = gr.Textbox(value="flow_best.pt", label="Checkpoint Path")
                samples_input = gr.Slider(1, 20, value=4, step=1, label="Samples")
                cfg_input = gr.Slider(-3, 15.0, value=3.0, step=0.1, label="CFG Strength")
                method_input = gr.Radio(["rk4", "rk45"], value="rk4", label="Method")
                steps_input = gr.Slider(1, 100, value=20, step=1, label="Steps")
                if torch.cuda.is_available(): 
                    device_input = gr.Radio(["cuda","cpu"], value="cuda", label="Device")
                else:
                    device_input = gr.Radio(["cpu"], value="cpu", label="Device")
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                output_gallery = gr.Gallery()
                num_samples = 4
                audio_outputs = [gr.Audio(label=f"Sample {i+1} Audio", visible=True) for i in range(num_samples)]
                midi_players = gr.HTML(label=f"MIDI Players")
        
        def show_samples(vmodel_input_ui, checkpoint_path_ui, n_samples_ui, cfg_strength_ui, method_ui, steps_ui, device_ui):
            # SHH: re. coding style: I know it's an embedded function but this is how most gradio apps are done. 
            print(f"show_samples: Generating samples with: n_samples={n_samples_ui}, cfg_strength={cfg_strength_ui}, method={method_ui}")
            
            latent_shape = infer_latent_shape(config)
            device = torch.device(device_ui)
            is_midi = any(x in str(config.data).lower() for x in ['pop909', 'midi'])
            keep_gray = ldcfg(config.codec, 'in_channels', 3) == 1
            
            all_filenames = []
            samples_left = int(n_samples_ui)
            batch_size = 10 if samples_left < 10 else 100
            
            while samples_left > 0:
                curr_batch_size = min(batch_size, samples_left)
                images, latents, nfe = generate_batch(vmodel, codec, latent_shape, method_ui, steps_ui, 
                                                    cfg_strength_ui, device, curr_batch_size, is_midi, keep_gray)
                
                tag = f"gradio_batch_{len(all_filenames)}_"
                files = save_sample_batch(images[:curr_batch_size], latents[:curr_batch_size], 
                                        "output", tag, method_ui, nfe, False)
                all_filenames.extend(files)
                samples_left -= curr_batch_size
                
                del images, latents
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()  # trying to save VRAM
            
            individual_images = [f for f in all_filenames if f.endswith('.png') and '_nfe_' in f and not f.endswith('_nfe_.png')]
            result_images = individual_images[:int(n_samples_ui)]
            print(f"Generated {len(result_images)} images for gradio display")
            
            # Create MIDI players and audio files
            all_midi_html = ''
            audio_files = []
            if is_midi:
                for i, img_file in enumerate(result_images):
                    try:
                        img = Image.open(img_file)
                        if img.size[0] > 128 and img.size[0] == img.size[1]:  # square image, convert to rect
                            print(f"--- img_size = f{img_size}.  Converting square image to rect")
                            img_rect = square_to_rect(img)
                            rect_file = img_file.replace('.png', '_rect.png')
                            img_rect.save(rect_file)
                            midi_file = img_file_2_midi_file(rect_file, require_onsets=False)
                        else:
                            midi_file = img_file_2_midi_file(img_file, require_onsets=False)

                        # Convert MIDI to audio
                        audio_file = midi_to_audio(midi_file)
                        if audio_file and os.path.exists(audio_file):
                            audio_files.append(audio_file)
                        else:
                            audio_files.append(None)
                        
                        # create MIDI HTML players
                        srcdoc = MIDIPlayer(midi_file, 700, styler=dark).html
                        srcdoc = srcdoc.replace("\"", "'")
                        html = f'''<h3>Sample {i+1}</h3><iframe srcdoc="{srcdoc}" height="300" width="100%" title="MIDI Player {i+1}"></iframe><br>'''
                        all_midi_html += html
                    except Exception as e:
                        print(f"Error creating MIDI for {img_file}: {e}")
                        all_midi_html += f"<p>Error creating MIDI player for sample {i+1}</p><br>"
       
            return [result_images, all_midi_html]+ audio_files
        

        generate_btn.click(show_samples, 
                          [vmodel_input, checkpoint_input, samples_input, cfg_input, method_input, steps_input, device_input], 
                          [output_gallery, midi_players]+ audio_outputs)
    
    return app

handle_config_path()
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers_sd")
def main(config: DictConfig) -> None:
    """Entry point for sample generation script"""
    print(f"Config: {OmegaConf.to_yaml(config)}")
    
    vmodel_path = config.get("vmodel_path", "vmodel_best")
    n_samples = config.get("n_samples", 100)
    method = config.get("method", "rk4")
    output_dir = config.get("output_dir", None)
    save_latents = config.get("save_latents", False)
    use_gradio = config.get("use_gradio", False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("main: device = ",device)
    
    if use_gradio:
        codec, vmodel = load_models_once(vmodel_path, config, device=device)
        app = create_gradio_interface(config, codec, vmodel)
        print("\n\nLaunching gradio app...")
        app.launch(share=True)
    else:
        filenames = generate_samples(vmodel_path, config, output_dir=output_dir, 
                                   n_samples=n_samples, device=device, method=method, 
                                   save_latents=save_latents)
        print(f"Generated {len(filenames)} files")
        print(f"Grid images: {[f for f in filenames if '_nfe_' in f]}")

if __name__ == "__main__":
    main()

