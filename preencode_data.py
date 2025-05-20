#! /usr/bin/env python3

# As per Esser et al "Scaling Rectified Flow Transformers....":
# "Before scaling up, we filter and preencode our data to ensure safe and efficient pretraining.""

import os
import shutil
from pathlib import Path
import torch
import random
import string
from tqdm import tqdm
import concurrent.futures
import threading
import time
import sys
from torchvision import transforms, datasets
from flocoder.data import ImageListDataset, fast_scandir, MIDIImageDataset, InfiniteDataset, RandomRoll, create_image_loaders, midi_transforms, image_transforms


def generate_random_string(length=6):
    """Generate a random string for unique augmentation identification."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class PreEncoder:
    def __init__(self, vae, data_path, output_dir, image_size=128, max_storage_gb=50, num_workers=None, batch_size=100, device='cuda'):
        self.vae = vae
        self.output_dir = Path(output_dir).expanduser().absolute()
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.current_storage = 0
        self.image_size = image_size
        self.batch_size = batch_size  # Set default batch size to 100
        self.storage_lock = threading.Lock()
        self.file_size = None
        self.device = device
    
        # Automatically determine optimal number of workers based on system
        if num_workers is None:
            # Use 75% of available cores, but ensure we don't go too high
            self.num_workers = min(int(os.cpu_count() * 0.75), 64)
        else:
            self.num_workers = num_workers
        print(f"Using {self.num_workers} workers for data loading")

        if self.output_dir.exists() and self.output_dir.is_dir():
            print(f"Found existing directory {self.output_dir}. Aborting")
            #shutil.rmtree(self.output_dir)
            sys.exit(1)
        print(f"Creating output directory {self.output_dir}")
        try:
            self.output_dir.mkdir(parents=True)
            print(f"Directory created successfully, exists check: {self.output_dir.exists()}")
        except Exception as e:
            print(f"Error creating directory: {e}")
        
        if data_path is None or 'flowers' in data_path.lower():
            transform = image_transforms(image_size=self.image_size)
            self.dataset = datasets.Flowers102(root=data_path, split='train', transform=transform, download=True)
        else: 
            transform = midi_transforms(image_size=self.image_size, random_roll=True)
            self.dataset = MIDIImageDataset(transform=transform, debug=True)

        self.dataset = InfiniteDataset(self.dataset)  # Wrap in infinite dataset for augmentation


    def encode_batch(self, batch):
        """Encode a batch of image tensors using VAE with proper scaling."""
        with torch.no_grad():
            batch = batch.to(self.device, non_blocking=True)
            if not self.vae.is_sd: 
                return self.vae.encode(batch)
            if batch.min() >= 0 and batch.max() <= 1:  # SD VAE expects images in [-1, 1]
                batch = 2 * batch - 1  # If images are [0, 1], convert to [-1, 1]  
            latents = self.vae.encode(batch).latent_dist.mean
            return latents * self.vae.scaling_factor   # SD uses this scaling factor to make latents closer to N(0,1)


    def process_dataset(self, augs_per=512):
        """Process dataset with parallel processing and threaded file saving."""
        print(f"Processing dataset with {self.dataset.actual_len} images")
        
        # Enable pinned memory for faster CPU->GPU transfers
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,  # Prefetch batches
        )
        max_batches = (augs_per * self.dataset.actual_len + 1) // self.batch_size
        print(f"Max batches to process: {max_batches}")
        
        print(f"\n{'='*80}")
        print(f"Processing up to {max_batches} batches with optimizations")
        print(f"Storage limit: {self.max_storage_bytes/1024**3:.1f} GB")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Average number of augmentations per image: {augs_per}")
        print(f"{'='*80}")
        
        # Create a thread pool for file I/O
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            # Create a progress bar
            with tqdm(total=max_batches, desc=f"Processing batches [0.00/{self.max_storage_bytes/1024**3:.1f} GB (0%)]") as pbar:
                for batch_idx, (images, classes) in enumerate(dataloader):
                    if batch_idx >= max_batches:
                        break
                        
                    if self.current_storage >= self.max_storage_bytes:
                        print(f"\nStorage limit reached: {self.current_storage/1024**3:.2f} GB")
                        break
                    
                    # Process batch
                    encoded_batch = self.encode_batch(images)
                    
                    # Submit file saving tasks to thread pool
                    future_to_filename = {}
                    
                    # Process each sample in the batch
                    for i in range(encoded_batch.shape[0]):
                        # Get single encoding and class
                        encoding = encoded_batch[i].cpu()  # Move to CPU for saving
                        class_idx = classes[i].item() if isinstance(classes, torch.Tensor) else classes
                        
                        # Create filename with class info and use hierarchical directories
                        sample_id = f"{batch_idx}_{i}"
                        # Use first 2 digits as subdirectory to avoid too many files in one dir
                        sub_dir = f"{batch_idx % 100:02d}"
                        sub_dir_path = self.output_dir / sub_dir
                        
                        # Create subdirectory if it doesn't exist
                        if not sub_dir_path.exists():
                            sub_dir_path.mkdir(exist_ok=True)
                        
                        # Include class in filename only if we have class info
                        if hasattr(self.dataset, 'n_classes') and self.dataset.n_classes > 0:
                            filename = f"sample_{sample_id}_class{class_idx}_{generate_random_string(4)}.pt"
                        else:
                            filename = f"sample_{sample_id}_{generate_random_string(4)}.pt"
                            
                        out_path = sub_dir_path / filename
                        
                        # Submit save task to thread pool
                        future = executor.submit(torch.save, encoding, out_path)
                        future_to_filename[future] = out_path
                    
                    # Wait for all files in this batch to be saved and update storage
                    batch_file_sizes = 0
                    for future in concurrent.futures.as_completed(future_to_filename):
                        filename = future_to_filename[future]
                        try:
                            # Get result (None for torch.save) and file size
                            future.result()  # This will raise exception if save failed
                            batch_file_sizes += os.path.getsize(filename)
                        except Exception as e:
                            print(f"Error saving {filename}: {e}")
                    
                    # Update storage tracking
                    with self.storage_lock:
                        self.current_storage += batch_file_sizes
                    
                    # Update progress
                    if batch_idx % 1 == 0: # leave it at 1 for accurate timing estimates
                        storage_pct = int(self.current_storage * 100 / self.max_storage_bytes)
                        pbar.set_description(
                            f"Processing batches [{self.current_storage/1024**3:.2f}/{self.max_storage_bytes/1024**3:.1f} GB ({storage_pct}%)]"
                        )
                        pbar.update(1)
                    
                    # Force CUDA synchronization and garbage collection
                    torch.cuda.synchronize()
                    if batch_idx % 50 == 0:  # Don't do this too often
                        torch.cuda.empty_cache()


def parse_args_with_config():
    """
    This lets you specify args via a YAML config file and/or override those with command line args.
    i.e. CLI args take precedence 
    """
    import argparse
    import yaml 

    print("first pass to check for config file")
    # First pass to check for config file
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None, help='path to config YAML file')
    args, _ = parser.parse_known_args()
    config = {}  # Initialize config

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config_full = yaml.safe_load(f)
        
        # Add global data if it exists
        if 'data' in yaml_config_full:
            config["data"] = yaml_config_full['data']
        
        # Process sections
        sections_to_process = ['vqgan', 'preencoding']
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

    print("Second pass to check for CLI args") 
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, 
                       default=config.get('data', None), 
                       help='path to top-level-directory containing custom image data')
    parser.add_argument('--output_dir', default=config.get('data', '.')+'_encoded', type=str,
                      help='directory to save encoded tensors')
    parser.add_argument('--max_storage_gb', type=float, default=50,
                      help='maximum storage to use in gigabytes')
    parser.add_argument('--augs_per', type=int, default=config.get('augs-per',512),
                      help='number of augmented versions to create per image')
    parser.add_argument('--num_workers', type=int, default=config.get('num-workers', None),
                      help='number of workers for data loading')
    parser.add_argument('--image_size', type=int, default=config.get('image-size', 128),
                      help='size to resize images to')
    parser.add_argument('--batch_size', type=int, default=config.get('batch-size', None),
                      help='batch size for processing')
    parser.add_argument('--vqgan-checkpoint', type=str, 
                       default=config.get('vqgan-checkpoint', None), 
                       help='path to load vqvgan checkpoint to resume training from')
    parser.add_argument('--no_grad_ckpt', action='store_true',
                      help='disable gradient checkpointing')
    
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
    # add --config to avoid errors
    parser.add_argument('--config', type=str, default=None, help='path to config YAML file')
    args = parser.parse_args()

    return args 


def preencode_data():
    # cuda setups for speed
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.deterministic = False 

    args = parse_args_with_config()
    print("args = ", args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize vae
    if 'SD' in args.vqgan_checkpoint or 'stable-diffusion' in args.vqgan_checkpoint:
        try:
            from diffusers.models import AutoencoderKL
        except ImportError:
            raise ImportError("To use SD VAE, you need to install diffusers. Try: pip install diffusers")

        print(f"Loading (VQ)VAE checkpoint from HuggingFace Diffusers")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").eval().to(device)
        vae.is_sd = True
        vae.scaling_factor = 1.0  # Use 1.0 as the scaling factor to avoid mangling
        if args.image_size % 8 != 0:
            print(f"Warning: SD VAE works best with image sizes divisible by 8. Current size: {args.image_size}")
    else:
        from flocoder.models.vqvae import VQVAE
        vae = VQVAE(
            in_channels=3,
            hidden_channels=args.hidden_channels,
            num_downsamples=args.num_downsamples,
            internal_dim=args.internal_dim,
            vq_embedding_dim=args.vq_embedding_dim,
            codebook_levels=args.codebook_levels,
            use_checkpoint=not args.no_grad_ckpt,# this refers to gradient checkpointing
            no_natten=False,
        ).eval().to(device)
        print(f"Loading (VQ)VAE checkpoint from {args.vqgan_checkpoint}")
        checkpoint = torch.load(args.vqgan_checkpoint, map_location=device, weights_only=True)
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae.is_sd = False
        vae.scaling_factor = 1.0 

    print("(VQ)VAE checkpoint loaded successfully")
    

    encoder = PreEncoder(vae, args.data, args.output_dir, args.image_size, args.max_storage_gb, 
                         batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    print(f'Writing to {args.output_dir}')
    if os.path.exists(args.output_dir):
        print(f"Warning/Error: Directory '{args.output_dir}' already exists.")
        print(f"Skipping execution to avoid overwriting it.")
        print(f"If you want to generate new data, either remove {args.output_dir} or choose a different --output-dir")
        sys.exit(1)
    encoder.process_dataset(augs_per=args.augs_per)

if __name__ == "__main__":
    preencode_data()
