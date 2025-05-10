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


from torchvision import transforms
from flocoder.data.datasets import ImageListDataset, fast_scandir, MIDIImageDataset
from flocoder.data.dataloaders import RandomRoll, create_image_loaders, midi_transforms
from flocoder.models.vqvae import VQVAE


def generate_random_string(length=6):
    """Generate a random string for unique augmentation identification."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class PreEncoder:
    def __init__(self, vqvae, data_path, output_dir, image_size=128, max_storage_gb=50, num_workers=None):
        self.vqvae = vqvae
        #self.output_dir = Path(output_dir)
        self.output_dir = Path(output_dir).expanduser().absolute()
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.current_storage = 0
        self.image_size = image_size

        # Automatically determine optimal number of workers based on system
        if num_workers is None:
            # Use 75% of available cores, but ensure we don't go too high
            self.num_workers = min(int(os.cpu_count() * 0.75), 64)
        else:
            self.num_workers = num_workers

        if self.output_dir.exists() and self.output_dir.is_dir():
            print(f"Found existing directory {self.output_dir}. Wiping it")
            shutil.rmtree(self.output_dir)
        print(f"Creating output directory {self.output_dir}")
        try:
            self.output_dir.mkdir(parents=True)
            print(f"Directory created successfully, exists check: {self.output_dir.exists()}")
        except Exception as e:
            print(f"Error creating directory: {e}")
        
        transform = midi_transforms(image_size=self.image_size, random_roll=True)
        self.dataset = MIDIImageDataset(transform=transform, debug=True)
    
    def encode_batch(self, batch):
        """Encode a batch of image tensors using VQVAE."""
        with torch.no_grad():
            batch = batch.to(self.vqvae.device)
            encoded = self.vqvae.encode(batch)
            return encoded
    
    def save_encodings(self, encoded_batch, filenames):
        """Save multiple encoded tensors and track storage usage."""
        # Create temporary directory for batch saving if needed
        # This could help with filesystem performance by localizing writes
        
        for enc, fname in zip(encoded_batch, filenames):
            # Generate unique identifier for this augmentation
            aug_id = generate_random_string()
            
            # Create filename preserving original name pattern
            out_path = self.output_dir / f"{fname}_{aug_id}.pt"
            torch.save(enc, out_path)
            
            # Update storage tracking
            file_size = os.path.getsize(out_path)
            self.current_storage += file_size
        
        # Periodically check total directory size
        if random.random() < 0.05:  # 5% chance to reduce overhead
            total = sum(
                os.path.getsize(f) for f in self.output_dir.glob('**/*')
                if f.is_file()
            )
            if total > self.current_storage:
                self.current_storage = total
  
    
    def process_dataset(self, augmentations_per_image=512, batch_size=None):
        """Process dataset with parallel augmentation rounds."""
        if batch_size is not None:
            self.batch_size = batch_size
            
        print(f"Processing dataset with {len(self.dataset)} images, performing {augmentations_per_image} augmentations per image")
        
        # Get filenames for all images in the dataset
        if hasattr(self.dataset, 'midi_img_file_list'):
            file_list = self.dataset.midi_img_file_list
        elif hasattr(self.dataset, '_image_files'):
            file_list = self.dataset._image_files
        elif hasattr(self.dataset, 'files'):
            file_list = self.dataset.files
        else:
            raise AttributeError("Cannot find image file list in dataset")
                
        filenames = [Path(f).stem for f in file_list]
        
        # Determine how many rounds to process in parallel
        max_parallel_rounds = min(16, os.cpu_count() // 4)
        
        # Create shared data with locks
        self.storage_lock = threading.Lock()
        self.total_batches_processed = 0
        num_batches_per_round = len(self.dataset) // self.batch_size + (1 if len(self.dataset) % self.batch_size else 0)
        total_batches = num_batches_per_round * augmentations_per_image
        
        print(f"\n{'='*80}")
        print(f"Starting processing with {max_parallel_rounds} parallel augmentation rounds")
        print(f"Storage limit: {self.max_storage_bytes/1024**3:.1f} GB")
        print(f"Total batches to process: {total_batches}")
        print(f"{'='*80}")
        
        # Create a global progress bar for batches
        with tqdm(total=total_batches, desc="Processing batches", unit="batch") as self.global_pbar:
            # Process augmentation rounds in batches
            for start_idx in range(0, augmentations_per_image, max_parallel_rounds):
                # Check if we've already hit storage limit
                if self.current_storage >= self.max_storage_bytes:
                    print(f"\nStorage limit of {self.max_storage_bytes/1024**3:.1f} GB reached")
                    break
                    
                # Calculate number of rounds for this batch
                end_idx = min(start_idx + max_parallel_rounds, augmentations_per_image)
                current_batch_size = end_idx - start_idx
                
                # Process in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=current_batch_size) as executor:
                    futures = [
                        executor.submit(
                            self._process_single_round, 
                            round_idx, 
                            filenames
                        )
                        for round_idx in range(start_idx, end_idx)
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result == "STORAGE_LIMIT_REACHED":
                                print(f"\nStorage limit of {self.max_storage_bytes/1024**3:.1f} GB reached")
                                return
                        except Exception as e:
                            print(f"Error: {e}")
                
                # Print periodic storage update (just once per parallel batch)
                print(f"Storage: {self.current_storage/1024**3:.2f}/{self.max_storage_bytes/1024**3:.1f} GB "
                      f"({int(self.current_storage * 100 / self.max_storage_bytes)}%)")
    
    def _process_single_round(self, round_idx, filenames):
        """Process a single augmentation round with frequent progress bar updates."""
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Process all batches with minimal output
        for batch_idx, (images, _) in enumerate(dataloader):
            # Check storage limit with thread safety
            with self.storage_lock:
                if self.current_storage >= self.max_storage_bytes:
                    return "STORAGE_LIMIT_REACHED"
            
            # Generate batch filenames
            batch_size = images.size(0)
            batch_filenames = [f"img_{round_idx}_{batch_idx}_{i}" for i in range(batch_size)]
            
            # Process batch
            encoded_batch = self.encode_batch(images)
            self.save_encodings(encoded_batch, batch_filenames)
            
            # Update the global progress bar after each batch - for frequent updates
            with self.storage_lock:
                self.total_batches_processed += 1
                self.global_pbar.update(1)
        
        return "COMPLETED"


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
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config_full = yaml.safe_load(f)
        
        # Initialize config
        config = {}
        
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
    parser.add_argument('--output_dir', default=config.get('data', None)+'_encoded', type=str,
                      help='directory to save encoded tensors')
    parser.add_argument('--max_storage_gb', type=float, default=50,
                      help='maximum storage to use in gigabytes')
    parser.add_argument('--augmentations_per_image', type=int, default=config.get('augmentations-per-image',512),
                      help='number of augmented versions to create per image')
    parser.add_argument('--image_size', type=int, default=128,
                      help='size to resize images to')
    parser.add_argument('--batch_size', type=int, default=config.get('batch-size', None),
                      help='batch size for processing')
    parser.add_argument('--checkpoint', default="vqvae_checkpoint_compressed4_epoch1500.pt",
                      help='path to VQVAE checkpoint file')
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
    parser.add_argument('--vq-embedding-dim', type=int, 
                       default=config.get('vq-embedding-dim', 256), 
                       help='pre-vq emb dim before compression')
    parser.add_argument('--compressed-dim', type=int, 
                       default=config.get('compressed-dim', 4), 
                       help='ACTUAL dims of codebook vectors')
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

    args = parse_args_with_config()
    print("args = ", args)

    # Initialize VQVAE
    vqvae = VQVAE(
        in_channels=3,
        hidden_channels=args.hidden_channels,
        num_downsamples=args.num_downsamples,
        vq_num_embeddings=args.vq_num_embeddings,
        vq_embedding_dim=args.vq_embedding_dim,
        compressed_dim=args.compressed_dim,
        codebook_levels=args.codebook_levels,
        use_checkpoint=not args.no_grad_ckpt,# this refers to gradient checkpointing
        no_natten=False,
    ).eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    vqvae = vqvae.to(device)
    vqvae.device = device

    print(f"Loading VQVAE checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    print("VQVAE checkpoint loaded successfully")
    
    encoder = PreEncoder(vqvae, args.data, args.output_dir, args.image_size, args.max_storage_gb,)
    print(f'Writing to {args.output_dir}')
    encoder.process_dataset(args.augmentations_per_image, batch_size=args.batch_size)

if __name__ == "__main__":
    preencode_data()

