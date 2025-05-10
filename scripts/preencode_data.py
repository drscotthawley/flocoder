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

from flocoder.data.datasets import ImageListDataset, fast_scandir, MIDIImageDataset, InfiniteDataset
from flocoder.data.dataloaders import RandomRoll, create_image_loaders, midi_transforms
from flocoder.models.vqvae import VQVAE


def generate_random_string(length=6):
    """Generate a random string for unique augmentation identification."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class PreEncoder:
    # only reason this is a class is because Claude made it that way ;-) -SHH
    def __init__(self, vqvae, data_path, output_dir, image_size=128, max_storage_gb=50, num_workers=None, batch_size=100):
        self.vqvae = vqvae
        self.output_dir = Path(output_dir).expanduser().absolute()
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.current_storage = 0
        self.image_size = image_size
        self.batch_size = batch_size  # Set default batch size to 100
        self.storage_lock = threading.Lock()
        self.file_size = None
    
    # Rest of initialization code...

        # Automatically determine optimal number of workers based on system
        if num_workers is None:
            # Use 75% of available cores, but ensure we don't go too high
            self.num_workers = min(int(os.cpu_count() * 0.75), 64)
        else:
            self.num_workers = num_workers
        print(f"Using {self.num_workers} workers for data loading")

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
        self.dataset = InfiniteDataset(self.dataset)  # Wrap in infinite dataset for augmentation


    def encode_batch(self, batch):
        """Encode a batch of image tensors using VQVAE with prefetching."""
        with torch.no_grad(): # and torch.amp.autocast(device_type='cuda'): # optional: Using autocast for mixed precision will shave a few minutes off
            batch = batch.to(self.vqvae.device, non_blocking=True) 
            return self.vqvae.encode(batch)
    
    def save_encodings(self, encoded_batch, filenames):
        """Save an entire batch of encoded tensors as a single file."""
        # Generate a unique batch identifier
        batch_id = generate_random_string(8)
        
        # Create batch data structure
        batch_data = {
            'encodings': encoded_batch,
            'filenames': filenames,
            'timestamp': time.time(),
            'batch_id': batch_id
        }
        assert encoded_batch.shape[0] == self.batch_size, f"Batch size mismatch: {encoded_batch.shape[0]} != {self.batch_size}"
        
        # Create filename for the batch
        out_path = self.output_dir / f"batch_{batch_id}.pt"
        
        # Save the batch as a single file
        torch.save(batch_data, out_path)
        
        # Update storage tracking
        file_size = os.path.getsize(out_path)
        with self.storage_lock:
            self.current_storage += file_size
        
        # Periodically check total directory size (less frequent now that we have fewer files)
        if random.random() < 0.01:  # 1% chance
            total = sum(
                os.path.getsize(f) for f in self.output_dir.glob('**/*')
                if f.is_file()
            )
            if total > self.current_storage:
                self.current_storage = total



    def process_dataset(self, augs_per=512):
        """Process dataset with optimized batch processing."""
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
        max_batches = (augs_per * self.dataset.actual_len + 1 )// self.batch_size
        print(f"Max batches to process: {max_batches}")
        
        print(f"\n{'='*80}")
        print(f"Processing up to {max_batches} batches with optimizations")
        print(f"Storage limit: {self.max_storage_bytes/1024**3:.1f} GB")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Average number of augmentations per image: {augs_per}")
        print(f"{'='*80}")
        
        # Create a progress bar
        with tqdm(total=max_batches, desc=f"Processing batches [0.00/{self.max_storage_bytes/1024**3:.1f} GB (0%)]") as pbar:
            # Process batches with prefetching
            batch_buffer = []
            buffer_size = 5  # Buffer several batches for bulk processing
            
            for batch_idx, (images, _) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                if self.current_storage >= self.max_storage_bytes:
                    print(f"\nStorage limit reached: {self.current_storage/1024**3:.2f} GB")
                    break
                
                # Process batch
                encoded_batch = self.encode_batch(images)
                
                # Add to buffer
                batch_filenames = [f"sample_{batch_idx}_{i}_{generate_random_string(4)}" 
                                for i in range(images.size(0))]
                batch_buffer.append((encoded_batch, batch_filenames))
                
                # Process buffer when full
                if len(batch_buffer) >= buffer_size:
                    self._process_batch_buffer(batch_buffer)
                    batch_buffer = []
                    
                    # Force CUDA synchronization and garbage collection
                    torch.cuda.synchronize()
                    if batch_idx % 50 == 0:  # Don't do this too often
                        torch.cuda.empty_cache()
                
                # Update progress
                if batch_idx % 1 == 0: # leave it at 1 for accurate timing estimates
                    storage_pct = int(self.current_storage * 100 / self.max_storage_bytes)
                    pbar.set_description(
                        f"Processing batches [{self.current_storage/1024**3:.2f}/{self.max_storage_bytes/1024**3:.1f} GB ({storage_pct}%)]"
                    )
                    pbar.update(1)
            
            # Process remaining batches
            if batch_buffer:
                self._process_batch_buffer(batch_buffer)


    def _process_batch_buffer(self, batch_buffer):
        """Process multiple batches efficiently."""
        # Combine batches into larger chunks for fewer file operations
        combined_batch_id = generate_random_string(8)
        combined_data = {
            'encodings': [item[0] for item in batch_buffer],
            'filenames': [item[1] for item in batch_buffer],
            'timestamp': time.time(),
            'batch_id': combined_batch_id
        }
        
        # Create filename for the combined batch
        out_path = self.output_dir / f"batch_{combined_batch_id}.pt"
        
        # Save combined data
        torch.save(combined_data, out_path)
        
        # Update storage tracking
        with self.storage_lock:
            self.file_size = os.path.getsize(out_path)
            self.current_storage += self.file_size


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
        
        config = {}  # Initialize config
        
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
    parser.add_argument('--augs_per', type=int, default=config.get('augs-per',512),
                      help='number of augmented versions to create per image')
    parser.add_argument('--num_workers', type=int, default=config.get('num-workers', None),
                      help='number of workers for data loading')
    parser.add_argument('--image_size', type=int, default=128,
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

    # cuda setups for speed
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.deterministic = False 

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

    print(f"Loading VQVAE checkpoint from {args.vqgan_checkpoint}")
    checkpoint = torch.load(args.vqgan_checkpoint, map_location=device, weights_only=True)
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    print("VQVAE checkpoint loaded successfully")
    
    encoder = PreEncoder(vqvae, args.data, args.output_dir, args.image_size, args.max_storage_gb, 
                         batch_size=args.batch_size, num_workers=args.num_workers)
    print(f'Writing to {args.output_dir}')
    #encoder.process_dataset(args.augmentations_per_image, batch_size=args.batch_size)
    encoder.process_dataset(augs_per=args.augs_per)

if __name__ == "__main__":
    preencode_data()

