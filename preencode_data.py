#! /usr/bin/env python3

# As per Esser et al "Scaling Rectified Flow Transformers....":
# "Before scaling up, we filter and preencode our data to ensure safe and efficient pretraining."

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
import hydra
from omegaconf import DictConfig
from types import SimpleNamespace

from flocoder.general import handle_config_path
from flocoder.codecs import load_codec
from flocoder.data import ImageListDataset, fast_scandir, MIDIImageDataset, InfiniteDataset
from flocoder.data import create_image_loaders, midi_transforms, image_transforms


def generate_random_string(length=6):
    """Generate a random string for unique augmentation identification."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def encode_batch(codec, batch, device):
    """Encode a batch of image tensors using codec with proper scaling."""
    with torch.no_grad():
        batch = batch.to(device, non_blocking=True)
        return codec.encode(batch)

def setup_dataset(data_path, image_size):
    """Set up and return the appropriate dataset based on the data path."""
    if data_path is None or 'flowers' in str(data_path).lower():
        transform = image_transforms(image_size=image_size)
        dataset = datasets.Flowers102(root=data_path, split='train', transform=transform, download=True)
    else: 
        transform = midi_transforms(image_size=image_size, random_roll=True)
        dataset = MIDIImageDataset(transform=transform, debug=True)
    
    return InfiniteDataset(dataset)  # Wrap in infinite dataset for augmentation


def setup_output_dir(output_dir):
    """Set up and validate the output directory."""
    output_dir = Path(output_dir).expanduser().absolute()
    
    if output_dir.exists() and output_dir.is_dir():
        print(f"Found existing directory {output_dir}. Aborting")
        sys.exit(1)
    
    print(f"Creating output directory {output_dir}")
    try:
        output_dir.mkdir(parents=True)
        print(f"Directory created successfully, exists check: {output_dir.exists()}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        sys.exit(1)
        
    return output_dir


def process_dataset(codec, dataset, output_dir, batch_size, max_storage_bytes, num_workers, device, augs_per=512):
    """Process dataset with parallel processing and threaded file saving."""
    print(f"Processing dataset with {dataset.actual_len} images")
    
    # Track storage usage
    current_storage = 0
    storage_lock = threading.Lock()
    
    # Enable pinned memory for faster CPU->GPU transfers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,  # Prefetch batches
    )
    max_batches = (augs_per * dataset.actual_len + 1) // batch_size
    print(f"Max batches to process: {max_batches}")
    
    print(f"\n{'='*80}")
    print(f"Processing up to {max_batches} batches with optimizations")
    print(f"Storage limit: {max_storage_bytes/1024**3:.1f} GB")
    print(f"Batch size: {batch_size}")
    print(f"Number of workers: {num_workers}")
    print(f"Average number of augmentations per image: {augs_per}")
    print(f"{'='*80}")
    
    # Create a thread pool for file I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Create a progress bar
        with tqdm(total=max_batches, desc=f"Processing batches [0.00/{max_storage_bytes/1024**3:.1f} GB (0%)]") as pbar:
            for batch_idx, (images, classes) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                if current_storage >= max_storage_bytes:
                    print(f"\nStorage limit reached: {current_storage/1024**3:.2f} GB")
                    break
                
                encoded_batch = encode_batch(codec, images, device) # Process batch
                
                future_to_filename = {} # Submit file saving tasks to thread pool
                
                # Process each sample in the batch
                for i in range(encoded_batch.shape[0]):
                    # Get single encoding and class
                    encoding = encoded_batch[i].cpu()  # Move to CPU for saving
                    class_idx = classes[i].item() if isinstance(classes, torch.Tensor) else classes
                    
                    # Create filename with class info and use hierarchical directories
                    sample_id = f"{batch_idx}_{i}"
                    # Use first 2 digits as subdirectory to avoid too many files in one dir
                    sub_dir = f"{batch_idx % 100:02d}"
                    sub_dir_path = output_dir / sub_dir
                    
                    # Create subdirectory if it doesn't exist
                    if not sub_dir_path.exists():
                        sub_dir_path.mkdir(exist_ok=True)
                    
                    # Include class in filename only if we have class info
                    if hasattr(dataset, 'n_classes') and dataset.n_classes > 0:
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
                with storage_lock:
                    current_storage += batch_file_sizes
                
                # Update progress
                if batch_idx % 1 == 0:  # leave it at 1 for accurate timing estimates
                    storage_pct = int(current_storage * 100 / max_storage_bytes)
                    pbar.set_description(
                        f"Processing batches [{current_storage/1024**3:.2f}/{max_storage_bytes/1024**3:.1f} GB ({storage_pct}%)]"
                    )
                    pbar.update(1)
                
                # Force CUDA synchronization and garbage collection
                torch.cuda.synchronize()
                if batch_idx % 50 == 0:  # Don't do this too often
                    torch.cuda.empty_cache()


handle_config_path()  # allow for full path in --config-name
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers")
def main(cfg) -> None:
    """Main entry point using Hydra."""
    # Debug - print the config structure
    print("Config keys:", list(cfg.keys()))
    print("Full config:", cfg)

    # Set up CUDA
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get configuration values with defaults
    data_path = cfg.data
    output_dir = f"{data_path}_encoded_{cfg.codec.choice}" if not cfg.get('output_dir') else cfg.output_dir
    image_size = cfg.get('image_size', 128)
    max_storage_gb = cfg.get('max_storage_gb', 50)
    batch_size = cfg.get('batch_size', 32)
    augs_per = cfg.get('augs_per', 512)
    num_workers = cfg.get('num_workers', min(int(os.cpu_count() * 0.75), 64))
    
    codec = load_codec(cfg, device)
    
    # Setup dataset and output directory
    dataset = setup_dataset(data_path, image_size)
    output_dir = setup_output_dir(output_dir)
    
    # Process the dataset
    process_dataset(
        codec=codec,
        dataset=dataset,
        output_dir=output_dir,
        batch_size=batch_size,
        max_storage_bytes=max_storage_gb * 1024**3,
        num_workers=num_workers,
        device=device,
        augs_per=augs_per
    )


if __name__ == "__main__":
    main()
