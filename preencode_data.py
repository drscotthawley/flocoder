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
from omegaconf import DictConfig, open_dict, OmegaConf
from types import SimpleNamespace

from flocoder.general import handle_config_path, ldcfg
from flocoder.codecs import setup_codec
from flocoder.data import ImageListDataset, fast_scandir, MIDIImageDataset, InfiniteDataset
from flocoder.data import create_image_loaders, midi_transforms, image_transforms
from flocoder.inpainting import InpaintingDataset


def generate_random_string(length=6):
    """Generate a random string for unique augmentation identification."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def encode_batch(codec, x, device, quantize=False):
    """Encode a batch of image tensors using codec with proper scaling."""
    with torch.no_grad():
        x = x.to(device, non_blocking=True)
        z = codec.encode(x)
        if quantize: 
            z_q, _ = codec.quantize(z)
            return z_q
        return z


def setup_dataset(data_path, image_size, config, split='train'):
    """Set up and return the appropriate dataset based on the data path."""
    transform = image_transforms(image_size=image_size)
    if data_path is None or 'flowers' in str(data_path).lower():
        dataset = datasets.Flowers102(root=data_path, split=split, transform=transform, download=True)
    elif 'stl10' in str(data_path).lower():
        dataset = datasets.STL10(root=data_path, split=split, transform=transform, download=True)
    elif 'food101' in str(data_path).lower():
        dataset = datasets.Food101(root=data_path, split=split, transform=transform, download=True)
    else:  # midi
        grayscale = ldcfg(config, 'in_channels', 3)==1
        transform = midi_transforms(image_size=image_size, random_roll=True, grayscale=grayscale)
        dataset = MIDIImageDataset(transform=transform, split=split, config=config, debug=True, total_only=True)
    
    dataset = InfiniteDataset(dataset)  # Wrap in infinite dataset for augmentation
    return dataset



def setup_output_dir(output_dir):
    """Set up and validate the output directory."""
    output_dir = Path(output_dir).expanduser().absolute()
    print(f"Trying to setup  output directory {output_dir}")
    
    if output_dir.exists() and output_dir.is_dir():
        print(f"Found existing directory {output_dir}\nAborting. Delete that directory and try again.")
        sys.exit(1)
    
    print(f"Creating output directory {output_dir}")
    try:
        output_dir.mkdir(parents=True)
        print(f"Directory created successfully, exists check: {output_dir.exists()}")
    except Exception as e:
        print(f"Error creating directory: {e}")
        sys.exit(1)
        
    return output_dir


def process_dataset(codec, dataset, output_dir, batch_size, max_storage_bytes, num_workers, device, 
        augs_per=768, quantize=False, inpainting=False):
    """Process dataset with optimized parallel encoding and saving"""
    print(f"\n\nProcessing dataset with {dataset.actual_len} images, augs_per = {augs_per}, quantize = {quantize}, inpainting = {inpainting}")
    print(f"Dataset type: {type(dataset)}")
    n_classes = 0
    if hasattr(dataset, '_labels'):
        n_classes = len(set(dataset._labels))
    has_classes = n_classes > 0
    print(f"has_classes value: {has_classes}")
    current_storage = 0
    storage_lock = threading.Lock()

    # Setup dataloader with optimizations
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                            pin_memory=True, persistent_workers=True, prefetch_factor=2)
    max_batches = (augs_per * dataset.actual_len + 1) // batch_size

    # Create class directories if needed
    if has_classes:
        print("n_classes =",n_classes)
        for class_idx in range(n_classes):
            (output_dir / str(class_idx)).mkdir(exist_ok=True)

    # Process batches with thread pool for I/O
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        with tqdm(total=max_batches, desc=f"Processing [0.00/{max_storage_bytes/1024**3:.1f} GB (0%)]") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches or current_storage >= max_storage_bytes: break
                if type(batch) == tuple:  # "standard" operation
                    target_images, classes = batch
                elif type(batch) == dict and inpainting:
                    target_images = batch['target_image']
                    mask_pixels = batch['mask_pixels']
                    source_images = batch['source_image']
                    classes = batch['label']
                    source_latents = encode_batch(codec, source_images, device, quantize=quantize)
                else:
                    raise ValueError(f"Unexpected batch format: {type(batch)}")

                target_latents = encode_batch(codec, target_images, device, quantize=quantize)

                future_to_path = {}

                # Submit file saving tasks
                class_indices = set()
                for i in range(target_latents.shape[0]):
                    encoding = target_latents[i].cpu()
                    class_idx = classes[i].item() if isinstance(classes, torch.Tensor) else classes
                    class_indices.add(class_idx)
                    sample_id = f"{batch_idx}_{i}"

                    if inpainting:
                        data_to_save = {
                            'target_latents': target_latents[i].cpu(),
                            'source_latents': source_latents[i].cpu(),
                            'mask_pixels': mask_pixels[i].cpu().bool() # bool to save storage space
                        }
                    else:
                        data_to_save = target_latents[i].cpu()  # Keep old format for compatibility


                    # Determine output path based on class info
                    if has_classes:
                        out_path = output_dir / str(class_idx) / f"sample_{sample_id}_{generate_random_string(4)}.pt"
                    else:
                        sub_dir = f"{batch_idx % 100:02d}"
                        sub_dir_path = output_dir / sub_dir
                        sub_dir_path.mkdir(exist_ok=True)
                        out_path = sub_dir_path / f"sample_{sample_id}_{generate_random_string(4)}.pt"

                    future = executor.submit(torch.save, data_to_save, out_path)
                    future_to_path[future] = out_path
                #if batch_idx % 10 == 0:
                #    print(f"Batch {batch_idx}: Encountered class indices: {sorted(class_indices)}")

                # Process completed saves and update storage count
                batch_file_sizes = 0
                for future in concurrent.futures.as_completed(future_to_path):
                    try:
                        future.result()
                        batch_file_sizes += os.path.getsize(future_to_path[future])
                    except Exception as e:
                        print(f"Error saving {future_to_path[future]}: {e}")

                with storage_lock:
                    current_storage += batch_file_sizes

                # Update progress display
                if batch_idx % 1 == 0:
                    storage_pct = int(current_storage * 100 / max_storage_bytes)
                    pbar.set_description(f"Processing [{current_storage/1024**3:.2f}/{max_storage_bytes/1024**3:.1f} GB ({storage_pct}%)]")
                    pbar.update(1)

                # Clear GPU memory periodically
                torch.cuda.synchronize()
                if batch_idx % 50 == 0: torch.cuda.empty_cache()



handle_config_path()  # allow for full path in --config-name
@hydra.main(version_base="1.3", config_path="configs", config_name="flowers")
def main(config) -> None:
    """Main entry point using Hydra."""
    OmegaConf.set_struct(config, False)  # make it mutable
    # Debug - print the config structure
    print("Config keys:", list(config.keys()))
    print("Full config:", config)
    # hack: neutralize/delete any 'flow' or 'preencoding' sections from config
    with open_dict(config):
        if 'flow' in config: del config['flow']

    # Set up CUDA
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get configuration values with defaults
    data_path = config.data
    inpainting=True # todo: move this somewhere like a config
    output_dir = f"{data_path}_encoded_{config.codec.choice}" if not ldcfg(config,'output_dir') else config.output_dir
    if inpainting: 
        output_dir += "_inpainting"
    print(f"preencode_data: output_dir = ",output_dir)
    image_size = ldcfg(config,'image_size', 128)
    max_storage_gb = ldcfg(config,'max_storage_gb', 60)
    batch_size = ldcfg(config,'batch_size', 32)
    augs_per = ldcfg(config,'augs_per', 512)
    quantize = ldcfg(config,'quantize', False)
    num_workers = ldcfg(config,'num_workers', min(int(os.cpu_count() * 0.75), 64))
    
    codec = setup_codec(config, device)
    
    for split in ['val','train']: 
        print(f"\nWorking on split =",split)
        # Setup dataset and output directory
        dataset = setup_dataset(data_path, image_size, config, split=split)

        if inpainting:  # this will make it return a dict
            #dataset = InpaintingDataset(dataset)
            mask_kwargs = {'choices':['brush','rectangles','noise'], 'p':[0.55, 0.4, 0.05]} # leave out 'total' and 'nothing' for on-the-fly, later
            print("creating InpaintingDataset with mask_kwargs =",mask_kwargs)
            dataset = InpaintingDataset(dataset, mask_kwargs=mask_kwargs)

        split_output_dir = setup_output_dir(output_dir+f"/{split}")
        
        # Process the dataset
        process_dataset(
            codec=codec,
            dataset=dataset,
            output_dir=split_output_dir,
            batch_size=batch_size,
            max_storage_bytes=max_storage_gb * 1024**3,
            num_workers=num_workers,
            device=device,
            augs_per=augs_per,
            quantize=quantize,
            inpainting=inpainting,
        )


if __name__ == "__main__":
    print("\nScript invoked via:\n", " ".join(sys.argv),"\n")
    print("cwd is",os.getcwd())
    main()
