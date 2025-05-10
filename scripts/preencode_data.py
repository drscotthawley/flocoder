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

from torchvision import transforms
from flocoder.data.datasets import ImageListDataset, fast_scandir
from flocoder.data.dataloaders import RandomRoll, create_image_loaders, standard_midi_transforms
from flocoder.models.vqvae import VQVAE


def generate_random_string(length=6):
    """Generate a random string for unique augmentation identification."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


class PreEncoder:
    # TODO: only reason this is a class is because Claude wanted to do it that way and I didn't resist. -SHH
    def __init__(self, vqvae, output_dir, image_size=128, max_storage_gb=10):
        self.vqvae = vqvae
        self.output_dir = Path(output_dir)
        self.max_storage_bytes = max_storage_gb * 1024**3
        self.current_storage = 0
        self.image_size = image_size

        # TODO: use create_image_loaders() instead of this 
        self.transform = standard_midi_transforms(image_size=image_size)
        
        if self.output_dir.exists() and self.output_dir.is_dir(): # wipe it if it exists
            print(f"Found existing directory {self.output_dir}. Wiping it")
            shutil.rmtree(self.output_dir)
        print(f"Creating directory {self.output_dir}")
        self.output_dir.mkdir(parents=True)
        
        # Load all splits of the dataset
        #data_path = "/data/POP909_images/"
        data_path = os.path.expanduser('~')+'/datasets/POP909_images'
        _, file_list = fast_scandir(data_path, ['jpg', 'jpeg', 'png'])
        print(f"Found {len(file_list)} images in {data_path}")
        self.dataset = ImageListDataset(file_list, self.transform)
        
    def encode_single(self, image_tensor):
        """Encode a single image tensor using VQVAE."""
        with torch.no_grad():
            encoded = self.vqvae.encode(image_tensor.unsqueeze(0).to(self.vqvae.device))
            return self.vqvae.compress(encoded)
    
    def save_encoding(self, encoded, output_path):
        """Save encoded tensor and track storage usage."""
        torch.save(encoded, output_path)
        file_size = os.path.getsize(output_path)
        self.current_storage += file_size
        
        # Periodically check total directory size
        if random.random() < 0.1:  # 10% chance
            total = sum(
                os.path.getsize(f) for f in self.output_dir.glob('**/*')
                if f.is_file()
            )
            if total > self.current_storage:
                self.current_storage = total
    
    def process_dataset(self, augmentations_per_image=5, batch_size=32):
        """Process entire dataset with batched processing."""
        # Create a dataset that gives us access to filenames
        print("self.dataset._image_files[0] =",self.dataset._image_files[0])
        filenames = [Path(f).stem for f in self.dataset._image_files]
        
        # Create indices for the entire dataset
        indices = list(range(len(self.dataset)))
        
        print(f"Processing dataset with {len(self.dataset)} images")
        
        for aug_idx in range(augmentations_per_image):
            print(f"\nGenerating augmentation set {aug_idx + 1}/{augmentations_per_image}")
            
            # Create batches of indices
            random.shuffle(indices)  # Shuffle for each augmentation round
            for i in tqdm(range(0, len(indices), batch_size)):
                if self.current_storage >= self.max_storage_bytes:
                    print(f"\nStorage limit of {self.max_storage_bytes/1024**3:.1f}GB reached.")
                    return
                
                batch_indices = indices[i:i + batch_size]
                batch = torch.stack([self.dataset[idx][0] for idx in batch_indices])
                batch_filenames = [filenames[idx] for idx in batch_indices]
                
                # Move batch to device and encode
                batch = batch.to(self.vqvae.device)
                encoded = self.vqvae.compress(self.vqvae.encode(batch))
                
                # Save each encoding in the batch
                for enc, fname in zip(encoded, batch_filenames):
                    # Generate unique identifier for this augmentation
                    aug_id = generate_random_string()
                    
                    # Create filename preserving original name pattern
                    out_path = self.output_dir / f"{fname}_{aug_id}.pt"
                    self.save_encoding(enc, out_path)
        


def parse_args_with_config():
    """
    This lets you specify args via a YAML config file and/or override those with command line args.
    i.e. CLI args take precedence 
    """
    import argparse
    import yaml 

    # First pass to check for config file
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default=None, help='path to config YAML file')
    args, _ = parser.parse_known_args()
    
    # Load config file if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            yaml_config_full = yaml.safe_load(f)
        if 'vqgan' in yaml_config_full: # grab just the vqgan section
            yaml_config = yaml_config_full['vqgan']
        if 'data' in yaml_config_full:
            yaml_config["data"] = yaml_config_full['data'] 
        
        # Flatten nested structure, and treat config variable names as CLI args
        for section, params in yaml_config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    config[key.replace('_', '-')] = value
            else:
                config[section.replace('_', '-')] = params

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, 
                       default=config.get('data', None), 
                       help='path to top-level-directory containing custom image data')
    parser.add_argument('--output_dir', default=None,
                      help='directory to save encoded tensors')
    parser.add_argument('--max_storage_gb', type=float, default=50,
                      help='maximum storage to use in gigabytes')
    parser.add_argument('--augmentations_per_image', type=int, default=1000,
                      help='number of augmented versions to create per image')
    parser.add_argument('--image_size', type=int, default=128,
                      help='size to resize images to')
    parser.add_argument('--batch_size', type=int, default=32,
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
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data, '_encoded')
        
    return args 




def main():

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
    
    encoder = PreEncoder(vqvae, args.output_dir, args.image_size, args.max_storage_gb)
    print(f'Writing to {args.output_dir}')
    encoder.process_dataset(args.augmentations_per_image, args.batch_size)

if __name__ == "__main__":
    main()

