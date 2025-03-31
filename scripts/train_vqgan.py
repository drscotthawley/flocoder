# dummy code for now
# scripts/train_vqvae.py
import torch
import yaml
from pathlib import Path

from flocoder.models.vqvae import VQVAE
from flocoder.data.datasets import create_midi_dataloader
from flocoder.training.vqvae_trainer import VQVAETrainer

def main():
    # Load config
    config_path = Path("configs/vqvae_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create model and optimizer
    model = VQVAE(**config['model'])
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])
    
    # Create dataloaders
    train_loader = create_midi_dataloader(**config['data']['train'])
    val_loader = create_midi_dataloader(**config['data']['val'])
    
    # Create trainer
    trainer = VQVAETrainer(model, optimizer, config)
    
    # Load checkpoint if provided
    if config.get('resume_from'):
        trainer.load_checkpoint(config['resume_from'])
    
    # Train model
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()