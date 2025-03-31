# dummy code for now
# flocoder/training/vqvae_trainer.py
import torch
from pathlib import Path
import logging

class VQVAETrainer:
    def __init__(self, model, optimizer, config, device=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Set up logging, checkpoint dirs, etc.
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
    def train(self, train_loader, val_loader=None, num_epochs=None):
        """Main training loop"""
        num_epochs = num_epochs or self.config.get('num_epochs', 100)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self._train_epoch(train_loader)
            
            # Validate if needed
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate(val_loader)
            
            # Save checkpoint
            self._save_checkpoint()
            
            # Log metrics
            self._log_metrics({**train_metrics, **val_metrics})
            
        return self.model
    
    def _train_epoch(self, train_loader):
        """Train for a single epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            # Forward pass
            x = batch[0].to(self.device)
            x_recon, commit_loss = self.model(x)
            
            # Loss calculation
            recon_loss = torch.nn.functional.mse_loss(x_recon, x)
            loss = recon_loss + commit_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
        return {"train_loss": epoch_loss / len(train_loader)}
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)
                x_recon, commit_loss = self.model(x)
                
                recon_loss = torch.nn.functional.mse_loss(x_recon, x)
                loss = recon_loss + commit_loss
                
                val_loss += loss.item()
                
        return {"val_loss": val_loss / len(val_loader)}
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        checkpoint_path = self.checkpoint_dir / f"vqvae_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / "vqvae_latest.pt"
        torch.save(checkpoint, latest_path)
        
    def _log_metrics(self, metrics):
        """Log metrics (to console, file, or tracking service)"""
        log_str = f"Epoch {self.current_epoch} | " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logging.info(log_str)
        
    def load_checkpoint(self, checkpoint_path):
        """Load from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.global_step = checkpoint['global_step']
        
        return self