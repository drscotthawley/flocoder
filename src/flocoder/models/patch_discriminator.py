from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F


class DiscrResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.skip = None if stride == 1 and in_channels == out_channels else \
                   spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0))
        self.norm1 = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.use_checkpoint = use_checkpoint

    def _forward(self, x):
        identity = self.skip(x) if self.skip else x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        return self.act(out)

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, n_layers=3, use_checkpoint=False):
        """
        PatchGAN discriminator with DiscrResBlocks.
        Args:
            in_channels: Number of input channels (3 for RGB)
            hidden_channels: Base channel count
            n_layers: Number of downsampling layers
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        
        # Initial conv layer
        layers = [
            spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers with DiscrResBlocks
        current_channels = hidden_channels
        for i in range(n_layers):
            next_channels = min(hidden_channels * (2 ** (i+1)), 512)
            layers.append(DiscrResBlock(current_channels, next_channels, 
                                      stride=2 if i < n_layers-1 else 1,
                                      use_checkpoint=use_checkpoint))
            current_channels = next_channels
            
        # Final layer for patch-wise predictions
        layers.append(
            spectral_norm(nn.Conv2d(current_channels, 1, kernel_size=4, 
                                  stride=1, padding=1))
        )
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        features = []
        for layer in self.main:
            x = layer(x)
            if isinstance(layer, (nn.LeakyReLU, DiscrResBlock)):
                features.append(x)
        return x, features
