import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint
import torchvision.models as models
import math
from .codecs import gn_groups


class VQGANPlusResidualBlock(nn.Module):
    """Residual block for VQGAN+ encoder/decoder"""
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm1 = nn.GroupNorm(gn_groups(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(gn_groups(8, out_channels), out_channels)
        self.act = nn.SiLU()
        
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.GroupNorm(gn_groups(8, out_channels), out_channels)
            )
        
        self.use_checkpoint = use_checkpoint
    
    def _forward(self, x):
        if self.skip:
            identity = self.skip(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + identity
        out = self.act(out)
        return out

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class VQGANPlusEncoder(nn.Module):
    """VQGAN+ Encoder - purely convolutional, no attention"""
    def __init__(self, in_channels=3, base_channels=128, channel_multipliers=[1,1,2,2,4], 
                 latent_channels=256, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        
        # Encoder blocks
        layers = []
        current_channels = base_channels
        
        for i, mult in enumerate(channel_multipliers):
            out_channels = base_channels * mult
      
            # Two residual blocks per stage (symmetric design)
            layers.append(VQGANPlusResidualBlock(current_channels, out_channels, 
                                               stride=2,
                                               use_checkpoint=use_checkpoint))
            layers.append(VQGANPlusResidualBlock(out_channels, out_channels, 
                                               stride=1, use_checkpoint=use_checkpoint))
            current_channels = out_channels
        
        # Final conv to latent space
        layers.append(nn.Conv2d(current_channels, latent_channels, 3, 1, 1))
        layers.append(nn.GroupNorm(gn_groups(8, latent_channels), latent_channels))
        layers.append(nn.SiLU())
        
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        return self.encoder(x)


class VQGANPlusDecoder(nn.Module):
    """VQGAN+ Decoder - purely convolutional, no attention"""
    def __init__(self, out_channels=3, base_channels=128, channel_multipliers=[1,1,2,2,4],
                 latent_channels=256, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.out_channels = out_channels
        
        # Reverse the multipliers for decoder
        rev_multipliers = list(reversed(channel_multipliers))
        
        # Initial conv from latent space
        self.init_conv = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels * rev_multipliers[0], 3, 1, 1),
            nn.GroupNorm(gn_groups(8, base_channels * rev_multipliers[0]), 
                        base_channels * rev_multipliers[0]),
            nn.SiLU()
        )
        
        # Decoder blocks  
        layers = []
        current_channels = base_channels * rev_multipliers[0]
        
        for i, mult in enumerate(rev_multipliers[1:], 1):
            out_channels = base_channels * mult
            
            # Upsample first (except for first block)
            if i > 0:
                layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            
            # Two residual blocks per stage (symmetric design)
            layers.append(VQGANPlusResidualBlock(current_channels, out_channels, 
                                               stride=1, use_checkpoint=use_checkpoint))
            layers.append(VQGANPlusResidualBlock(out_channels, out_channels, 
                                               stride=1, use_checkpoint=use_checkpoint))
            current_channels = out_channels
        
        # Final upsample and output conv
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv2d(current_channels, self.out_channels, 3, 1, 1))
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        return self.decoder(x)


class GaussianBlur(nn.Module):
    """4x4 Gaussian blur kernel with stride 2 for downsampling"""
    def __init__(self):
        super().__init__()
        # Precomputed 4x4 Gaussian kernel
        kernel = torch.tensor([[1., 2., 2., 1.],
                              [2., 4., 4., 2.], 
                              [2., 4., 4., 2.],
                              [1., 2., 2., 1.]]) / 36.0
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))
        
    def forward(self, x):
        # Apply Gaussian blur with stride 2 for downsampling
        return F.conv2d(x, self.kernel.repeat(x.shape[1], 1, 1, 1), 
                       stride=2, padding=1, groups=x.shape[1])


class VQGANPlusDiscrResBlock(nn.Module):
    """Discriminator residual block with GroupNorm"""
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=True):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1, stride, 0))
        
        # Use GroupNorm instead of BatchNorm
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


class VQGANPlusDiscriminator(nn.Module):
    """VQGAN+ PatchGAN discriminator with improvements"""
    def __init__(self, in_channels=3, base_channels=128, n_layers=3, use_checkpoint=True):
        super().__init__()
        
        # Initial conv - 3x3 instead of 4x4
        layers = [
            spectral_norm(nn.Conv2d(in_channels, base_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        current_channels = base_channels
        for i in range(n_layers):
            next_channels = min(base_channels * (2 ** (i+1)), 512)
            
            # Add Gaussian blur before downsampling (except last layer)
            if i < n_layers-1:
                layers.append(GaussianBlur())
                layers.append(VQGANPlusDiscrResBlock(current_channels, next_channels, 
                                                   stride=1, use_checkpoint=use_checkpoint))
            else:
                layers.append(VQGANPlusDiscrResBlock(current_channels, next_channels, 
                                                   stride=1, use_checkpoint=use_checkpoint))
            current_channels = next_channels
            
        # Final prediction layer - 3x3 instead of 4x4
        layers.append(
            spectral_norm(nn.Conv2d(current_channels, 1, 3, 1, 1))
        )
        
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        features = []
        for layer in self.main:
            x = layer(x)
            if isinstance(layer, (nn.LeakyReLU, VQGANPlusDiscrResBlock)):
                features.append(x)
        return x, features


class ResNet50PerceptualLoss(nn.Module):
    """ResNet50 logits-based perceptual loss as used in VQGAN+"""
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.resnet.eval()
        
        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img1, img2):
        # Normalize to ImageNet range
        img1_norm = (img1 - self.mean) / self.std
        img2_norm = (img2 - self.mean) / self.std
        
        # Get logits (before final classification layer)
        with torch.no_grad():
            logits1 = self.resnet(img1_norm)
            logits2 = self.resnet(img2_norm)
        
        return F.mse_loss(logits1, logits2)


def lecam_loss(d_real, d_fake, reg_weight=0.001):
    """LeCAM regularization for discriminator stability"""
    return reg_weight * (torch.mean(F.relu(1.0 + d_real)) + torch.mean(F.relu(1.0 - d_fake)))


def hinge_d_loss(real_pred, fake_pred):
    """Hinge loss for discriminator"""
    return torch.mean(F.relu(1.0 - real_pred)) + torch.mean(F.relu(1.0 + fake_pred))


class VQGANPlusAdversarialLoss(nn.Module):
    """Complete adversarial loss module for VQGAN+"""
    def __init__(self, device, in_channels=3, use_checkpoint=True):
        super().__init__()
        self.device = device
        self.discriminator = VQGANPlusDiscriminator(
            in_channels=in_channels, 
            use_checkpoint=use_checkpoint
        ).to(device)
        
        self.register_buffer('real_label', torch.ones(1))
        self.register_buffer('fake_label', torch.zeros(1))

    def feature_matching_loss(self, real_features, fake_features):
        """Feature matching loss between real and fake features"""
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(real_features)

    def discriminator_loss(self, real_images, fake_images):
        """Compute discriminator loss with LeCAM regularization"""
        real_pred, real_features = self.discriminator(real_images)
        fake_pred, _ = self.discriminator(fake_images.detach())
        
        # Hinge loss
        d_loss = hinge_d_loss(real_pred, fake_pred)
        
        # Add LeCAM regularization
        lecam_reg = lecam_loss(real_pred, fake_pred)
        
        return d_loss + lecam_reg, real_features

    def generator_loss(self, fake_images, real_features=None):
        """Compute generator loss with optional feature matching"""
        fake_pred, fake_features = self.discriminator(fake_images)
        g_loss = -torch.mean(fake_pred)
        
        if real_features is not None:
            fm_loss = self.feature_matching_loss(real_features, fake_features)
            g_loss = g_loss + fm_loss
            
        return g_loss


class VQGAN_Plus(nn.Module):
    """Complete VQGAN+ model with RVQ quantization"""
    def __init__(self, in_channels=3, hidden_channels=128, num_downsamples=4, 
                 vq_num_embeddings=1024, internal_dim=256, codebook_levels=4, 
                 vq_embedding_dim=8, commitment_weight=0.25, use_checkpoint=True):
        super().__init__()
        
        # Import RVQ from your existing code
        from vector_quantize_pytorch import ResidualVQ
        
        self.num_downsamples = num_downsamples
        self.use_checkpoint = use_checkpoint
        self.codebook_levels = codebook_levels
        
        # Calculate channel multipliers based on num_downsamples
        # Default: [1,1,2,2,4] for 4 downsamples, adjust as needed
        if num_downsamples == 3:
            channel_multipliers = [1,2,4]
        elif num_downsamples == 4:
            channel_multipliers = [1,1,2,4]
        elif num_downsamples == 5:
            channel_multipliers = [1,1,2,2,4]
        else:
            # Fallback - create appropriate multipliers
            channel_multipliers = [1] + [2**min(i, 2) for i in range(num_downsamples-1)]
        
        # VQGAN+ Encoder
        self.encoder = VQGANPlusEncoder(
            in_channels=in_channels,
            base_channels=hidden_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=internal_dim,
            use_checkpoint=use_checkpoint
        )
        
        # Compression to VQ embedding dimension
        self.compress = nn.Sequential(
            nn.Conv2d(internal_dim, vq_embedding_dim, 1),
            nn.GroupNorm(gn_groups(8, vq_embedding_dim), vq_embedding_dim),
            nn.SiLU(),
            nn.Conv2d(vq_embedding_dim, vq_embedding_dim, 3, padding=1)
        )
        
        # RVQ quantizer (matching your existing setup)
        self.vq = ResidualVQ(
            dim=vq_embedding_dim,
            codebook_size=vq_num_embeddings,
            decay=0.95,
            commitment_weight=commitment_weight,
            num_quantizers=codebook_levels,
            kmeans_init=True,
            kmeans_iters=15,
            threshold_ema_dead_code=2,
            rotation_trick=True,
            orthogonal_reg_weight=0.2
        )
        
        # VQGAN+ Decoder
        self.decoder = VQGANPlusDecoder(
            out_channels=in_channels,
            base_channels=hidden_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=vq_embedding_dim,
            use_checkpoint=use_checkpoint
        )
        
        self.info = None

    def encode(self, x, debug=False):
        """Encode input to latent space"""
        
        z = self.encoder(x)
        z = self.compress(z)
        
        return z
    
    def decode(self, z_q, noise_strength=0.0):
        """Decode quantized latents to images"""
        return self.decoder(z_q)

    @torch.no_grad()
    def calc_distance_stats(self, z_compressed_flat, z_q):
        """Diagnostic: Calculate distances between encoder outputs and codebook vectors"""
        distances = torch.norm(z_compressed_flat.unsqueeze(1) -
                             self.vq.codebooks[0], dim=-1)
        return {
            'codebook_mean_dist': distances.mean().item(),
            'codebook_max_dist': distances.max().item()
        }
        
    def forward(self, x, noise_strength=None, minval=0, get_stats=False):
        """Full forward pass: encode -> quantize -> decode"""
        z = self.encode(x)
        
        if noise_strength is None:
            noise_strength = 0.05 if self.training else 0.0

        if self.info is None: 
            self.info = z.shape
          
        # Prepare for VQ: permute and reshape
        z = z.permute(0, 2, 3, 1)
        z = z.reshape(-1, z.shape[-1])
              
        z_q, indices, commit_loss = self.vq(z)
        # Undo the reshape and permute
        z_q = z_q.view(x.shape[0], x.shape[2] // (2 ** self.num_downsamples), 
                    x.shape[3] // (2 ** self.num_downsamples), -1)
        z_q = z_q.permute(0, 3, 1, 2)
        x_recon = self.decode(z_q, noise_strength=noise_strength)
        assert x_recon.shape == x.shape, f"Shape mismatch: {x_recon.shape} (recon) vs. {x.shape} (original)"

        if get_stats:
            stats = self.calc_distance_stats(z, z_q)
            return x_recon, commit_loss.mean(), stats
        else:
            return x_recon, commit_loss.mean()


# Factory functions for easy instantiation
def create_vqgan_plus_encoder(in_channels=3, latent_channels=256, use_checkpoint=True):
    """Create VQGAN+ encoder with default settings"""
    return VQGANPlusEncoder(
        in_channels=in_channels,
        base_channels=128,
        channel_multipliers=[1,1,2,2,4],
        latent_channels=latent_channels,
        use_checkpoint=use_checkpoint
    )


def create_vqgan_plus_decoder(out_channels=3, latent_channels=256, use_checkpoint=True):
    """Create VQGAN+ decoder with default settings"""
    return VQGANPlusDecoder(
        out_channels=out_channels,
        base_channels=128,
        channel_multipliers=[1,1,2,2,4],
        latent_channels=latent_channels,
        use_checkpoint=use_checkpoint
    )


def create_vqgan_plus_discriminator(in_channels=3, use_checkpoint=True):
    """Create VQGAN+ discriminator with default settings"""
    return VQGANPlusDiscriminator(
        in_channels=in_channels,
        base_channels=128,
        n_layers=3,
        use_checkpoint=use_checkpoint
    )




class VQGANPlusPatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, n_layers=3, use_checkpoint=False):
        super().__init__()

        # Use 3x3 instead of 4x4 convolutions
        layers = [
            spectral_norm(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        current_channels = hidden_channels
        for i in range(n_layers):
            next_channels = min(hidden_channels * (2 ** (i+1)), 512)

            # Add Gaussian blur before downsampling (except last layer)
            if i < n_layers-1:
                # Precomputed 4x4 Gaussian blur kernel
                layers.append(GaussianBlur())
                layers.append(VQGANPlusDiscrResBlock(current_channels, next_channels,
                                          stride=2, use_checkpoint=use_checkpoint))
            else:
                layers.append(VQGANPlusDiscrResBlock(current_channels, next_channels,
                                          stride=1, use_checkpoint=use_checkpoint))
            current_channels = next_channels

        # Final prediction layer - 3x3 instead of 4x4
        layers.append(
            spectral_norm(nn.Conv2d(current_channels, 1, kernel_size=3, stride=1, padding=1))
        )

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        features = []
        for layer in self.main:
            x = layer(x)
            if isinstance(layer, (nn.LeakyReLU, VQGANPlusDiscrResBlock)):
                features.append(x)
        return x, features


# Add Gaussian blur layer
class GaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        # 4x4 Gaussian kernel
        kernel = torch.tensor([[1., 2., 2., 1.],
                              [2., 4., 4., 2.],
                              [2., 4., 4., 2.],
                              [1., 2., 2., 1.]]) / 36.0
        self.register_buffer('kernel', kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # Apply Gaussian blur with stride 2 for downsampling
        return F.conv2d(x, self.kernel.repeat(x.shape[1], 1, 1, 1),
                       stride=2, padding=1, groups=x.shape[1])
