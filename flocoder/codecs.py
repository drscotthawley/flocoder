# flocoder/models/vqvae.py
# Note that a VQGAN model is a VQVAE model that is trained with an adversarial loss.
# We will sometimes use VQGAN and VQVAE interchangeably. 
# The architecture of the two models is the same, and the moniker "VQVAE" is preferred. 


import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import warnings
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True") # annoying warnings when grad checkpointing. it's fine, really
#from flash_attn import flash_attn_func
import os
import math

from .general import ldcfg

# Optional: NATTEN
# Execution is noticeably faster with NATTEN (at least on GPU)
# Note that a model trained on NATTEN requires NATTEN for inference  
# TODO: add some code that can convert a NATTEN model to a non-NATTEN model  
try:
    import natten
    print("Using NATTEN version ", natten.__version__)
except ImportError:
    warnings.warn("Warning: NATTEN not found. Running without. You might want to install it.")
    natten = None



def gn_groups(proposed, channels, max_groups=8, verbose=True):
    """Lil utility so that when you change input channels, GroupNorm doens't bring everything to a halt"""
    if channels % proposed == 0: return proposed # we're good.
    for candidate in range(proposed, channels): # if there's a mismatch Find nearest divisor >= proposed
        if channels % candidate == 0:
            if verbose: print(f"gn_groups: rounded proposed={proposed} up to {candidate} for channels={channels}")
            return candidate

    if verbose: print(f"gn_groups: no valid divisor found for proposed={proposed}, channels={channels}, falling back to LayerNorm behavior")
    return 1


class Normalize(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-6, affine=True):
        super().__init__()
        num_groups = gn_groups(num_groups, num_channels)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)
    def forward(self, x):
        return self.norm(x)
    
class AttnBlock(nn.Module):
    # Use the AttnBlock from the official VQGAN code. aka "non-local block"
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_



class NATTENBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, num_heads=8, init_scale=0.02):
        super().__init__()
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.head_dim = dim // num_heads
        self.scaling = (self.head_dim ** -0.5) * 0.5
        
        # Replace LayerNorm with GroupNorm
        self.norm = nn.GroupNorm(num_groups=gn_groups(8,dim), num_channels=dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initialize with smaller weights
        nn.init.normal_(self.qkv.weight, std=init_scale)
        nn.init.normal_(self.proj.weight, std=init_scale)
        
        if natten is None:
            raise ImportError("Please install NATTEN: pip install natten")
            
    def _forward(self, x):
        B, C, H, W = x.shape
        identity = x
        
        # Apply GroupNorm (works in channel-first format)
        x = self.norm(x)
        
        # Only permute once for the linear layers
        x = x.permute(0, 2, 3, 1)  # B H W C
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, H, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(3, 0, 4, 1, 2, 5)  # 3 B heads H W dim
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = natten.functional.na2d_qk(q, k, self.kernel_size)
        attn = attn.softmax(dim=-1)
        x = natten.functional.na2d_av(attn, v, self.kernel_size)
            
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)  # Back to B C H W
        return identity + (x * self.gamma)  # Scaled attention plus identity
        
    def forward(self, x):
        if x.requires_grad:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)




class EncDecResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False, attention='natten', dropout_rate=0.1, dropout2d_rate=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(gn_groups(8,out_channels), out_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(gn_groups(8,out_channels), out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if dropout2d_rate is None: dropout2d_rate = max(0.05, dropout_rate-0.05)
        self.dropout2d = nn.Dropout2d(dropout2d_rate)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(gn_groups(8, out_channels), out_channels)
            )

        self.use_checkpoint = use_checkpoint
        if attention == 'natten' and natten:
            self.attn = NATTENBlock(out_channels)
        elif attention == 'full':  
            self.attn = AttnBlock(out_channels)
        else:
            self.attn = None


    def _forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.silu(out)
        out = self.dropout(self.dropout2d(out))

        if self.attn:
            out = self.attn(out) 

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)
        out = self.dropout(self.dropout2d(out))
        return out

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class NoiseInjection(nn.Module):
    def __init__(self, channels, noise_strength=0.0):
        super().__init__()
        self.to_noise_scale = nn.Conv2d(channels, channels, 1)  # Generate per-pixel noise scales
        self.to_noise_bias = nn.Conv2d(channels, channels, 1)   # Generate per-pixel noise biases
        self.noise_strength = noise_strength


        # Initialize to near-zero
        nn.init.zeros_(self.to_noise_scale.weight)
        nn.init.zeros_(self.to_noise_bias.weight)
        
    def forward(self, x, noise_strength=None): # default noise_strength is that this is a no-op
        if noise_strength is None: noise_strength = self.noise_strength
        if noise_strength == 0.0: # note there is no restriction on whether we're training or not
            return x
            
        # Generate unique noise for this forward pass
        noise = torch.randn_like(x)
        
        # Generate spatially-varying scales and biases
        noise_scale = self.to_noise_scale(x)
        noise_bias = self.to_noise_bias(x)
        
        return x + noise_strength * (noise * noise_scale + noise_bias)
    


class Decoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, num_downsamples=3, 
                 internal_dim=256, vq_embedding_dim=4, inject_noise=True, use_checkpoint=False,):
        super().__init__()
        
        decoder_layers = []
        noise_strength = 0.0# leave this off for now and just try Dropout #  0.01 if inject_noise else 0.0

        # projection
        current_channels = hidden_channels * (2 ** (num_downsamples - 1))
        next_layers = [
            SpatialNonLocalAttention(vq_embedding_dim),
            nn.Conv2d(vq_embedding_dim, internal_dim, 1),  # Expand to internal_dim
            nn.GroupNorm(gn_groups(vq_embedding_dim, internal_dim), internal_dim),
            nn.SiLU(),
            nn.Conv2d(internal_dim, current_channels, 1),  # Then to current_channels
            NoiseInjection(current_channels, noise_strength=0.01), # try to inject a little generative "information"
            EncDecResidualBlock(current_channels, current_channels, 
                            use_checkpoint=use_checkpoint, attention='full',  dropout_rate=0.05)
        ]
        decoder_layers.extend(next_layers)

        # Upsampling blocks
        for i in range(num_downsamples - 1, -1, -1):
            out_channels = hidden_channels * (2 ** max(0, i - 1))
            if i == 0:
                out_channels = hidden_channels
            if i > num_downsamples-2:
                attention, mode = 'natten','bicubic'
            else:
                attention, mode = None, 'bilinear'

            block = [
                #nn.Upsample(scale_factor=2, mode=mode, align_corners=False), use PixelShuffle instead of Upsample
                nn.Conv2d(current_channels, current_channels * 4, 3, padding=1),
                nn.SiLU(),
                nn.PixelShuffle(2), 
                NoiseInjection(current_channels),#  noise_strength=0.002),
                EncDecResidualBlock(current_channels, out_channels, 
                                  use_checkpoint=use_checkpoint, attention=attention, dropout_rate=0, dropout2d_rate=0.1),
                NoiseInjection(out_channels),# noise_strength=0.001),
                EncDecResidualBlock(out_channels, out_channels, 
                                  use_checkpoint=use_checkpoint, attention=None, dropout_rate=0.0, dropout2d_rate=0.0)
            ]
            decoder_layers.extend(block)
            current_channels = out_channels

        # Final layers
        final_layers = [
            NoiseInjection(current_channels),# noise_strength=0.0),
            nn.Conv2d(current_channels, 64, 3, padding=1),
            nn.SiLU(),
            NoiseInjection(64),#  noise_strength=0.0),
            nn.Conv2d(64, in_channels, 3, padding=1),
        ]
        decoder_layers.extend(final_layers)
        
        self.layers = nn.ModuleList(decoder_layers)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, noise_strength=0.0):
        for layer in self.layers:
            if isinstance(layer, NoiseInjection): # default noise_strength means this is a no-op
                x = layer(x, noise_strength)
            elif self.use_checkpoint and self.training and isinstance(layer, EncDecResidualBlock):
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x



class DimensionReduction(nn.Module):
    "added this before RVQ. It's effectively the very end of the encoder"
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(in_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, bottleneck_dim)
        )
        self.expand = nn.Sequential(
            nn.Linear(bottleneck_dim, in_dim//2),
            nn.ReLU(),
            nn.Linear(in_dim//2, in_dim)
        )


class SpatialNonLocalAttention(nn.Module):
   def __init__(self, channels, h=16, w=16, reduction_factor=2):
       super().__init__()
       self.channels, self.h, self.w = channels, h, w
       reduced_dim = max(1, channels // reduction_factor)
       import math; self.scale = math.log(10000.0)
       
       # Projections with compact initialization
       self.q_proj, self.k_proj = nn.Conv2d(channels, reduced_dim, 1), nn.Conv2d(channels, reduced_dim, 1)
       self.v_proj, self.out_proj = nn.Conv2d(channels, channels, 1), nn.Conv2d(channels, channels, 1)
       for proj in [self.q_proj, self.k_proj, self.v_proj]: nn.init.xavier_uniform_(proj.weight, gain=0.01)
       nn.init.zeros_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
       
   def _apply_rope(self, x):
       b, hw, c = x.shape; device = x.device
       if c % 2 != 0: x = F.pad(x, (0, 1)); c = x.shape[-1]  # Handle odd dimensions
       
       # Generate position encodings
       pos = torch.arange(hw, device=device).unsqueeze(1)
       dim_t = torch.arange(0, c//2, device=device)
       inv_freq = torch.exp(-dim_t * self.scale / (c//2))
       pos_enc = pos * inv_freq.unsqueeze(0)
       pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1).unsqueeze(0).expand(b, -1, -1)
       
       # Apply rotation efficiently
       x_even, x_odd = x[..., 0::2], x[..., 1::2]
       pos_sin, pos_cos = pos_enc[..., :c//2], pos_enc[..., c//2:]
       x_out = torch.empty_like(x)
       x_out[..., 0::2] = x_even * pos_cos - x_odd * pos_sin
       x_out[..., 1::2] = x_odd * pos_cos + x_even * pos_sin
       return x_out
   
   def _forward(self, x):
       b, c, h, w = x.shape
       
       # Project, reshape, and apply RoPE
       q = self._apply_rope(self.q_proj(x).reshape(b, -1, h*w).permute(0, 2, 1))  # [b, hw, c']
       k = self._apply_rope(self.k_proj(x).reshape(b, -1, h*w).permute(0, 2, 1))  # [b, hw, c']
       v = self.v_proj(x).reshape(b, c, h*w).permute(0, 2, 1)  # [b, hw, c]
       
       # Compute attention and apply to values
       attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * (q.size(-1) ** -0.5), dim=-1)
       out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, h, w)
       return x + self.out_proj(out)  # Residual connection
   
   def forward(self, x):
       return checkpoint(self._forward, x, use_reentrant=False) if x.requires_grad and self.training else self._forward(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, num_downsamples=3, 
                 vq_num_embeddings=512, internal_dim=256, codebook_levels=3, 
                 vq_embedding_dim=4, commitment_weight=0.25,
                 use_checkpoint=False, no_natten=False):
        super().__init__()
        global natten 

        if no_natten:
            natten = None
        
        self.num_downsamples = num_downsamples
        self.use_checkpoint = use_checkpoint
        self.codebook_levels = codebook_levels
        
        # Encoder with checkpointing
        encoder_layers = []
        in_channels_current = in_channels
        for i in range(num_downsamples):
            out_channels = hidden_channels * (2 ** i)
            if i >= num_downsamples-2: 
                attention = 'natten'
            else: 
                attention = None
            encoder_layers.append(EncDecResidualBlock(in_channels_current, out_channels, 
                                stride=2, use_checkpoint=use_checkpoint, attention=attention, dropout_rate=0.05))
            
            encoder_layers.append(EncDecResidualBlock(out_channels, out_channels, 
                                stride=1, use_checkpoint=use_checkpoint, attention=attention, dropout_rate=0.15))
            in_channels_current = out_channels
                
        encoder_layers.append(EncDecResidualBlock(in_channels_current, internal_dim, 
                            stride=1, use_checkpoint=use_checkpoint, attention=attention, dropout_rate=0.15))
        encoder_layers.append(nn.Conv2d(internal_dim, internal_dim, 1)) # final conv2d undoes swish at end of EncDecResidualBlock
        
        # added this extra set of compression layers
        compress_layers = [
            nn.Conv2d(internal_dim, vq_embedding_dim, 1),  # Compress to 8 dimensions
            nn.GroupNorm(gn_groups(2, vq_embedding_dim), vq_embedding_dim),
            nn.SiLU(),
            nn.Conv2d(vq_embedding_dim, vq_embedding_dim, 3, padding=1)]
        encoder_layers.extend(compress_layers)
        encoder_layers.append(SpatialNonLocalAttention(vq_embedding_dim)) 
        self.encoder = nn.Sequential(*encoder_layers)
        self.info = None


        # Vector Quantizer
        # self.vq = VectorQuantize( # non-residual VQ. 
        #     dim=internal_dim,
        #     codebook_size=vq_num_embeddings,
        #     decay=0.95,
        #     commitment_weight=1.0,
        #     kmeans_init=True,
        #     threshold_ema_dead_code=2
        # )
        self.vq = ResidualVQ(
            dim = vq_embedding_dim,
            codebook_size = vq_num_embeddings,
            decay=0.95,
            commitment_weight=commitment_weight,
            num_quantizers = codebook_levels,
            kmeans_init = True,
            kmeans_iters = 15, 
            threshold_ema_dead_code = 2,
            rotation_trick = True,
            orthogonal_reg_weight=0.2,
            #implicit_neural_codebook=True, # trying it :shrug: IDK
        )
        #self.vq = initialize_vq_with_normal_codebook(self.vq, level=0) # didn't help. not including

        print("vq_embedding_dim, internal_dim = ",vq_embedding_dim, internal_dim)
        
        uncompress_layers = []
        self.decoder = Decoder( in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_downsamples=num_downsamples,
            internal_dim=internal_dim,
            vq_embedding_dim=vq_embedding_dim,
            use_checkpoint=use_checkpoint,
        )
        #----- end of init


    def encode(self, x, debug=False):
        if debug: print("\n vqvae: starting self.encode", flush=True)
        if self.use_checkpoint and self.training:
            return checkpoint(self.encoder, x, use_reentrant=False)
        if debug: print("vqvae: calling self.encoder", flush=True)
        z = self.encoder(x)
        if debug: print("encode: z.shape = ", z.shape, flush=True)
        return z
    
    def decode(self, z_q, noise_strength=0.0):
        #z_q = self.expand(z_q)  # undoes the "compress" from self.encode
        return self.decoder(z_q, noise_strength=noise_strength)

    @torch.no_grad()
    def calc_distance_stats(self, z_compressed_flat, z_q):
        """Diagnostic: Calculate distances between encoder outputs and codebook vectors"""
        distances = torch.norm(z_compressed_flat.unsqueeze(1) -
                             self.vq.codebooks[0], dim=-1)  # For first codebook
        return {
            'codebook_mean_dist': distances.mean().item(),
            'codebook_max_dist': distances.max().item()
        }
        
    def forward(self, x, noise_strength=None, minval=0, get_stats=False):
        z = self.encode(x)
        if noise_strength is None:
            noise_strength = 0.05 if self.training else 0.0

        if self.info is None: 
            self.info = z.shape
            print("\nINFO: .encode output z.shape",self.info,"\n",flush=True)
          
        # prepare for VQ: permute and reshape
        z = z.permute(0, 2, 3, 1)
        z = z.reshape(-1, z.shape[-1])
              
        z_q, indices, commit_loss = self.vq(z)  # Apply vector quantization

        # undo the reshape and permute
        z_q = z_q.view(x.shape[0], x.shape[2] // (2 ** self.num_downsamples), 
                    x.shape[3] // (2 ** self.num_downsamples), -1)
        z_q = z_q.permute(0, 3, 1, 2)
        
        x_recon = self.decode(z_q, noise_strength=noise_strength)

        if get_stats: # for some diagnostics
            stats = self.calc_distance_stats(z, z_q)
            return x_recon, commit_loss.mean(), stats
        else:
            return x_recon, commit_loss.mean()



class SimpleResizeAE(nn.Module):
    """Just resizes tensors using bilinear interpolation. 
    For testing/exploration - reconstructions will be blocky/blurry."""
    def __init__(self, 
                 latent_shape=(4,16,16), # tuple for latent dimensions
                 mode='bicubic',         # interpolation/resampling mode
                 ): 
        super().__init__()
        self.latent_shape  = latent_shape
        self.orig_shape    = None
        self.mode          = mode
        
    def encode(self, x):
        """Resize input to latent_size using interpolation. 
        any extra latent channels are just copies of the mean"""
        self.orig_shape = x.shape[1:]   # save orig shape for later decode
        if self.latent_shape is None: 
            return x   # No-Op! 
        c, h, w = self.latent_shape
        small = F.interpolate(x, size=(h, w), mode=self.mode, align_corners=False)
        if c == x.shape[-3]:  # no change in channels
            return small
        mean_channel = torch.mean(small, dim=1, keepdim=True)
        return torch.cat([small, mean_channel.repeat(1, c - x.shape[-3], 1, 1)], dim=1)


    def decode(self, z, orig_shape=None, noise_strength=0.0): # noise_strength unused, for API compatibility
        """Resize latent back to original dimensions."""
        target_shape = orig_shape if orig_shape is not None else self.orig_shape
        if target_shape is None or target_shape==self.latent_shape: return z    # No-op
        h, w = (target_shape[-2], target_shape[-1]) 
        # Only use the first 3 channels for decoding; any extras are 
        return F.interpolate(z[:,:3], size=(h, w), mode=self.mode, align_corners=False)

    
    def forward(self, x, noise_strength=0.0, minval=0, get_stats=False):
        """Encode and decode in one step, auto-storing original size."""
        z = self.encode(x)
        recon = self.decode(z)
        if get_stats: return recon, 0.0, {'codebook_mean_dist': 0.0, 'codebook_max_dist': 0.0}
        return recon, 0.0




class SD_VAE_Wrapper(nn.Module):
    """Wrapper for Stable Diffusion VAE to make it compatible with our standard interface."""
    def __init__(self, pretrained_model_name="stabilityai/sd-vae-ft-mse"):
        super().__init__()
        from diffusers.models import AutoencoderKL  # lazy import, only use if needed
        self.vae = AutoencoderKL.from_pretrained(pretrained_model_name).eval()

    def encode(self, x):
        """Encode images to latent space."""
        # Dynamically scale to [-1,1] range regardless of input range
        x_min, x_max = x.min(), x.max()
        self.last_min, self.last_max = x_min, x_max  # Store for decode scaling
        if x_min != -1 or x_max != 1:
            x = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        return self.vae.encode(x).latent_dist.sample()

    def decode(self, z, orig_size=None, noise_strength=0.0):
        """Decode latents to images, automatically handling the .sample() call."""
        decoded = self.vae.decode(z).sample
        if hasattr(self, 'last_min') and hasattr(self, 'last_max'):
            return 0.5 * (decoded + 1) * (self.last_max - self.last_min) + self.last_min
        return decoded

    def forward(self, x, noise_strength=0.0, minval=0, get_stats=False):
        """Forward pass through encoder and decoder."""
        self.last_min, self.last_max = x.min(), x.max()  # Store for decode scaling
        z = self.encode(x)
        recon = self.decode(z)
        if get_stats: return recon, 0.0, {'codebook_mean_dist': 0.0, 'codebook_max_dist': 0.0}
        return recon, 0.0




def setup_codec(config, device, no_natten=False, load_checkpoint=True, eval=True):
    """Load the appropriate codec model based on configuration."""
    import os
    import torch
    from types import SimpleNamespace

    if config.codec.choice == "resize":
        print("Using SimpleResizeAE")
        codec = SimpleResizeAE(latent_shape=config.codec.get('latent_shape', (4, 16, 16))).eval().to(device)

    elif config.codec.choice == "sd":
        print("Loading SD VAE via SD_VAE_Wrapper")
        codec = SD_VAE_Wrapper(
            pretrained_model_name="stabilityai/sd-vae-ft-mse"
        ).eval().to(device)

        if hasattr(config, 'image_size') and config.image_size % 8 != 0:
            print(f"Warning: SD VAE works best with image sizes divisible by 8. Current size: {config.image_size}")
    elif config.codec.choice == "vqgan_plus":
        print("Loading VQGAN+ model")
        from .vqgan_plus import VQGAN_Plus
        codec = VQGAN_Plus(
            in_channels=ldcfg(config, 'in_channels', 3),
            hidden_channels=ldcfg(config, 'hidden_channels', 128),  # Note: default 128 for VQGAN+
            num_downsamples=ldcfg(config, 'num_downsamples', 4),   # Note: default 4 for VQGAN+
            #internal_dim=ldcfg(config, 'internal_dim', 256),
            internal_dim=ldcfg(config, 'internal_dim', ldcfg(config, 'vq_embedding_dim', 8), supply_defaults=True), # fall back to vq_embedding_dim
            vq_embedding_dim=ldcfg(config, 'vq_embedding_dim', 8), # Note: default 8 for VQGAN+
            codebook_levels=ldcfg(config, 'codebook_levels', 4),
            vq_num_embeddings=ldcfg(config, 'vq_num_embeddings', 1024), # Note: default 1024 for VQGAN+
            commitment_weight=ldcfg(config, 'commitment_weight', 0.25),
            use_checkpoint=not config.get('no_grad_ckpt', False),
        ).to(device)
    else:
        # Original VQVAE path
        # Get model parameters, handle nested config structures
        # Create the VQVAE model
        print("Loading VQVAE model")
        codec = VQVAE(
            in_channels=ldcfg(config, 'in_channels', 3),
            hidden_channels=ldcfg(config, 'hidden_channels', 256),
            num_downsamples=ldcfg(config, 'num_downsamples', 3),
            internal_dim=ldcfg(config, 'internal_dim', 256),
            vq_embedding_dim=ldcfg(config, 'vq_embedding_dim', 4),
            codebook_levels=ldcfg(config, 'codebook_levels', 4),
            vq_num_embeddings=ldcfg(config, 'vq_num_embeddings', 512),
            commitment_weight=ldcfg(config, 'commitment_weight', 0.5),
            use_checkpoint=not config.get('no_grad_ckpt', False),
            no_natten=no_natten,
        ).to(device)

        if load_checkpoint:
            # Figure out checkpoint path from different possible config structures
            if hasattr(config, 'vqgan_checkpoint'):
                checkpoint_path = config.vqgan_checkpoint
            elif hasattr(config, 'codec') and hasattr(config.codec, 'checkpoint'):
                checkpoint_path = config.codec.checkpoint
            else:
                raise ValueError("Could not find codec checkpoint path in config")
    
            if checkpoint_path.lower() != "sd" and not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Codec checkpoint file {checkpoint_path} not found.")
    
            print(f"Loading codec checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            codec.load_state_dict(checkpoint['model_state_dict'])

    if eval: codec=code.eval()
    print("Codec model ready")
    return codec
