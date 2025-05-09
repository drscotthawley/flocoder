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



class Normalize(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-6, affine=True):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, 
                                num_channels=num_channels, 
                                eps=eps, 
                                affine=affine)
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
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim)
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
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False, attention='natten'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(8, out_channels)
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

        if self.attn:
            out = self.attn(out) 

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.silu(out)
        return out

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.to_noise_scale = nn.Conv2d(channels, channels, 1)  # Generate per-pixel noise scales
        self.to_noise_bias = nn.Conv2d(channels, channels, 1)   # Generate per-pixel noise biases
        
        # Initialize to near-zero
        nn.init.zeros_(self.to_noise_scale.weight)
        nn.init.zeros_(self.to_noise_bias.weight)
        
    def forward(self, x, noise_strength=0.0): # default noise_strength is that this is a no-op
        if noise_strength == 0.0:
            return x
            
        # Generate unique noise for this forward pass
        noise = torch.randn_like(x)
        
        # Generate spatially-varying scales and biases
        noise_scale = self.to_noise_scale(x)
        noise_bias = self.to_noise_bias(x)
        
        return x + noise_strength * (noise * noise_scale + noise_bias)
    


class Decoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, num_downsamples=3, 
                 vq_embedding_dim=256, inject_noise=True, use_checkpoint=False,
                 init_layers = []):
        super().__init__()
        
        decoder_layers = init_layers

        # projection
        current_channels = hidden_channels * (2 ** (num_downsamples - 1))
        next_layers = [
            nn.Conv2d(vq_embedding_dim, current_channels, 1),
            NoiseInjection(current_channels), # note NoiseInjection defaults to a no-op; only here for playing with later
            EncDecResidualBlock(current_channels, current_channels, 
                              use_checkpoint=use_checkpoint, attention='full')
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
                nn.Upsample(scale_factor=2, mode=mode, align_corners=False),
                NoiseInjection(current_channels),
                EncDecResidualBlock(current_channels, out_channels, 
                                  use_checkpoint=use_checkpoint, attention=attention),
                NoiseInjection(out_channels),
                EncDecResidualBlock(out_channels, out_channels, 
                                  use_checkpoint=use_checkpoint, attention=None)
            ]
            decoder_layers.extend(block)
            current_channels = out_channels

        # Final layers
        final_layers = [
            NoiseInjection(current_channels),
            nn.Conv2d(current_channels, 64, 3, padding=1),
            nn.SiLU(),
            NoiseInjection(64),
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



class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, num_downsamples=3, 
                 vq_num_embeddings=512, vq_embedding_dim=128, codebook_levels=3, 
                 compressed_dim=4, commitment_weight=0.25,
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
                                stride=2, use_checkpoint=use_checkpoint, attention=attention))
            encoder_layers.append(EncDecResidualBlock(out_channels, out_channels, 
                                stride=1, use_checkpoint=use_checkpoint, attention=attention))
            in_channels_current = out_channels
                
        encoder_layers.append(EncDecResidualBlock(in_channels_current, vq_embedding_dim, 
                            stride=1, use_checkpoint=use_checkpoint, attention=attention))
        encoder_layers.append(nn.Conv2d(vq_embedding_dim, vq_embedding_dim, 1)) # final conv2d undoes swish at end of EncDecResidualBlock
        
        # added this extra set of compression layers
        compress_layers = [
            nn.Conv2d(vq_embedding_dim, compressed_dim, 1),  # Compress to 8 dimensions
            nn.GroupNorm(2, compressed_dim),
            nn.SiLU(),
            nn.Conv2d(compressed_dim, compressed_dim, 3, padding=1)]
        encoder_layers.extend(compress_layers)
        self.encoder = nn.Sequential(*encoder_layers)
        self.info = None


        # Vector Quantizer
        # self.vq = VectorQuantize( # non-residual VQ. 
        #     dim=vq_embedding_dim,
        #     codebook_size=vq_num_embeddings,
        #     decay=0.95,
        #     commitment_weight=1.0,
        #     kmeans_init=True,
        #     threshold_ema_dead_code=2
        # )
        self.vq = ResidualVQ(
            dim = compressed_dim,
            codebook_size = vq_num_embeddings,
            decay=0.95,
            commitment_weight=commitment_weight,
            num_quantizers = codebook_levels,
            kmeans_init = True,
            kmeans_iters = 15, 
            threshold_ema_dead_code = 2,
            rotation_trick = True,
            orthogonal_reg_weight=0.1,
            #implicit_neural_codebook=True, # trying it :shrug: IDK
        )
        #self.vq = initialize_vq_with_normal_codebook(self.vq, level=0) # didn't help. not including

        print("compressed_dim, vq_embedding_dim = ",compressed_dim, vq_embedding_dim)
        
        # self.expand = nn.Sequential(
        #     nn.Conv2d(compressed_dim, vq_embedding_dim, 1),  # Expand back to original dimension
        #     nn.GroupNorm(compressed_dim, vq_embedding_dim),
        #     nn.SiLU(),
        # )
        uncompress_layers = [
            nn.Conv2d(compressed_dim, vq_embedding_dim, 1),  # Expand back to original dimension
            nn.GroupNorm(compressed_dim, vq_embedding_dim),
            nn.SiLU(),
        ]
        self.decoder = Decoder( in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_downsamples=num_downsamples,
            vq_embedding_dim=vq_embedding_dim,
            use_checkpoint=use_checkpoint,
            init_layers=uncompress_layers,
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
        
    def forward(self, x, noise_strength=0.0, minval=0, get_stats=False):
        z = self.encode(x)

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
