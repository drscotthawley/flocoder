# from TadaoYamoaka's CIFAR10 flow matching code  
import math
from functools import partial
from typing import List, Optional

import torch
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

import gc
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum, nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


def Downsample(dim, dim_out):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1),
    )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2),
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        time_emb = self.mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        condition=False,
        n_classes=10,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint=use_checkpoint

        # determine dimensions
        self.channels = channels
        self.condition = condition
        input_channels = channels

        init_dim = dim
        self.init_conv = nn.Conv2d(
            input_channels, init_dim, 1, padding=0
        )  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.condition:
            self.cond_mlp = nn.Sequential(
                nn.Embedding(n_classes, time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.out_dim = channels

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def _forward(self, x, time, cond=None):
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        if self.condition and cond is not None:
            t += self.cond_mlp(cond)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

    def forward(self, x, time, cond=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, time, cond, use_reentrant=False)
        return self._forward(x, time, cond)


class UnetStage(nn.Module):
    """
    Single U-Net stage that can accept additional skip connections from a previous stage
    """
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        time_dim=None,  # Allow time_dim to be passed in
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        input_channels = channels

        init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if time_dim is None:
            time_dim = dim * 4
        self.time_dim = time_dim

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Track the number of channels at each resolution for skip connections
        self.down_channels = []

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                    ]
                )
            )
            self.down_channels.append(dim_in)

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        self.down_channels.append(mid_dim)  # Add middle block channels

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_out, time_emb_dim=time_dim),  # *2 for skip connection
                        block_klass(dim_out * 2, dim_out, time_emb_dim=time_dim),  # *2 for skip connection
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, t, prev_features: Optional[List[torch.Tensor]] = None):
        """
        Args:
            x: Input tensor
            t: Time embedding
            prev_features: Optional list of features from previous stage's upward path
        Returns:
            output: Final output tensor
            features: List of features from current stage for next stage
        """
        x = self.init_conv(x)
        r = x.clone()

        features = []

        # Down path
        for i, (block1, block2, attn, downsample) in enumerate(self.downs):
            # If we have features from previous stage, add them
            if prev_features is not None and i < len(prev_features):
                x = torch.cat([x, prev_features[i]], dim=1)
                # Adjust channels with a 1x1 conv if needed
                if x.shape[1] != self.down_channels[i]:
                    x = nn.Conv2d(x.shape[1], self.down_channels[i], 1).to(x.device)(x)

            x = block1(x, t)
            features.append(x)

            x = block2(x, t)
            x = attn(x)
            features.append(x)

            x = downsample(x)

        # Middle
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        features.append(x)

        # Up path
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, features.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, features.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        return x, features

class MultiStageUnet(nn.Module):
    """
    Multiple U-Net stages connected with cross-stage skip connections
    """
    def __init__(
        self,
        dim,
        num_stages=2,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        condition=False,
        n_classes=10,
    ):
        super().__init__()

        self.num_stages = num_stages
        self.condition = condition

        # Time embeddings (shared across stages)
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.condition:
            self.cond_mlp = nn.Sequential(
                nn.Embedding(n_classes, time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # Create multiple UNet stages
        self.stages = nn.ModuleList([
            UnetStage(
                dim=dim,
                dim_mults=dim_mults,
                channels=channels,
                resnet_block_groups=resnet_block_groups,
                time_dim=time_dim
            )
            for _ in range(num_stages)
        ])

    def forward(self, x, time, cond=None):
        # Time embedding
        t = self.time_mlp(time)
        if self.condition and cond is not None:
            t = t + self.cond_mlp(cond)

        prev_features = None
        for stage in self.stages:
            x, prev_features = stage(x, t, prev_features)

        return x





class MRUnet_Nasty(nn.Module):
    # MultiResolution Unet - concats a coarse copy of input into the front of the bottleneck,
    # and a hook at the end of the bottleneck to train against a (outside the scope of this) coarsened target
    # made by SHH with Claude ;-)  this includes downsampled versions of input
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        condition=False,
        n_classes=10,
        bottleneck_loss_weight=0.1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.bottleneck_loss_weight = bottleneck_loss_weight
        self.channels = channels

        # determine dimensions
        input_channels = channels
        init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # condition embeddings
        self.condition = condition
        if self.condition:
            self.cond_mlp = nn.Sequential(
                nn.Embedding(n_classes, time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        # Bottleneck start - for concatenation with downsampled input
        mid_dim = dims[-1]
        bottleneck_in_dim = mid_dim + channels  # Adding channels for downsampled input
        
        # Bottleneck start - handles the concatenation
        self.bottleneck_start = nn.Sequential(
            nn.Conv2d(bottleneck_in_dim, mid_dim, 3, padding=1),
            nn.GroupNorm(resnet_block_groups, mid_dim),
            nn.SiLU(),
        )

        # Middle blocks (equivalent to your mid_block1, mid_attn, mid_block2)
        # self.bottleneck = nn.Sequential(
        #     block_klass(mid_dim, mid_dim, time_emb_dim=time_dim),
        #     Residual(PreNorm(mid_dim, Attention(mid_dim))),
        #     block_klass(mid_dim, mid_dim + channels, time_emb_dim=time_dim),  # +channels for hook
        # )
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim + channels, time_emb_dim=time_dim)  # +channels for hook

        self.bn_size = None  # bottleneck image height & width, computed on the fly

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.out_dim = channels
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def downsampler(self, input_img, x=None, mode='bilinear'):
        """Resize input image to match the spatial dimensions of x"""
        if x is not None: self.bn_size = (x.shape[2], x.shape[3])
        assert self.bn_size is not None, "No memory of bottleneck size self.bn_size - call forward first or pass in x"
        return F.interpolate(input_img, size=self.bn_size, mode=mode, align_corners=False)


    def _forward(self, x, time, target=None, training=False, cond=None):
        losses = {}

        x0 = x.clone() 
        
        # Initialize
        x = self.init_conv(x)
        r = x.clone()

        # Time embedding
        t = self.time_mlp(time)
        if self.condition and cond is not None:
            t += self.cond_mlp(cond)

        # Downward path
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # Get downsampled input for bottleneck concat
        with torch.no_grad():
            downsampled_input = self.downsampler(x0, x)

        # Bottleneck with concatenation of explicitly coursened image
        bottleneck_input = torch.cat([x, downsampled_input], dim=1)
        x = self.bottleneck_start(bottleneck_input)
        
        # Process through bottleneck
        #x = self.bottleneck(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        self.bottleneck_target_hook = x[:, -self.channels:, :, :] # hook for extra loss computation re. a downsampled target
        # e.g, use something like this:
        # v_guess = model(x, t, cond)
        # downsampled_target = model.downsampler(target) 
        # bottleneck_loss = F.mse_loss(model.bottleneck_target_hook, downsampled_target) # compare with hook

        # Upward path
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        
        if training:
            return x, losses
        return x

    def forward(self, x, time, target=None, training=False, cond=None):
        if self.use_checkpoint and self.training:
            return checkpoint(
                self._forward, x, time, target, training, cond, use_reentrant=False
            )
        return self._forward(x, time, target, training, cond)
    




class MRUnet(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=4,
        condition=False,
        n_classes=10,
        use_checkpoint=False,
    ):
        super().__init__()
        self.use_checkpoint=use_checkpoint

        # determine dimensions
        self.channels = channels
        self.condition = condition
        input_channels = channels

        init_dim = dim
        self.init_conv = nn.Conv2d(
            input_channels, init_dim, 1, padding=0
        )  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        if self.condition:
            self.cond_mlp = nn.Sequential(
                nn.Embedding(n_classes, time_dim),
                nn.Linear(time_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        bottleneck_in_dim = mid_dim + channels  # Adding channels for downsampled input
        
        # Bottleneck start - handles the concatenation
        self.bottleneck_start = nn.Sequential(
            nn.Conv2d(bottleneck_in_dim, mid_dim, 3, padding=1),
            nn.GroupNorm(resnet_block_groups, mid_dim),
            nn.SiLU(),
        )
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                        ),
                    ]
                )
            )

        self.out_dim = channels

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        self.bn_size = None

    @torch.no_grad()
    def shrinker(self, input_img, x=None, mode='bilinear'):
        """Resize input image to match the spatial dimensions of x"""
        if x is not None: self.bn_size = (x.shape[2], x.shape[3])
        assert self.bn_size is not None, "No memory of bottleneck size self.bn_size - call forward first or pass in x"
        return F.interpolate(input_img, size=self.bn_size, mode=mode, align_corners=False)


    def _forward(self, x, time, cond=None):
        x0 = x.clone()
        
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        if self.condition and cond is not None:
            t += self.cond_mlp(cond)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)


        # Get downsampled input for bottleneck concat
        with torch.no_grad():
            downsampled_input = self.shrinker(x0, x)

        # Bottleneck with concatenation of explicitly coursened image
        bottleneck_input = torch.cat([x, downsampled_input], dim=1)
        x = self.bottleneck_start(bottleneck_input)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        self.bottleneck_target_hook = x[:, -self.channels:, :, :] # for loss comparison with downsampled target velocity output

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

    def forward(self, x, time, cond=None):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, time, cond, use_reentrant=False)
        return self._forward(x, time, cond)
