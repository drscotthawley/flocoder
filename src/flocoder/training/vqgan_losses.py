import torch.nn.functional as F
import torch
from torch.nn.utils import spectral_norm
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.flocoder.models.patch_discriminator import PatchDiscriminator

# Note: not all of these actually get used; and/or can be affected via config.lamba_* values 

def piano_roll_rgb_cross_entropy(pred,                  # pred: [B, 3, H, W] tensor of predicted probabilities 
                                 target,                # target: [B, 3, H, W] tensor of ground truth RGB where:
                                                            # - black = [0,0,0] (background)
                                                            # - red = [1,0,0] (onset)
                                                            # - green = [0,1,0] (sustain)
                                 temperature=0.25,      # Temperature for softening predictions (higher = softer)
                                 onset_threshold=0.3,   # Value above which a color of onset channel is considered "on"
                                 sustain_threshold=0.5, # Value above which a color of sustain channel is considered "on"
                                 eps=1e-8,              # Small value to avoid log(0)
                                 debug=False):
    """    Compute cross entropy loss for RGB piano roll images with thresholding    """
    #targets & preds may have imagenet norm (if dataloader norm'd them) so before doing BCE loss we need to un-norm them
    target_unnorm = target
    #mean = torch.tensor([0.485, 0.456, 0.406])[None,:,None,None].to(target.device) 
    #std = torch.tensor([0.229, 0.224, 0.225])[None,:,None,None].to(target.device)
    #target_unnorm = target * std + mean
    # if debug:
    #     print(f"Unnormalized target mins: {[target_unnorm[:,i].min().item() for i in range(3)]}")
    #     print(f"Unnormalized target maxs: {[target_unnorm[:,i].max().item() for i in range(3)]}")
    
    # Different thresholds for each color (background, onset, sustain) channel
    thresholds = torch.tensor([onset_threshold, sustain_threshold, 1.0])[None,:,None,None].to(target.device)
    target_binary = torch.where(target_unnorm > thresholds, torch.ones_like(target), torch.zeros_like(target))
    pred = pred / temperature # Scale logits by temperature
    loss = F.binary_cross_entropy_with_logits(pred, target_binary)
    return loss.mean() # Sum across channels and average over batch and spatial dimensions



def perceptual_loss(vgg, img1, img2):
    # Normalize images to VGG range
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(img1.device)
    img1, img2 = (img1 - mean) / std, (img2 - mean) / std

    features1, features2 = vgg(img1), vgg(img2) # Get features from multiple layers
    
    # Compute loss at each layer
    loss = 0
    for f1, f2 in zip(features1, features2):
        loss += F.mse_loss(f1, f2)
    return loss



def pwrspec(y, eps=1e-8): 
    # called by spectral_loss
    if y.dtype != torch.float: 
        y = y.to(torch.float)  # Cast to float even if MP's on, b/c fft isn't optimized for half
    return torch.log(eps + torch.abs(torch.fft.fft2(y))) # use the log-magnitude. (no phase info)

def spectral_loss(x, x_recon):
    x_spec, x_recon_spec = pwrspec(x), pwrspec(x_recon)
    if torch.is_autocast_enabled(): 
        x_spec = x_spec.to(torch.half)
        x_recon_spec = x_recon_spec.to(torch.half) 
    return F.mse_loss(x_spec, x_recon_spec)



def compute_vqgan_losses(recon, target_imgs, vq_loss, vgg, adv_loss=None, epoch=None, config=None, fakez_recon=None): #,sinkhorn_loss=None,)
    """Compute many losses in a single place. Returns dict of loss tensors."""
    losses = {
        'ce': piano_roll_rgb_cross_entropy(recon, target_imgs),
        'mse': F.mse_loss(recon, target_imgs),
        'vq': vq_loss, # vq_loss is already computed in the quantizer (bottleneck of the VQGAN)
        'perceptual': perceptual_loss(vgg, recon, target_imgs), 
        'spectral': spectral_loss(recon, target_imgs),
        'huber': F.huber_loss(recon, target_imgs, delta=1.0) 
    }
    
    # Only add adversarial losses after warmup
    if adv_loss is not None and epoch >= config.warmup_epochs:
        d_loss, real_features = adv_loss.discriminator_loss(target_imgs, recon)
        g_loss = adv_loss.generator_loss(recon, real_features)
        losses['d_loss'] = d_loss
        losses['g_loss'] = config.lambda_adv * g_loss

    return losses


def get_total_vqgan_loss(losses, config=None):
    """Compute weighted sum of losses."""
    total = (
        config.lambda_mse*losses['mse'] + \
        config.lambda_l1*losses['huber'] + \
        config.lambda_vq*losses['vq'] + \
        config.lambda_perc * losses['perceptual'] + \
        config.lambda_spec * losses['spectral'] \
        + config.lambda_ce*losses['ce']
    )

    if 'g_loss' in losses: total = total + losses['g_loss']
    # note: d_loss gets updated elsewhere and not included in total vqgan loss
    if 's_loss' in losses: total = total + config.lambda_sinkhorn*losses['s_loss'] # haven't found this helpful, so s_loss probably won't be in there


    return total


#------------------------------------------------------------------
# Next section used in training VQGAN. Besides oridnary reconstruction loss, we include a Patch-based discriminator model

def hinge_d_loss(real_pred, fake_pred):
    return torch.mean(F.relu(1.0 - real_pred)) + torch.mean(F.relu(1.0 + fake_pred))

class AdversarialLoss(nn.Module):
    def __init__(self, device, use_checkpoint=False):
        super().__init__()

        self.device = device
        self.discriminator = PatchDiscriminator(use_checkpoint=use_checkpoint).to(device)
        self.criterion = hinge_d_loss

        self.register_buffer('real_label', torch.ones(1))
        self.register_buffer('fake_label', torch.zeros(1))
        self.to(device)


    def get_target_tensor(self, prediction, target_is_real):
        target = self.real_label if target_is_real else self.fake_label
        return target.expand_as(prediction)

    def feature_matching_loss(self, real_features, fake_features):
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.l1_loss(fake_feat, real_feat.detach())
        return loss / len(real_features)

    def discriminator_loss(self, real_images, fake_images):
        real_pred, real_features = self.discriminator(real_images)
        fake_pred, _ = self.discriminator(fake_images.detach())
        return hinge_d_loss(real_pred, fake_pred), real_features

    def generator_loss(self, fake_images, real_features=None):
        fake_pred, fake_features = self.discriminator(fake_images)
        g_loss = -torch.mean(fake_pred)
        if real_features is not None:
            fm_loss = self.feature_matching_loss(real_features, fake_features)
            g_loss = g_loss + fm_loss
        return g_loss
#------------------------------------------------------------------
    
