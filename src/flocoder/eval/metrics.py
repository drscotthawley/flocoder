import torch 
import math

def rgb2g(img_t):
   """Convert RGB piano roll to grayscale float where: BLACK->0, RED->1.0, GREEN->0.5
   Changes image from [3,H,W] to [1,H,W], and can include batch dimension."""
   red = (img_t[-3] > 0.5).float()  # 1.0 for red
   green = (img_t[-2] > 0.5).float() * 0.5  # 0.5 for green
   return (red + green).unsqueeze(-3)

def g2rgb(gf_img): # gf = greyscale float
   """Convert grayscale back to RGB: 0->BLACK, 1.0->RED, 0.5->GREEN"""
   if gf_img.shape[-3] == 3: return gf_img 
   gf = gf_img.squeeze(-3)
   return torch.stack([(gf >= 0.75).float(), (torch.abs(gf - 0.5) < 0.25).float(), torch.zeros_like(gf)], dim=-3)


def targ_pred_mask_to_rgb(t_mask, p_mask):
    # put t_mask on red channel, and p_mask on gren channel and leave blue as zero
    rgb_tensor = torch.cat([t_mask.unsqueeze(0), p_mask.unsqueeze(0), torch.zeros_like(t_mask).unsqueeze(0)], dim=0)
    #rgb_tensor = torch.cat([t_mask, p_mask, torch.zeros_like(t_mask)], dim=1)
    return rgb_tensor

def mask_to_rgb(mask, color=[1,1,1]):  # default white
    rgb = torch.zeros((mask.shape[0], 3, mask.shape[1], mask.shape[2]), device=mask.device)
    for i in range(3):
        rgb[:,i,:,:] = mask * color[i]
    return rgb

def analyze_positions(p_mask, t_mask, limit=10):
    # Get indices of positive pixels
    p_pos = torch.nonzero(p_mask, as_tuple=True)  # Get separate tensors for each dimension
    t_pos = torch.nonzero(t_mask, as_tuple=True)

    print(f"\nShape of prediction mask: {p_mask.shape}")
    print(f"Shape of target mask: {t_mask.shape}")

    print(f"\nFirst {limit} prediction positions:")
    # Print full coordinate tuples for each position
    for i in range(min(limit, len(p_pos[0]))):
        pos = tuple(dim[i].item() for dim in p_pos)
        print(f"   {pos}")

    print(f"\nFirst {limit} target positions:")
    for i in range(min(limit, len(t_pos[0]))):
        pos = tuple(dim[i].item() for dim in t_pos)
        print(f"   {pos}")


def calculate_note_metrics(pred, target, threshold=0.4, minval=None, maxval=None, debug=False):
    if debug: print(f"Before:  pred.shape = {pred.shape}, target.shape = {target.shape}")
    pred, target = g2rgb(pred), g2rgb(target)
    if debug: print(f"After:  pred.shape = {pred.shape}, target.shape = {target.shape}")
    if minval is None:  minval = target.min()
    if maxval is None:  maxval = target.max()
    pred_clamped = torch.clamp(pred.clone(), minval, maxval)
    target_unit = (target - minval) / (maxval - minval)  # rescale to [0,1]
    pred_unit = (pred_clamped - minval) / (maxval - minval)

       
    pred_binary = torch.where(pred_unit > threshold, torch.ones_like(pred_unit), torch.zeros_like(pred_unit))
    target_binary = torch.where(target_unit > threshold, torch.ones_like(target_unit), torch.zeros_like(target_unit))
    metrics, metric_images = {}, {}

    # print a range of values to check for alignment issues
    b, i_start, j_start, square_size = 1, 50, 50, 6
    c = 1 # channel
    if debug:
        print("target_unit[square] = \n", target_unit[b,c,i_start:i_start+square_size, j_start:j_start+square_size].cpu().numpy())
        print("pred_unit[square] = \n",     pred_unit[b,c,i_start:i_start+square_size, j_start:j_start+square_size].cpu().numpy())
        print("target_binary[square] = \n", target_binary[b,c,i_start:i_start+square_size, j_start:j_start+square_size].cpu().numpy())
        print("pred_binary[square] = \n",     pred_binary[b,c,i_start:i_start+square_size, j_start:j_start+square_size].cpu().numpy())
    
    # separate masks for onset and sustain
    for i, name in enumerate(['onset', 'sustain']):
        channel = 0 if i == 0 else 1  # Red=onset, Green=sustain
        
        # Initialize total counters
        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0
        
        # Initialize aggregated metric images
        tp_img = torch.zeros_like(pred_binary[:,0])
        tn_img = torch.zeros_like(pred_binary[:,0])
        fp_img = torch.zeros_like(pred_binary[:,0])
        fn_img = torch.zeros_like(pred_binary[:,0])
        targpred_img = torch.zeros_like(target)
        
        # Process each batch item separately
        for b in range(pred_binary.shape[0]):
            p_mask = pred_binary[b, channel]
            t_mask = target_binary[b, channel]
            
            # Calculate metrics for this batch item
            tp_batch = (p_mask == 1) & (t_mask == 1)
            tn_batch = (p_mask == 0) & (t_mask == 0)
            fp_batch = (p_mask == 1) & (t_mask == 0)
            fn_batch = (p_mask == 0) & (t_mask == 1)
            
            # Add to totals
            total_tp += torch.sum(tp_batch).float()
            total_tn += torch.sum(tn_batch).float()
            total_fp += torch.sum(fp_batch).float()
            total_fn += torch.sum(fn_batch).float()
            
            # Add to metric images
            tp_img[b] = tp_batch.float()
            tn_img[b] = tn_batch.float()
            fp_img[b] = fp_batch.float()
            fn_img[b] = fn_batch.float()
            targpred_img[b] = targ_pred_mask_to_rgb(t_mask, p_mask)
        
        if debug:
            print(f"Channel: {channel}")
            print(f"   Number of 1s in p_mask: {torch.sum(pred_binary[:,channel] == 1)}")
            print(f"   Number of 1s in t_mask: {torch.sum(target_binary[:,channel] == 1)}")
            print(f"   Number of positions where both are 1: {total_tp}")
            
        metrics.update({
            f'{name}_sensitivity': (total_tp / (total_tp + total_fn + 1e-8)).item(), # aka recall
            f'{name}_specificity': (total_tn / (total_tn + total_fp + 1e-8)).item(),
            f'{name}_precision': (total_tp / (total_tp + total_fp + 1e-8)).item(),
            f'{name}_f1': (2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)).item()
        })
        
        # Convert binary masks to RGB for visualization
        def mask_to_rgb(mask):
            return mask.unsqueeze(1).repeat(1, 3, 1, 1)
            
        metric_images.update({
            f'{name}_tp': mask_to_rgb(tp_img),
            f'{name}_tn': mask_to_rgb(tn_img), 
            f'{name}_fp': mask_to_rgb(fp_img),
            f'{name}_fn': mask_to_rgb(fn_img),
            f'{name}_targpred': targpred_img,
        })
        
        if debug:
            print(f" {name}: tp, tn, fp, fn =", [int(x.item()) for x in [total_tp, total_tn, total_fp, total_fn]])
            
    return metrics, metric_images




def get_discriminator_stats(adv_loss, real_images, fake_images):
    with torch.no_grad():
        d_real = adv_loss.discriminator(real_images)[0].mean()  # Add [0] to get first element of tuple
        d_fake = adv_loss.discriminator(fake_images)[0].mean()  # Add [0] to get first element of tuple
        return {
            'd_real_mean': d_real.item(),
            'd_fake_mean': d_fake.item(),
            'd_conf_gap': (d_real - d_fake).item()
        }

def get_gradient_stats(discriminator):
    total_norm = 0.0
    for p in discriminator.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return {'d_grad_norm': math.sqrt(total_norm)}


