
import os 
import torch 
from pathlib import Path


def keep_recent_files(keep=5, directory='checkpoints', pattern='*.pt'):
    # delete all but the n most recent checkpoints/images (so the disk doesn't fill!)
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[keep:]:
        f.unlink()


def save_checkpoint(model, epoch=None, optimizer=None, keep=5, prefix="vqgan", ckpt_dir='checkpoints'):

    keep_recent_files(keep=keep, directory=ckpt_dir, pattern=f'{prefix}*.pt')
 
    ckpt_path = f'{ckpt_dir}/{prefix}.pt' 
    save_dict = {'model_state_dict': model.state_dict()}
    if epoch is not None: 
        ckpt_path.replace('.pt', f'_{epoch}.pt')
        save_dict['epoch'] = epoch
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(save_dict, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path