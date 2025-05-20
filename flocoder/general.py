import os 
import torch 
from pathlib import Path


def handle_config_path():
    """Making up for Hydra weirdness: allow for --config-name to include full file path"""
    import os, sys

    for i, arg in enumerate(sys.argv):
        path = None
        if arg.startswith('--config-name='):  # Handle equals format
            path = arg.split('=', 1)[1]
        elif arg == '--config-name' and i+1 < len(sys.argv):  # Handle space format
            path = sys.argv[i+1]

        if path and '/' in path and path.endswith('.yaml') and os.path.exists(os.path.expanduser(path)):
            full_path = os.path.expanduser(path)
            config_dir, config_file = os.path.dirname(full_path), os.path.basename(full_path).replace('.yaml', '')

            # Update args based on format used
            if arg.startswith('--config-name='):
                sys.argv[i] = f"--config-name={config_file}"
                sys.argv.insert(i, f"--config-path={config_dir}")
            else:
                sys.argv[i+1] = config_file
                sys.argv.insert(i+1, f"--config-path={config_dir}")
            break


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
