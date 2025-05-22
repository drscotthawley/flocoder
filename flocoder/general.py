import os 
import torch 
from pathlib import Path
import warnings



def handle_config_path():
    """Making up for Hydra weirdness: allow for --config-name to include full file path, with or without equals sign"""
    import os, sys

    # First pass: Fix arguments without equals sign
    for i, arg in enumerate(sys.argv):
        if arg == '--config-name' and i+1 < len(sys.argv):
            # Convert "--config-name value" to "--config-name=value"
            sys.argv[i] = f"--config-name={sys.argv[i+1]}"
            sys.argv.pop(i+1)  # Remove the next argument as it's now part of this one
            break  # Only do this once to avoid messing up indices

    # Second pass: Handle full paths with equals sign
    for i, arg in enumerate(sys.argv):
        path = None
        if arg.startswith('--config-name='):  # Handle equals format
            path = arg.split('=', 1)[1]
            
        if path and '/' in path and path.endswith('.yaml') and os.path.exists(os.path.expanduser(path)):
            full_path = os.path.expanduser(path)
            config_dir, config_file = os.path.dirname(full_path), os.path.basename(full_path).replace('.yaml', '')
            
            sys.argv[i] = f"--config-name={config_file}"
            sys.argv.insert(i, f"--config-path={config_dir}")
            break


def ldcfg(config, key, default=None, supply_defaults=False, debug=True, verbose=True):
    # little helper function: hydra/omegaconf is nice but also a giant pain.
    # ithis gives precedence to anything in vqgan section, else falls back to main config, else default, else None
    # re. supply_defaults: Hydra is tricksy enough that for some things you may just want execution to crash if config is misread
    cfg_dict = config
    answer = None
    if hasattr(config, 'to_container'):  # OmegaConf objects
        cfg_dict = OmegaConf.to_container(config, resolve=True)
    if 'flow' in cfg_dict and key in cfg_dict['flow']:   # the order of cases is important here 
        answer =  cfg_dict['flow'][key]
    elif 'preencoding' in cfg_dict and key in cfg_dict['preencoding']: 
        answer = cfg_dict['preencoding'][key]
    elif 'codec' in cfg_dict and key in cfg_dict['codec']: 
        return cfg_dict['codec'][key]
    elif key in cfg_dict: 
        answer = cfg_dict[key]
    else:
        warnings.warn(f"couldn't find key {key} anywhere in config={cfg_dict}")
        answer = default if supply_defaults else None

    if verbose: print(f'lcfg: {key} := {answer}')
    return answer


def keep_recent_files(keep=5, directory='checkpoints', pattern='*.pt'):
    # delete all but the n most recent checkpoints/images (so the disk doesn't fill!)
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[keep:]:
        f.unlink()




def load_model_checkpoint(model, checkpoint, frozen_only=False):
    """
    Load parameters from a checkpoint into the model.
    If frozen_only=True, loads only frozen parameters. Otherwise, loads all parameters.
    """
    checkpoint_state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()

    if frozen_only:
        # Only load frozen parameters (original behavior)
        filtered_state_dict = {}

        # Get only frozen parameters
        for name, param in model.named_parameters():
            if not param.requires_grad and name in checkpoint_state_dict:
                filtered_state_dict[name] = checkpoint_state_dict[name]

        # Include buffers for frozen parts
        for name, buffer in model.named_buffers():
            if (name.startswith('encoder.') or name.startswith('compress.')) and name in checkpoint_state_dict:
                filtered_state_dict[name] = checkpoint_state_dict[name]

        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
        print(f"Loaded {len(filtered_state_dict)} frozen parameters.")
    else:
        # Load all parameters
        model.load_state_dict(checkpoint_state_dict)
        print(f"Loaded all parameters from checkpoint.")

    return model



def save_checkpoint(model, epoch=None, optimizer=None, keep=5, prefix="vqgan", ckpt_dir='checkpoints'):

    keep_recent_files(keep=keep, directory=ckpt_dir, pattern=f'{prefix}*.pt')
 
    ckpt_path = f'{ckpt_dir}/{prefix}.pt' 
    save_dict = {'model_state_dict': model.state_dict()}
    if epoch is not None: 
        ckpt_path = ckpt_path.replace('.pt', f'_{epoch}.pt')
        save_dict['epoch'] = epoch
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(save_dict, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path


