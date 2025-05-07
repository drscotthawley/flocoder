#! /usr/bin/env python3

## NOTICE: This code is a work in progress and may not run as-is.

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
#from scipy import integrate # this is CPU only ewww
from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter  # SHH: I prefer wandb to tb
import wandb
use_wandb = True
from torchvision import datasets, transforms

from tqdm.auto import tqdm
#from dist_metrics import compare_distributions, make_histograms
from PIL import Image
from pathlib import Path

from flocoder.models.unet import Unet
from flocoder.models.vqvae import VQVAE

import gc


image_size = (16,16) # (64, 64) 
channels = 4
C, H, W = channels, image_size[0], image_size[1]
batch_size=12288//3
print("batch_size = ",batch_size)
B = batch_size
learning_rate = 1e-3
num_epochs = 999999 # 1000  or just let it go til the disk fills with checkpoints TODO: keep only last few checkpoints
eps = 0.001 # used in integration
condition = False  # Enaable class-conditioning
n_classes = None # updates in main()


class EMA:
    def __init__(self, model, decay=0.99, device=None):
        self.model = model
        self.decay = decay
        self.device = device if device is not None else 'cuda'
        self.shadow = {}
        self.backup = {}

        # Initialize on CPU
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.cpu().clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    assert name in self.shadow
                    # Move shadow to device, do computation, then back to CPU
                    new_average = (self.decay * self.shadow[name].to(self.device) +
                                 (1.0 - self.decay) * param.data)
                    self.shadow[name] = new_average.cpu().clone()

    def eval(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store current parameters and move them to CPU
                self.backup[name] = param.data.cpu().clone()
                # Move EMA parameters to device
                param.data = self.shadow[name].to(self.device)

    def train(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Restore original parameters from CPU to device
                param.data = self.backup[name].to(self.device)
                # Clear backup to free memory
                self.backup[name] = None


def euler_sampler(model, shape, sample_N, device):
    model.eval()
    #cond = torch.arange(10).repeat(shape[0] // 10).to(device) if condition else None
    cond = torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device) # this is for a 10x10 grid of outputs, with one class per column
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        dt = 1.0 / sample_N
        for i in range(sample_N):
            num_t = i / sample_N * (1 - eps) + eps
            t = torch.ones(shape[0], device=device) * num_t
            pred = model(x, t * 999, cond)

            x = x.detach().clone() + pred * dt

        nfe = sample_N
        return x.cpu(), nfe


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))


def rk45_sampler(model, shape, device):
    rtol = atol = 1e-05
    model.eval()
    #cond = torch.arange(n_classes).repeat(shape[0] // n_classes).to(device) if condition else None 
    cond = None# torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device) # this is for a 10x10 grid of outputs, with one class per column
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model(x, vec_t * 999)#, cond)

            return to_flattened_numpy(drift)

        solution = integrate.solve_ivp(
            ode_func,
            (eps, 1),
            to_flattened_numpy(x),
            rtol=rtol,
            atol=atol,
            method="RK45",
        )
        nfe = solution.nfev
        x = torch.tensor(solution.y[:, -1]).reshape(shape).type(torch.float32)
        model.train()
        return x, nfe


def imshow_old(img, filename):
    #metrics = compare_distributions(img, target_img)  # TODO: create target_img
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    npimg = img.permute(1, 2, 0).numpy()
    plt.imshow(npimg)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)

def imshow(img, filename):
    #img = img * 0.5 + 0.5
    imin, imax = img.min(), img.max()
    img = (img - imin) / (imax - imin) # rescale via max/min
    img = np.clip(img, 0, 1)
    npimg = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(npimg)
    pil_img.save(filename)


def save_img_grid(img, epoch, method, nfe, tag=""):
    filename = f"{method}_epoch_{epoch + 1}_nfe_{nfe}.png"
    img_grid = torchvision.utils.make_grid(img, nrow=10)
    #print(f"img.shape = {img.shape}, img_grid.shape = {img_grid.shape}") 
    file_path = os.path.join(f"output_midi-vq-{H}x{W}", filename)
    imshow(img_grid, file_path)
    name = f"demo/{tag}{method}"
    if 'euler' in name: name = name + f"_nf{nfe}"
    if use_wandb: wandb.log({name: wandb.Image(file_path, caption=f"Epoch: {epoch}")})


@torch.no_grad()
def eval(model, vqvae, epoch, method, device, sample_N=None, batch_size=100, tag="", images=None):
    model.eval()
    # saves sample generated images, unless images are passed through (target data)
    if images is None:
        if method == "euler":
            images, nfe = euler_sampler( model, shape=(batch_size, C, H, W), sample_N=sample_N, device=device)
        elif method == "rk45":
            images, nfe = rk45_sampler(model, shape=(batch_size, C, H, W), device=device)
    else:
        nfe=0
    #save_img_grid(images, f"{method}_epoch_{epoch + 1}_nfe_{nfe}.png")
    decoded_images = vqvae.decode(vqvae.expand(images.to(vqvae.device))).cpu()
    save_img_grid(images, epoch, method, nfe, tag=tag)
    save_img_grid(decoded_images, epoch, method, nfe, tag=tag+"decoded_")
    return model.train()


def keep_recent_files(n=5, directory='checkpoints', pattern='*.pt'):
    # delete all but the n most recent checkpoints/images (so the disk doesn't fill!)
    # default kwarg values guard
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[n:]:
        f.unlink()


class VQEncodedDataset(Dataset):
    def __init__(self, data_dir="/data/encoded-POP909"):
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob("*.pt"))

        # Extract class number from first file to determine number of classes
        sample_file = self.files[0].name
        pattern = re.compile(r'.*_(\d+)_.*\.pt')

        # Get all unique class numbers to determine total number of classes
        self.class_numbers = set()
        for f in self.files:
            match = pattern.match(f.name)
            if match:
                self.class_numbers.add(int(match.group(1)))

        self.n_classes = len(self.class_numbers)
        print(f"Found {len(self.files)} encoded samples across {self.n_classes} classes")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]

        class_num = 0

        # Load encoded tensor to CPU and detach from computation graph
        encoded = torch.load(file_path, map_location='cpu',weights_only=True)
        encoded = encoded.detach().requires_grad_(False)

        return encoded, class_num

def warp_time(t, dt=None, s=.5):
    """Parametric Time Warping: s = slope in the middle.
        s=1 is linear time, s < 1 goes slower near the middle, s>1 goes slower near the ends
        s = 1.5 gets very close to the "cosine schedule", i.e. (1-cos(pi*t))/2, i.e. sin^2(pi/2*x)"""
    if s<0 or s>1.5: raise ValueError(f"s={s} is out of bounds.")
    tw = 4*(1-s)*t**3 + 6*(s-1)*t**2 + (3-2*s)*t
    if dt:                           # warped time-step requested; use derivative
        return tw,  dt * 12*(1-s)*t**2 + 12*(s-1)*t + (3-2*s)
    return tw

def main():
    global n_classes 

    print("WIP WARNING: This training code is being refactored from an older, messier repo and may not yet work. Please stay tuned...")

    os.makedirs(f"output_midi-vq-{H}x{W}", exist_ok=True)
    dataset = VQEncodedDataset("/data/encoded-POP909")
    n_classes = 0
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )
    dataloader = train_dataloader # alias to avoid errors later
    print(f"Configuring model for {n_classes} classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=",device)

    # Set VQVAE parameters
    hidden_channels = 256
    num_downsamples = 3
    vq_num_embeddings = 32
    vq_embedding_dim = 256
    codebook_levels = 4
    compressed_dim = 4
    no_grad_ckpt=False

    # Initialize VQVAE
    vqvae = VQVAE(
        in_channels=3,
        hidden_channels=hidden_channels,
        num_downsamples=num_downsamples,
        vq_num_embeddings=vq_num_embeddings,
        vq_embedding_dim=vq_embedding_dim,
        compressed_dim=compressed_dim,
        codebook_levels=codebook_levels,
        use_checkpoint=not no_grad_ckpt,# this refers to gradient checkpointing
        no_natten=False,
    ).eval()

    vqvae_cp_file = "midi_vqvae_latest.pt"
    checkpoint = torch.load(vqvae_cp_file, map_location=device, weights_only=True)
    vqvae.load_state_dict(checkpoint['model_state_dict'])
    vqvae.to(device)
    vqvae.device = device
    print("vqvae checkpoints loaded")

    model = Unet(
        dim=H,
        channels=C,
        dim_mults=(1, 2, 4),
        condition=condition,
        n_classes=n_classes,
        #use_checkpoint=True,
    )
    model.to(device)
    #model_cp_file = "model_epoch_2300.pt"
    #checkpoint = torch.load(model_cp_file, map_location=device, weights_only=True)
    #model.load_state_dict(checkpoint)
    #print("model checkpoint loaded")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_wandb: wandb.init(project=f"TadaoY-midi-VQ-{H}x{W}")

    ema = EMA(model, decay=0.999, device=device)

    rolling_avg_loss, alpha = None, 0.99 
    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch}/{num_epochs}:')
        for batch, cond in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            z0 = torch.randn_like(batch)
            t = torch.rand(batch.shape[0], device=device) * (1 - eps) + eps
            t = warp_time(t)          

            t_expand = t.view(-1, 1, 1, 1).repeat(
                1, batch.shape[1], batch.shape[2], batch.shape[3]
            )
            perturbed_data = t_expand * batch + (1 - t_expand) * z0
            target = batch - z0 # velocity

            score = model(
                perturbed_data, t * 999, cond.to(device) if condition else None
            )

            losses = torch.square(score - target)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
            loss = torch.mean(losses)

            loss.backward()

            # for training stability : adaptive LR and gradient clipping
            if rolling_avg_loss is None:  
                rolling_avg_loss = loss.item()
            else:
                rolling_avg_loss = alpha * rolling_avg_loss + (1 - alpha) * loss.item()
                if loss.item() > 3 * rolling_avg_loss:  # Loss spike detected
                    optimizer.param_groups[0]['lr'] *= 0.5  # Halve the learning rate
            if use_wandb:
                wandb.log({
                    "Loss/train": loss.item(),
                    "Learning Rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch })
            pbar.set_postfix({"Loss/train":f"{loss.item():.4g}"})
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # gradient clipping for stability

            optimizer.step()
            ema.update()


        if epoch < 5 or (epoch + 1) % 10 == 0: # evals every 1 at beginning, then every 20 later
            print("Generating sample outputs...")
            images = batch[:100].cpu()
            #eval(model, epoch, "euler", device, sample_N=1)
            #eval(model, epoch, "euler", device, sample_N=2)
            #eval(model, epoch, "euler", device, sample_N=10)
            batch, score, target = None, None, None
            gc.collect()  # force clearing of GPU memory cache
            torch.cuda.empty_cache()
            eval(model, vqvae, epoch, "target_data", device, tag="", images=images)
            eval(model, vqvae, epoch, "rk45", device, tag="")
            ema.eval()
            eval(model, vqvae, epoch, "rk45", device, tag="ema_")
            ema.train()

        if (epoch + 1) % 25 == 0: # checkpoint every 100
            print("Saving checkpoint...")
            directory = f"output_midi-vq-{H}x{W}"
            torch.save( model.state_dict(), os.path.join(directory, f"model_epoch_{epoch + 1}.pt"),)
            # Save EMA weights
            ema.eval()  # Switch to EMA weights
            torch.save( model.state_dict(),  # Now contains EMA weights
                os.path.join(directory, f"model_ema_epoch_{epoch + 1}.pt"))
            ema.train()  # Switch back to regular weights
            keep_recent_files(10, directory=directory, pattern="*.pt") # occasionally clean up the disk
            keep_recent_files(100, directory=directory, pattern="*.png")


if __name__ == "__main__":
    main()
