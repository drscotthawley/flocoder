# This is code by Tadao Yamaoka, modified and re-shared with his permission
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from scipy import integrate
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter  # SHH: I prefer wandb to tb
import wandb
use_wandb = True 

from torchvision import datasets, transforms

from tqdm.auto import tqdm
#from dist_metrics import compare_distributions, make_histograms
from PIL import Image
from pathlib import Path

# use my utils and such 
from flocoder.training.losses import sinkhorn_loss
from flocoder.eval.metrics import fid_score


# import UNet from the legacy dir: Add the directory containing this script to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from unet_ty import Unet_TY as Unet


# run metaparameters
image_size = (32,32) # (64, 64) 
channels = 3
batch_size=256
print("batch_size = ",batch_size)
B, C, H, W = batch_size, channels, image_size[0], image_size[1]
learning_rate = 0.001
num_epochs = 999999 # 1000  or just let it go til the disk fills with checkpoints TODO: keep only last few checkpoints
eps = 0.001
condition = True  # Enaable class-conditioning
#n_classes = 10 # cifar10
#n_classes = 102 # flowers102
n_classes = None # global for now, but it will update in main()


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
    cond = torch.randint(n_classes,(10,)).repeat(shape[0] // 10).to(device) # this is for a 10x10 grid of outputs, with one class per column
    with torch.no_grad():
        z0 = torch.randn(shape, device=device)
        x = z0.detach().clone()

        def ode_func(t, x):
            x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
            vec_t = torch.ones(shape[0], device=x.device) * t
            drift = model(x, vec_t * 999, cond)

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
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    npimg = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(npimg)
    pil_img.save(filename)


def save_img_grid(img, epoch, method, nfe):
    filename = f"{method}_epoch_{epoch + 1}_nfe_{nfe}.png"
    img_grid = torchvision.utils.make_grid(img, nrow=10)
    #print(f"img.shape = {img.shape}, img_grid.shape = {img_grid.shape}") 
    file_path = os.path.join(f"output_flowers-{H}x{W}", filename)
    imshow(img_grid, file_path)
    name = f"demo/{method}"
    if 'euler' in name: name = name + f"_nf{nfe}"
    if use_wandb: wandb.log({name: wandb.Image(file_path, caption=f"Epoch: {epoch}")})


def eval(model, epoch, method, device, sample_N=None, batch_size=100, target=None):
    # saves sample generated images
    if method == "euler":
        gen_images, nfe = euler_sampler(
            model, shape=(batch_size, C, H, W), sample_N=sample_N, device=device
        )
    elif method == "rk45":
        gen_images, nfe = rk45_sampler(model, shape=(batch_size, C, H, W), device=device)
    #save_img_grid(gen_images, f"{method}_epoch_{epoch + 1}_nfe_{nfe}.png")
    save_img_grid(gen_images, epoch, method, nfe)

    # metrics on generated outputs:
    if target is not None:
        sinkhorn = sinkhorn_loss(target[:batch_size], gen_images, device=device)
        fid = fid_score(target[:batch_size], gen_images)
        if use_wandb: wandb.log({'epoch':epoch, 'metrics/sinkhorn': sinkhorn, 'metrics/FID': fid } )  



def keep_recent_files(n=5, directory='checkpoints', pattern='*.pt'):
    # delete all but the n most recent checkpoints/images (so the disk doesn't fill!)
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[n:]:
        f.unlink()

def main():
    global n_classes 

    os.makedirs(f"output_flowers-{H}x{W}", exist_ok=True)
    #writer = SummaryWriter(log_dir="runs/experiment1")
    if use_wandb: wandb.init(project=f"TadaoY-flowers-{H}x{W}")
   
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    #train_dataset = datasets.CIFAR10( root="./data", train=True, download=True, transform=transform)
    train_dataset = datasets.Flowers102( root="~/datasets/Oxford_Flowers102", split="train", download=True, transform=transform)
    dataset = train_dataset # alias to make coding easier
    #n_classes == len(train_dataset.classes)
    n_classes = len(set(train_dataset._labels))  # Flowers102 uses _labels internally
    print(f"Configuring model for {n_classes} classes")

    # observation: we have no data augmentation??
    dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=",device)

    model = Unet(
        dim=H,
        channels=C,
        dim_mults=(1, 2, 4),
        condition=condition,
        n_classes=n_classes,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{num_epochs}:')
        for batch, cond in pbar:
            batch = batch.to(device)

            optimizer.zero_grad()

            z0 = torch.randn_like(batch)
            t = torch.rand(batch.shape[0], device=device) * (1 - eps) + eps

            t_expand = t.view(-1, 1, 1, 1).repeat(
                1, batch.shape[1], batch.shape[2], batch.shape[3]
            )
            perturbed_data = t_expand * batch + (1 - t_expand) * z0
            target = batch - z0

            score = model(
                perturbed_data, t * 999, cond.to(device) if condition else None
            )

            losses = torch.square(score - target)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)

            loss = torch.mean(losses)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss/train":f"{(total_loss / len(dataloader)):.4g}"})

        #print(f"\nEpoch {epoch}/{num_epochs}: Loss/train= {total_loss / len(dataloader)}")
        #writer.add_scalar("Loss/train", total_loss / len(dataloader), epoch)
        if use_wandb: wandb.log({"Loss/train": total_loss / len(dataloader), "epoch":epoch})

        if epoch < 5 or (epoch + 1) % 20 == 0: # evals every 1 at beginning, then every 20 later
            print("\nRunning evals...") 
            #eval(model, epoch, "euler", device, sample_N=1)
            #eval(model, epoch, "euler", device, sample_N=2)
            #eval(model, epoch, "euler", device, sample_N=10)
            eval(model, epoch, "rk45", device, target=target)

        if (epoch + 1) % 100 == 0: # checkpoint every 100
            print("Saving checkpoint...")
            directory = f"output_flowers-{H}x{W}"
            torch.save(
                model.state_dict(),
                os.path.join(directory, f"model_epoch_{epoch + 1}.pt"),)
            keep_recent_files(5, directory=directory, pattern="*.pt") # occasionally clean up the disk
            keep_recent_files(100, directory=directory, pattern="*.png")


if __name__ == "__main__":
    main()
