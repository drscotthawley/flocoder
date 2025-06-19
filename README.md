

<p align="center">
  <img src="images/logo.png" width="135px">
</p>

# flocoder

This is a (Work In Progress!) teaching and research package for exploring latent generative flow matching models. (The name is inspired by "vocoder.")

This project initially started as a way to provide a lightweight, fast (and interpretable?) upgrade to the diffusion model system [Pictures of MIDI](https://huggingface.co/spaces/drscotthawley/PicturesOfMIDI) for MIDI piano roll images, but `flocoder` is intended to work on more general datasets too. 


## Quickstart

Head over to [`notebooks/SD_Flower_Flow.ipynb`](https://github.com/drscotthawley/flocoder/blob/main/notebooks/SD_Flower_Flow.ipynb) and run through it for a taste. It will run on Colab. 

## Overview

Check out the sets of slides linked to on [`notebooks/README.md`](https://github.com/drscotthawley/flocoder/blob/main/notebooks/README.md).

## Architecture Overview

<img src="images/flow_schematic.jpg" width="350" alt="MIDI Flow Architecture">

The above diagram illustrates the architecture of our intended model: a VQVAE compresses MIDI data into a discrete latent space, while a flow model learns to generate new samples in the continuous latent space.  

Though we can also flow in the continuous space of a VAE like the one for Stable Diffusion, which may be easier for starters. 

## Installation

```bash
# Clone the repository
git clone https://github.com/drscotthawley/flocoder.git
cd flocoder

# Install uv if not already installed
# On macOS/Linux:
# curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows PowerShell:
# irm https://astral.sh/uv/install.ps1 | iex

# Create a virtual environment with uv, specifying Python 3.10
uv venv --python=python3.10

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the package in editable mode (See below if you get NATTEN errors!)
uv pip install -e .

# Recommended: Install development dependencies (jupyter, others...)
uv pip install -e ".[dev]"

# Recommended: install NATTEN separately with special flags
uv pip install natten --no-build-isolation
# if that fails, see NATTEN's install instructions (https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md)
# and specify exact version number, e.g.
# uv pip install natten==0.20.1+torch270cu128 -f https://whl.natten.org
# or build fromt the top of the source, e.g.:
# uv pip install --no-build-isolation git+https://github.com/SHI-Labs/NATTEN
```

## Project Structure

The project is organized as follows:

- `flocoder/`: Main package code
- `scripts/`: Training and evaluation scripts
- `configs/`: Configuration files for models and training
- `notebooks/`: Jupyter notebooks for tutorials and examples
- `tests/`: Unit tests


## Training

The package includes multiple training scripts located in main directory.

You can skip the autoencoder/"codec" training if you'd rather use the pretrained Stable Diffusion VAE, e.g. for what follows: 
```bash
export CONFIG_FILE=flowers_sd.yaml
```

### Optional: Training a VQGAN
You can use use the Stable Diffusion VAE to get started quickly. (It will auto-download).
But if you want to train your own...

```bash
export CONFIG_FILE=flowers_vqgan.yaml 
#export CONFIG_FILE=midi.yaml 
./train_vqgan.py --config-name $CONFIG_FILE
```

The autoencoder AKA "codec" (e.g. VQGAN) compresses roll images into a quantized latent representation.
This will save checkpoints in the `checkpoints/` directory. Use that checkpoint to pre-encode your data like so... 

### Pre-Encoding Data (with frozen augmentations)
Takes about 20 minutes to run on a single GPU.
```bash
./preencode_data.py --config-name $CONFIG_FILE
```

### Training the Flow Model

```bash
./train_flow.py --config-name $CONFIG_FILE
```

The flow model operates in the latent space created by the autoencoder.

### Generating Samples

```bash
# Generate new MIDI samples
./generate_samples.py --config-name $CONFIG_FILE
# or with optional gradio UI:
#./generate_samples.py --config-name $CONFIG_FILE +use_gradio=true
```

This generates new samples by sampling from the flow model and decoding through the VQVAE.

# Contributing

Contributions are VERY welcome!  See [Contributing.md](Contributing.md). Thanks in advance. 

# Discussions

Discussions are open!  Rather than starting some ad-hoc Discord server, let's share ideas, questions, insights, etc. using the Discussions tab. 

# TODO

- [x] Add Discussions area
- [x] Add Style Guide
- [x] Replace custom config/CLI arg system with Hydra or other package
- [x] Rename "vae"/"vqvae"/"vqgan" variable as just "codec"
- [x] Replace Class in `preencode_data.py` with functions as per Style Guide 
- [x] Research: Figure out why conditioning fails for latent model
- [x] Add Standalone sampler script / Gradio demo?
- [x] Add metrics (to wandb out) to quantify flow training progress (sinkhorn, FID)
- [x] Add Contributing guidelines
- [ ] Try [variable size scheduler](https://github.com/ancestor-mithril/bs-scheduler)
- [ ] Add audio example, e.g. using DAC
- [ ] low-priority: Make RK4(5) integrator fully CUDA-compatible
- [ ] Straighter/OT paths:  Add ReFlow, Minibatch OT, Ray's Rays, Curvature penalty,...
- [ ] Add jitter / diffusion for comparison 
- [ ] Add Documentation
- [ ] Improve overall introduction/orientation
- [ ] Fix "code smell" throughout -- repeated methods, hard-coded values, etc.
- [ ] Research: Figure out how to accelerate training of flows!!
- [ ] Research: Figure out how to accelerate training of vqgan
- [ ] Research: improve output quality of midi-flow (and midi-vqgan)
- [ ] Inference speedup: investigate model quantization / pruning (pytorch.ao?) 
- [ ] Ops: Add tests
- [ ] Ops: Add CI
- [ ] Investigate ["Mean Flows for One-step Generative Modeling"](https://arxiv.org/abs/2505.13447)
      

# Acknowledgement

This project is generously supported by Hyperstate Music AI.

# License

This project is licensed under the terms of the MIT license.
