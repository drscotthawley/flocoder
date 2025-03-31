# Flocoder

A (quantized) latent generative flow matching model. (The name is inspired by "vocoder.")

This is primarily intended as a lightweight, fast (and interpretable?) upgrade to the diffusion model system [Pictures of MIDI](https://huggingface.co/spaces/drscotthawley/PicturesOfMIDI) for MIDI piano roll images, but is designed to work on more general datasets too. 

## NOTE: This is all just dummy code right now, nothing works! -SHH

## Architecture Overview

<img src="images/flow_schematic.jpg" width="350" alt="MIDI Flow Architecture">

The above diagram illustrates the architecture of our model: a VQVAE compresses MIDI data into a discrete latent space, while a flow model learns to generate new samples in the continuous latent space.

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

# Create a virtual environment with uv
uv venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the package in editable mode
uv pip install -e .

# You also need to install NATTEN separately with special flags
uv pip install natten --no-build-isolation

# Optional: Install development dependencies (jupyter, others...)
uv pip install -e ".[dev]"
```

## Project Structure

The project is organized as follows:

- `src/flocoder/`: Main package code
  - `models/`: Neural network model definitions (VQVAE, UNet)
  - `data/`: Data loading and processing utilities
  - `training/`: Training logic and utilities
  - `utils/`: General utilities and helper functions
- `scripts/`: Training and evaluation scripts
- `configs/`: Configuration files for models and training
- `notebooks/`: Jupyter notebooks for tutorials and examples
- `tests/`: Unit tests

## Training

The package includes several training scripts located in the `scripts/` directory:

### Training the VQVAE

```bash
# Train the VQVAE model
python scripts/train_vqvae.py --config configs/vqvae_config.yaml
```

The VQVAE compresses MIDI piano roll images into a quantized latent representation.

### Training the Flow Model

```bash
# Train the flow matching model
python scripts/train_flow.py --config configs/flow_config.yaml
```

The flow model operates in the latent space created by the VQVAE encoder.

### Generating Samples

```bash
# Generate new MIDI samples
python scripts/generate_samples.py --checkpoint models/flow_checkpoint.pt --output samples/
```

This generates new samples by sampling from the flow model and decoding through the VQVAE.

## Configuration

Model and training parameters can be customized by editing the YAML files in the `configs/` directory.

## License

This project is licensed under the terms of the MIT license.
