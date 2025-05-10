# Flocoder

A (quantized) latent generative flow matching model. (The name is inspired by "vocoder.")

This is primarily intended as a lightweight, fast (and interpretable?) upgrade to the diffusion model system [Pictures of MIDI](https://huggingface.co/spaces/drscotthawley/PicturesOfMIDI) for MIDI piano roll images, but is designed to work on more general datasets too. 

## NOTE: This is a refactor of older/messier code. Not everything in this refactor works yet. -SHH

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

# Create a virtual environment with uv, specifying Python 3.10
uv venv --python=python3.10

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install the package in editable mode
uv pip install -e .

# Recommended: Install development dependencies (jupyter, others...)
uv pip install -e ".[dev]"

# Recommended: install NATTEN separately with special flags
uv pip install natten --no-build-isolation
# if that fails, see NATTEN's install instructions (https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md)
# and specify exact version number, e.g.
# uv pip install natten==0.17.3+torch250cu124 -f https://shi-labs.com/natten/wheels/
# or build fromt the top of the source, e.g.:
# uv pip install --no-build-isolation git+https://github.com/SHI-Labs/NATTEN
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

### Training the VQGAN
(terminology note: VQGAN = VQVAE + attention + adversarial loss. We use VQGAN/VQVAE somewhat interchangeably.)

```bash
python scripts/train_vqgan.py --config configs/pop909_config.yaml
```

The VQVAE compresses MIDI piano roll images into a quantized latent representation.
This will save checkpoints in the `checkpoints/` directory. Use that checkpoint to pre-encode your data like so... 

### Pre-Encoding Data (with frozen augmentations)
```bash
python scripts/preencode_data.py --config configs/pop909_config.yaml --checkpoint [your_vqgan_checkpoint.pt]
```

### Training the Flow Model

```bash
python scripts/train_flow.py --config configs/pop909_config.yaml
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
