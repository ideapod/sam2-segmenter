# sam2-segmenter

Segment images using [SAM2](https://github.com/facebookresearch/sam2) (Segment Anything Model 2) and prepare them for use with [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).

## What it does

Given an input image, `segment_image.py` automatically detects and segments all objects using SAM2, then saves the results in the format expected by SAM 3D Objects:

```
output/
├── image.png       ← copy of input image
├── 0.png           ← mask for largest object (RGBA)
├── 1.png           ← mask for second largest
├── ...
└── _preview.png    ← visualisation of all segments
```

## Setup

### 1. Create the environment

```bash
mamba env create -f environment.yml
mamba activate sam2-segmenter
```

### 2. Download a SAM2 checkpoint

```bash
mkdir -p checkpoints/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/sam2/
```

Other available checkpoints (smaller/faster):
- `sam2.1_hiera_base_plus.pt`
- `sam2.1_hiera_small.pt`
- `sam2.1_hiera_tiny.pt`

## Usage

```bash
# Segment an image (outputs to notebook/images/<image_stem>/ by default)
python segment_image.py --image /path/to/photo.jpg

# Keep only the 10 largest segments
python segment_image.py --image /path/to/photo.jpg --max-masks 10

# Custom output directory
python segment_image.py --image /path/to/photo.jpg --output-dir /path/to/output/

# Specify checkpoint manually
python segment_image.py --image /path/to/photo.jpg --checkpoint checkpoints/sam2/sam2.1_hiera_large.pt
```

## Options

| Flag | Description |
|------|-------------|
| `--image` | Path to input image (required) |
| `--checkpoint` | Path to SAM2 `.pt` checkpoint |
| `--model-cfg` | SAM2 model config YAML (auto-detected from checkpoint name) |
| `--output-dir` | Output directory (default: `notebook/images/<image_stem>/`) |
| `--max-masks` | Keep only the N largest segments |
| `--no-preview` | Skip saving the preview PNG |

## Platform support

Runs on:
- **Apple Silicon (Mac)** — uses Metal/MPS automatically
- **NVIDIA GPU** — uses CUDA automatically
- **CPU** — fallback if no GPU available
