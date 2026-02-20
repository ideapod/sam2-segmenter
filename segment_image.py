"""
Segment an image using SAM2 and save masks in the format expected by demo_multi3.py.

Usage:
    python segment_image.py --image /path/to/image.jpg
    python segment_image.py --image /path/to/image.jpg --preset historical

Output:
    notebook/images/<image_name>/
        image.png       ← copy of input image
        0.png           ← mask for object 0
        1.png           ← mask for object 1
        ...
        _preview.png    ← visualisation of all segments
        _preprocessed.png ← preprocessed image used for segmentation (if preset used)

Requirements:
    pip install sam2 opencv-python-headless
    # Download SAM2 checkpoint, e.g.:
    # wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/sam2/
"""

import os
import sys
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    "default": {
        "points_per_side": 32,
        "pred_iou_thresh": 0.80,
        "stability_score_thresh": 0.92,
        "min_mask_region_area": 500,
        "clahe": False,
        "sharpen": False,
    },
    "historical": {
        # Lower confidence thresholds to catch low-contrast objects (horses,
        # carts, people) in old, grainy, faded photographs.
        # Denser point grid to avoid missing small/occluded objects.
        "points_per_side": 64,
        "pred_iou_thresh": 0.70,
        "stability_score_thresh": 0.85,
        "min_mask_region_area": 200,
        "clahe": True,       # adaptive contrast enhancement
        "sharpen": True,     # edge sharpening to fight grain/blur
    },
}


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_rgb: np.ndarray, clahe: bool = False, sharpen: bool = False) -> np.ndarray:
    """
    Optionally apply CLAHE contrast enhancement and unsharp-mask sharpening.
    Returns a uint8 RGB numpy array.
    """
    if not clahe and not sharpen:
        return image_rgb

    try:
        import cv2
    except ImportError:
        print("⚠ opencv not installed — skipping preprocessing. Install with: pip install opencv-python-headless")
        return image_rgb

    img = image_rgb.copy()

    if clahe:
        # Convert to LAB, apply CLAHE only to the L (luminance) channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe_op.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        print("  ✓ CLAHE contrast enhancement applied")

    if sharpen:
        # Unsharp mask: output = original + amount * (original - blurred)
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        print("  ✓ Unsharp mask sharpening applied")

    return img


# ---------------------------------------------------------------------------
# SAM2 automatic mask generator
# ---------------------------------------------------------------------------

def load_sam2(checkpoint_path=None, model_cfg=None, preset_cfg=None):
    """Load SAM2 automatic mask generator."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print("✗ SAM2 not installed. Install with:")
        print("    pip install sam2")
        print("  Then download a checkpoint, e.g.:")
        print("    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/sam2/")
        sys.exit(1)

    # Default checkpoint location
    if checkpoint_path is None:
        candidates = [
            "checkpoints/sam2/sam2.1_hiera_large.pt",
            "checkpoints/sam2/sam2.1_hiera_base_plus.pt",
            "checkpoints/sam2/sam2.1_hiera_small.pt",
            "checkpoints/sam2/sam2.1_hiera_tiny.pt",
            "checkpoints/sam2/sam2_hiera_large.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                checkpoint_path = c
                break
        if checkpoint_path is None:
            print("✗ No SAM2 checkpoint found. Download one, e.g.:")
            print("    wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -P checkpoints/sam2/")
            sys.exit(1)

    # Infer model config from checkpoint name if not provided
    if model_cfg is None:
        name = Path(checkpoint_path).stem
        if "large" in name:
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "base_plus" in name:
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "small" in name:
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif "tiny" in name:
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        else:
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print(f"Loading SAM2 from: {checkpoint_path}")
    import torch
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    cfg = preset_cfg or PRESETS["default"]
    print(f"  points_per_side={cfg['points_per_side']}, "
          f"pred_iou_thresh={cfg['pred_iou_thresh']}, "
          f"stability_score_thresh={cfg['stability_score_thresh']}, "
          f"min_mask_region_area={cfg['min_mask_region_area']}")

    sam2 = build_sam2(model_cfg, checkpoint_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=cfg["points_per_side"],
        pred_iou_thresh=cfg["pred_iou_thresh"],
        stability_score_thresh=cfg["stability_score_thresh"],
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=cfg["min_mask_region_area"],
    )
    print("✓ SAM2 loaded")
    return mask_generator


def generate_masks(mask_generator, image_rgb: np.ndarray, points_per_side: int = 32) -> list:
    """Run SAM2 automatic segmentation and return sorted masks (largest first)."""
    import threading
    import time

    h, w = image_rgb.shape[:2]
    total_points = points_per_side * points_per_side

    # Rough time estimate: ~0.5ms per point per megapixel on MPS/CUDA
    megapixels = (h * w) / 1_000_000
    est_seconds = max(10, int(total_points * megapixels * 0.5))
    est_str = f"~{est_seconds}s" if est_seconds < 60 else f"~{est_seconds // 60}m{est_seconds % 60:02d}s"

    print(f"Generating masks...")
    print(f"  Image size : {w}x{h} ({megapixels:.1f} MP)")
    print(f"  Grid       : {points_per_side}x{points_per_side} = {total_points} sample points")
    print(f"  Estimated  : {est_str} (varies by hardware)")
    print(f"  Progress   : ", end="", flush=True)

    # Spinner thread — prints a dot every 5 seconds while SAM2 is running
    done = threading.Event()

    def spinner():
        start = time.time()
        while not done.is_set():
            time.sleep(5)
            if not done.is_set():
                elapsed = int(time.time() - start)
                print(f".", end="", flush=True)
                if elapsed % 30 == 0 and elapsed > 0:
                    print(f" [{elapsed}s]", end="", flush=True)

    t = threading.Thread(target=spinner, daemon=True)
    t.start()

    start = time.time()
    masks = mask_generator.generate(image_rgb)
    elapsed = time.time() - start

    done.set()
    t.join(timeout=1)
    print(f" done ({elapsed:.1f}s)")

    # Sort by area descending so the most prominent objects come first
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    print(f"✓ Found {len(masks)} segments")
    return masks


# ---------------------------------------------------------------------------
# Save in the format expected by demo_multi3.py / load_masks()
# ---------------------------------------------------------------------------

def save_output(image_path: str, image_rgb: np.ndarray, masks: list, output_dir: str):
    """
    Save image.png + 0.png, 1.png, ... into output_dir.

    Each mask PNG is RGBA where the alpha channel encodes the mask
    (255 = object, 0 = background), matching the demo dataset format.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the source image as image.png
    dest_image = os.path.join(output_dir, "image.png")
    shutil.copy2(image_path, dest_image)
    print(f"✓ Image saved to: {dest_image}")

    h, w = image_rgb.shape[:2]

    total = len(masks)
    for idx, mask_data in enumerate(masks):
        binary = mask_data["segmentation"].astype(np.uint8)  # H x W, values 0/1

        # Build RGBA PNG: colour from image, alpha from mask
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = image_rgb
        rgba[..., 3] = binary * 255  # alpha: 255 inside mask, 0 outside

        out_path = os.path.join(output_dir, f"{idx}.png")
        Image.fromarray(rgba, mode="RGBA").save(out_path)

        # Progress bar
        pct = (idx + 1) / total
        bar = ("█" * int(pct * 20)).ljust(20)
        print(f"\r  Saving masks: [{bar}] {idx+1}/{total}", end="", flush=True)

    print()  # newline after progress bar
    print(f"✓ {len(masks)} masks saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Optional: visualise all masks overlaid on the image
# ---------------------------------------------------------------------------

def save_preview(image_rgb: np.ndarray, masks: list, out_path: str):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.imshow(image_rgb)
    np.random.seed(42)
    colors = np.random.rand(len(masks), 3)
    for i, m in enumerate(masks):
        overlay = np.zeros((*m["segmentation"].shape, 4))
        overlay[..., :3] = colors[i]
        overlay[..., 3] = m["segmentation"] * 0.45
        ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(f"{len(masks)} segments", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Preview saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment an image with SAM2 and prepare it for demo_multi3.py"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default=None, help="Path to SAM2 .pt checkpoint")
    parser.add_argument("--model-cfg", default=None, help="SAM2 model config YAML")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write output (default: notebook/images/<image_stem>/)",
    )
    parser.add_argument(
        "--max-masks",
        type=int,
        default=None,
        help="Keep only the N largest segments (default: keep all)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="default",
        help=(
            "Segmentation preset. "
            "'historical' uses lower confidence thresholds + CLAHE contrast "
            "enhancement + sharpening — good for old, grainy, low-contrast photos. "
            "(default: default)"
        ),
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip saving preview PNG")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        sys.exit(1)

    # Determine output directory
    image_stem = Path(image_path).stem
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(script_dir, "notebook", "images", image_stem)

    print(f"\nInput  : {image_path}")
    print(f"Output : {output_dir}")
    print(f"Preset : {args.preset}\n")

    # Load image
    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    print(f"✓ Image loaded: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # Apply preprocessing if the preset calls for it
    preset_cfg = PRESETS[args.preset]
    if preset_cfg["clahe"] or preset_cfg["sharpen"]:
        print("Preprocessing image...")
        image_processed = preprocess_image(
            image_rgb,
            clahe=preset_cfg["clahe"],
            sharpen=preset_cfg["sharpen"],
        )
        # Save preprocessed image for inspection
        os.makedirs(output_dir, exist_ok=True)
        pre_path = os.path.join(output_dir, "_preprocessed.png")
        Image.fromarray(image_processed).save(pre_path)
        print(f"  ✓ Preprocessed image saved to: {pre_path}")
    else:
        image_processed = image_rgb

    # Load SAM2 and generate masks (run on preprocessed image)
    mask_generator = load_sam2(args.checkpoint, args.model_cfg, preset_cfg)
    masks = generate_masks(mask_generator, image_processed, points_per_side=preset_cfg["points_per_side"])

    if args.max_masks:
        masks = masks[: args.max_masks]
        print(f"  Keeping top {args.max_masks} masks by area")

    # Save outputs (masks applied to original image, not preprocessed)
    save_output(image_path, image_rgb, masks, output_dir)

    # Save preview using original image
    if not args.no_preview:
        preview_path = os.path.join(output_dir, "_preview.png")
        save_preview(image_rgb, masks, preview_path)

    print(f"\n{'='*60}")
    print("✓ Done! Run the model with:")
    print(f"    python demo_multi3.py --image {os.path.join(output_dir, 'image.png')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
