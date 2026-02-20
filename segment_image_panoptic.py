"""
Segment an image using panoptic segmentation (Mask2Former) + SAM2 boundary refinement.

Mask2Former produces semantically meaningful segments — "building", "person", "vehicle",
"sky", etc. — rather than edge fragments.  SAM2 then sharpens each mask's boundaries
to be pixel-accurate.

Usage:
    python segment_image_panoptic.py --image /path/to/image.jpg
    python segment_image_panoptic.py --image /path/to/image.jpg --no-refine
    python segment_image_panoptic.py --image /path/to/image.jpg --min-area 0.005

Output:
    notebook/images/<image_name>/
        image.png           ← copy of input image
        0.png               ← mask for segment 0
        1.png               ← mask for segment 1
        ...
        _preview.png        ← colour-coded visualisation of all segments
        _labels.txt         ← segment index → class label mapping

Requirements:
    pip install transformers accelerate sam2
    # SAM2 checkpoint, e.g.:
    # wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \\
    #      -P checkpoints/sam2/

Models used:
    Panoptic : facebook/mask2former-swin-large-coco-panoptic  (~830 MB, downloads automatically)
    Refinement: SAM2 (local checkpoint)
"""

import os
import sys
import argparse
import shutil
import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Step 1 — Panoptic segmentation with Mask2Former
# ---------------------------------------------------------------------------

def run_panoptic(image_pil: Image.Image, device: str):
    """
    Run Mask2Former panoptic segmentation.

    Returns a list of dicts:
        {
            "label":       str   — human-readable class name
            "score":       float — confidence
            "mask":        np.ndarray (H, W, bool) — binary mask
            "area_frac":   float — fraction of image covered
        }
    sorted largest-area-first.
    """
    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        import torch
    except ImportError:
        print("✗ transformers not installed. Install with:")
        print("    pip install transformers accelerate")
        sys.exit(1)

    MODEL = "facebook/mask2former-swin-large-coco-panoptic"
    print(f"Loading Mask2Former ({MODEL})...")
    print("  (First run downloads ~830 MB — cached afterwards)")

    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL)
    model = model.to(device)
    model.eval()
    print("✓ Mask2Former loaded")

    print("Running panoptic segmentation...")
    import torch
    with torch.no_grad():
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    # Post-process to panoptic map
    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image_pil.size[::-1]],  # (H, W)
    )[0]

    seg_map = result["segmentation"].cpu().numpy()   # H x W, int ids (0 = background)
    segments_info = result["segments_info"]

    h, w = seg_map.shape
    total_pixels = h * w

    segments = []
    for seg in segments_info:
        seg_id = seg["id"]
        label_id = seg["label_id"]
        score = seg.get("score", 1.0)
        label = model.config.id2label.get(label_id, f"class_{label_id}")

        mask = (seg_map == seg_id)
        area_frac = mask.sum() / total_pixels

        segments.append({
            "label": label,
            "score": score,
            "mask": mask,
            "area_frac": area_frac,
        })

    # Sort largest first
    segments.sort(key=lambda s: s["area_frac"], reverse=True)
    print(f"✓ Found {len(segments)} panoptic segments")
    for i, s in enumerate(segments):
        print(f"  [{i:3d}] {s['label']:30s}  area={s['area_frac']*100:.1f}%  score={s['score']:.2f}")

    return segments


# ---------------------------------------------------------------------------
# Step 2 — SAM2 boundary refinement
# ---------------------------------------------------------------------------

def load_sam2_predictor(checkpoint_path=None, model_cfg=None, device="cpu"):
    """Load SAM2 in prompted-predictor mode (not auto-generator)."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("✗ SAM2 not installed. Install with: pip install sam2")
        sys.exit(1)

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
            print("✗ No SAM2 checkpoint found.")
            sys.exit(1)

    if model_cfg is None:
        name = Path(checkpoint_path).stem
        if "large" in name:
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "base_plus" in name:
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "small" in name:
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        else:
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    print(f"Loading SAM2 predictor from: {checkpoint_path}")
    sam2 = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2)
    print("✓ SAM2 predictor loaded")
    return predictor


def refine_mask_with_sam2(predictor, image_rgb: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
    """
    Use SAM2 in prompted mode to refine a coarse panoptic mask.

    Strategy: sample foreground and background points from the coarse mask,
    then let SAM2 snap boundaries to image edges.
    Returns a refined binary mask (H, W, bool).
    """
    import torch

    h, w = coarse_mask.shape
    fg_pixels = np.argwhere(coarse_mask)
    bg_pixels = np.argwhere(~coarse_mask)

    if len(fg_pixels) < 5 or len(bg_pixels) < 5:
        return coarse_mask  # too small to refine

    # Sample up to 10 foreground and 5 background points
    rng = np.random.default_rng(42)
    n_fg = min(10, len(fg_pixels))
    n_bg = min(5, len(bg_pixels))
    fg_sample = fg_pixels[rng.choice(len(fg_pixels), n_fg, replace=False)]
    bg_sample = bg_pixels[rng.choice(len(bg_pixels), n_bg, replace=False)]

    # SAM2 expects (x, y) — argwhere gives (row, col) = (y, x)
    fg_pts = fg_sample[:, ::-1].tolist()   # (x, y)
    bg_pts = bg_sample[:, ::-1].tolist()

    point_coords = np.array(fg_pts + bg_pts, dtype=np.float32)
    point_labels = np.array([1] * n_fg + [0] * n_bg, dtype=np.int32)

    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    # Pick the mask that best overlaps with the coarse mask
    best_idx = 0
    best_iou = -1.0
    for i, m in enumerate(masks):
        intersection = (m & coarse_mask).sum()
        union = (m | coarse_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    # Fall back to coarse if refinement makes things worse (IoU < 0.5)
    if best_iou < 0.5:
        return coarse_mask

    return masks[best_idx].astype(bool)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_output(image_path: str, image_rgb: np.ndarray, segments: list, output_dir: str):
    """Save image.png + 0.png, 1.png, ... and a label map."""
    os.makedirs(output_dir, exist_ok=True)

    dest_image = os.path.join(output_dir, "image.png")
    shutil.copy2(image_path, dest_image)
    print(f"✓ Image copied to: {dest_image}")

    h, w = image_rgb.shape[:2]
    label_lines = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        binary = seg["mask"].astype(np.uint8)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = image_rgb
        rgba[..., 3] = binary * 255

        out_path = os.path.join(output_dir, f"{idx}.png")
        Image.fromarray(rgba, mode="RGBA").save(out_path)
        label_lines.append(f"{idx}\t{seg['label']}\t{seg['area_frac']*100:.1f}%")

        pct = (idx + 1) / total
        bar = ("█" * int(pct * 20)).ljust(20)
        print(f"\r  Saving masks: [{bar}] {idx+1}/{total}", end="", flush=True)

    print()

    # Write label map
    labels_path = os.path.join(output_dir, "_labels.txt")
    with open(labels_path, "w") as f:
        f.write("index\tlabel\tarea\n")
        f.write("\n".join(label_lines))
    print(f"✓ {len(segments)} masks saved to: {output_dir}")
    print(f"✓ Label map saved to: {labels_path}")


def save_preview(image_rgb: np.ndarray, segments: list, out_path: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(image_rgb)

    np.random.seed(42)
    colors = np.random.rand(len(segments), 3)
    patches = []

    for i, seg in enumerate(segments):
        overlay = np.zeros((*seg["mask"].shape, 4))
        overlay[..., :3] = colors[i]
        overlay[..., 3] = seg["mask"] * 0.5
        ax.imshow(overlay)
        patches.append(mpatches.Patch(color=colors[i], label=f"{i}: {seg['label']} ({seg['area_frac']*100:.1f}%)"))

    ax.axis("off")
    ax.set_title(f"{len(segments)} panoptic segments", fontsize=13)
    # Legend (up to 20 entries to keep it readable)
    ax.legend(handles=patches[:20], loc="upper right", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Preview saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Panoptic segmentation (Mask2Former + SAM2 refinement) for demo_multi3.py"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.002,
        help="Discard segments covering less than this fraction of the image (default: 0.002 = 0.2%%)",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip SAM2 boundary refinement (faster, less accurate edges)",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to SAM2 .pt checkpoint")
    parser.add_argument("--model-cfg", default=None, help="SAM2 model config YAML")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write output (default: notebook/images/<image_stem>/)",
    )
    parser.add_argument("--no-preview", action="store_true", help="Skip saving preview PNG")
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        sys.exit(1)

    image_stem = Path(image_path).stem
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(script_dir, "notebook", "images", image_stem)
    )

    print(f"\nInput  : {image_path}")
    print(f"Output : {output_dir}")
    print(f"Min area filter : {args.min_area*100:.2f}% of image")
    print(f"SAM2 refinement : {'off' if args.no_refine else 'on'}\n")

    device = get_device()
    print(f"Device : {device}\n")

    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    print(f"✓ Image loaded: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # --- Step 1: Panoptic segmentation ---
    segments = run_panoptic(image_pil, device)

    # Filter tiny segments
    before = len(segments)
    segments = [s for s in segments if s["area_frac"] >= args.min_area]
    if before != len(segments):
        print(f"  Filtered {before - len(segments)} segments below {args.min_area*100:.2f}% area threshold")
        print(f"  Keeping {len(segments)} segments")

    if not segments:
        print("✗ No segments remaining after filtering. Try lowering --min-area.")
        sys.exit(1)

    # --- Step 2: SAM2 boundary refinement ---
    if not args.no_refine:
        predictor = load_sam2_predictor(args.checkpoint, args.model_cfg, device)
        predictor.set_image(image_rgb)

        print(f"\nRefining {len(segments)} masks with SAM2...")
        for i, seg in enumerate(segments):
            pct = (i + 1) / len(segments)
            bar = ("█" * int(pct * 20)).ljust(20)
            print(f"\r  [{bar}] {i+1}/{len(segments)}", end="", flush=True)
            seg["mask"] = refine_mask_with_sam2(predictor, image_rgb, seg["mask"])
        print("\n✓ Refinement complete")

    # --- Save ---
    save_output(image_path, image_rgb, segments, output_dir)

    if not args.no_preview:
        preview_path = os.path.join(output_dir, "_preview.png")
        save_preview(image_rgb, segments, preview_path)

    print(f"\n{'='*60}")
    print("✓ Done! Run the model with:")
    print(f"    python demo_multi3.py --image {os.path.join(output_dir, 'image.png')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
