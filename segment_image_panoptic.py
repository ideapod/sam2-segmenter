"""
Segment an image using a hybrid panoptic + SAM2 residual pipeline.

Pipeline:
  1. Mask2Former panoptic segmentation — produces semantically meaningful segments
     (building, sky, road, person, horse, etc.) for things it was trained on.
  2. SAM2 boundary refinement — sharpens each panoptic mask to pixel accuracy.
  3. SAM2 residual segmentation — runs SAM2 auto-segmentation on the area NOT
     covered by Mask2Former, to catch objects missing from COCO vocabulary
     (horse-drawn carts, wagons, Victorian-era signage, etc.).
  4. Merge & deduplicate — combine panoptic + residual, drop overlapping fragments.

Usage:
    python segment_image_panoptic.py --image /path/to/image.jpg
    python segment_image_panoptic.py --image /path/to/image.jpg --no-refine
    python segment_image_panoptic.py --image /path/to/image.jpg --min-area 0.005

Output:
    notebook/images/<image_name>/
        image.png           <- copy of input image
        0.png, 1.png, ...   <- one mask per segment
        _preview.png        <- colour-coded visualisation with legend
        _labels.txt         <- index, label, area for each segment

Requirements:
    pip install transformers accelerate scipy sam2
    # SAM2 checkpoint, e.g.:
    # wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
    #      -P checkpoints/sam2/

Models:
    Panoptic  : facebook/mask2former-swin-large-coco-panoptic (~830 MB, auto-download)
    Refinement: SAM2 local checkpoint
"""

import os
import sys
import argparse
import shutil
import threading
import time
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
    Returns list of dicts sorted largest-area-first:
      { label, score, mask (H,W bool), area_frac, source="panoptic" }
    """
    try:
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        import torch
    except ImportError:
        print("✗ transformers not installed.  pip install transformers accelerate")
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
    with torch.no_grad():
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

    result = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=[image_pil.size[::-1]],   # (H, W)
    )[0]

    seg_map = result["segmentation"].cpu().numpy()
    h, w = seg_map.shape
    total_pixels = h * w

    segments = []
    for seg in result["segments_info"]:
        label = model.config.id2label.get(seg["label_id"], f"class_{seg['label_id']}")
        mask  = (seg_map == seg["id"])
        area_frac = mask.sum() / total_pixels
        segments.append({
            "label":     label,
            "score":     seg.get("score", 1.0),
            "mask":      mask,
            "area_frac": area_frac,
            "source":    "panoptic",
        })

    segments.sort(key=lambda s: s["area_frac"], reverse=True)
    print(f"✓ Found {len(segments)} panoptic segments")
    for i, s in enumerate(segments):
        print(f"  [{i:3d}] {s['label']:30s}  area={s['area_frac']*100:.1f}%  score={s['score']:.2f}")

    return segments


# ---------------------------------------------------------------------------
# SAM2 helpers — predictor (prompted) and auto-generator (residual)
# ---------------------------------------------------------------------------

def _sam2_checkpoint_and_cfg(checkpoint_path=None, model_cfg=None):
    """Resolve checkpoint path and model config."""
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
            print("✗ No SAM2 checkpoint found in checkpoints/sam2/")
            sys.exit(1)

    if model_cfg is None:
        name = Path(checkpoint_path).stem
        if "large"     in name: model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "base_plus" in name: model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "small"   in name: model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        else:                     model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

    return checkpoint_path, model_cfg


def load_sam2_predictor(checkpoint_path=None, model_cfg=None, device="cpu"):
    """SAM2 in prompted-predictor mode for boundary refinement."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print("✗ SAM2 not installed.  pip install sam2")
        sys.exit(1)

    checkpoint_path, model_cfg = _sam2_checkpoint_and_cfg(checkpoint_path, model_cfg)
    print(f"Loading SAM2 predictor from: {checkpoint_path}")
    sam2 = build_sam2(model_cfg, checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2)
    print("✓ SAM2 predictor loaded")
    return predictor


def load_sam2_generator(checkpoint_path=None, model_cfg=None, device="cpu",
                        points_per_side=32, pred_iou_thresh=0.75,
                        stability_score_thresh=0.88, min_mask_region_area=300):
    """SAM2 automatic mask generator for residual segmentation."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError:
        print("✗ SAM2 not installed.  pip install sam2")
        sys.exit(1)

    checkpoint_path, model_cfg = _sam2_checkpoint_and_cfg(checkpoint_path, model_cfg)
    print(f"Loading SAM2 generator from: {checkpoint_path}")
    sam2 = build_sam2(model_cfg, checkpoint_path, device=device)
    generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_region_area,
    )
    print("✓ SAM2 generator loaded")
    return generator


# ---------------------------------------------------------------------------
# Step 2 — SAM2 boundary refinement of panoptic masks
# ---------------------------------------------------------------------------

def refine_mask_with_sam2(predictor, coarse_mask: np.ndarray) -> np.ndarray:
    """
    Prompt SAM2 with points sampled from a coarse panoptic mask and return
    the best-matching refined mask.  Falls back to coarse if IoU < 0.5.
    """
    import torch

    fg_pixels = np.argwhere(coarse_mask)
    bg_pixels = np.argwhere(~coarse_mask)

    if len(fg_pixels) < 5 or len(bg_pixels) < 5:
        return coarse_mask

    rng = np.random.default_rng(42)
    n_fg = min(10, len(fg_pixels))
    n_bg = min(5,  len(bg_pixels))
    fg_sample = fg_pixels[rng.choice(len(fg_pixels), n_fg, replace=False)]
    bg_sample = bg_pixels[rng.choice(len(bg_pixels), n_bg, replace=False)]

    # argwhere → (row, col); SAM2 expects (x, y) = (col, row)
    point_coords = np.array(
        fg_sample[:, ::-1].tolist() + bg_sample[:, ::-1].tolist(),
        dtype=np.float32,
    )
    point_labels = np.array([1] * n_fg + [0] * n_bg, dtype=np.int32)

    with torch.no_grad():
        masks, _, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    coarse_bool = coarse_mask.astype(bool)
    best_idx, best_iou = 0, -1.0
    for i, m in enumerate(masks):
        m_bool = m.astype(bool)
        inter = (m_bool & coarse_bool).sum()
        union = (m_bool | coarse_bool).sum()
        iou   = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_idx = iou, i

    if best_iou < 0.5:
        return coarse_mask

    return masks[best_idx].astype(bool)


# ---------------------------------------------------------------------------
# Step 3 — SAM2 residual segmentation on uncovered area
# ---------------------------------------------------------------------------

def run_residual_sam2(generator, image_rgb: np.ndarray,
                      covered_mask: np.ndarray, min_area_frac: float) -> list:
    """
    Run SAM2 automatic segmentation restricted to pixels NOT covered by
    Mask2Former.  Returns new segment dicts with source="residual".

    Strategy:
      - Black-out the already-covered region so SAM2 ignores it.
      - Run SAM2 auto-generator on the masked image.
      - Keep only masks that are mostly (>60%) in the uncovered area.
      - Filter by min_area_frac as usual.
    """
    h, w = image_rgb.shape[:2]
    total_pixels = h * w

    # Pixels not yet assigned to any panoptic segment
    residual_region = ~covered_mask   # (H, W) bool

    # Give SAM2 a blacked-out version so it focuses on the residual area
    masked_image = image_rgb.copy()
    masked_image[covered_mask] = 0

    print(f"  Residual area: {residual_region.sum() / total_pixels * 100:.1f}% of image")
    print(f"  Running SAM2 on residual region...", end="", flush=True)

    done = threading.Event()
    def spinner():
        while not done.is_set():
            time.sleep(5)
            if not done.is_set():
                print(".", end="", flush=True)
    t = threading.Thread(target=spinner, daemon=True)
    t.start()

    t0 = time.time()
    raw_masks = generator.generate(masked_image)
    done.set(); t.join(timeout=1)
    print(f" done ({time.time()-t0:.1f}s)")
    print(f"  SAM2 found {len(raw_masks)} raw residual segments")

    segments = []
    for m in raw_masks:
        seg_mask = m["segmentation"].astype(bool)
        area_frac = seg_mask.sum() / total_pixels

        if area_frac < min_area_frac:
            continue

        # Only keep if the majority of this mask falls in the residual region
        overlap_frac = (seg_mask & residual_region).sum() / seg_mask.sum()
        if overlap_frac < 0.6:
            continue

        segments.append({
            "label":     "unknown",
            "score":     float(m.get("predicted_iou", 0.0)),
            "mask":      seg_mask,
            "area_frac": area_frac,
            "source":    "residual",
        })

    # Sort largest first
    segments.sort(key=lambda s: s["area_frac"], reverse=True)
    print(f"  Kept {len(segments)} residual segments after filtering")
    return segments


# ---------------------------------------------------------------------------
# Step 4 — Deduplicate / merge overlapping masks
# ---------------------------------------------------------------------------

def deduplicate(segments: list, iou_threshold: float = 0.5) -> list:
    """
    Remove residual masks that heavily overlap with panoptic masks
    (or with each other).  Keeps the higher-scoring / larger mask.
    Simple greedy NMS by IoU.
    """
    kept = []
    for seg in segments:
        m = seg["mask"].astype(bool)
        suppress = False
        for k in kept:
            k_mask = k["mask"].astype(bool)
            inter = (m & k_mask).sum()
            union = (m | k_mask).sum()
            if union > 0 and inter / union > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(seg)
    return kept


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_output(image_path: str, image_rgb: np.ndarray, segments: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    dest_image = os.path.join(output_dir, "image.png")
    shutil.copy2(image_path, dest_image)
    print(f"✓ Image copied to: {dest_image}")

    h, w = image_rgb.shape[:2]
    label_lines = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = image_rgb
        rgba[..., 3]  = seg["mask"].astype(np.uint8) * 255
        Image.fromarray(rgba, mode="RGBA").save(os.path.join(output_dir, f"{idx}.png"))
        label_lines.append(
            f"{idx}\t{seg['label']}\t{seg['area_frac']*100:.1f}%\t{seg['source']}"
        )

        pct = (idx + 1) / total
        bar = ("█" * int(pct * 20)).ljust(20)
        print(f"\r  Saving masks: [{bar}] {idx+1}/{total}", end="", flush=True)

    print()

    labels_path = os.path.join(output_dir, "_labels.txt")
    with open(labels_path, "w") as f:
        f.write("index\tlabel\tarea\tsource\n")
        f.write("\n".join(label_lines))
    print(f"✓ {len(segments)} masks saved to: {output_dir}")
    print(f"✓ Label map: {labels_path}")


def save_preview(image_rgb: np.ndarray, segments: list, out_path: str):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(image_rgb)

    np.random.seed(42)
    colors  = np.random.rand(len(segments), 3)
    patches = []

    for i, seg in enumerate(segments):
        overlay = np.zeros((*seg["mask"].shape, 4))
        overlay[..., :3] = colors[i]
        overlay[..., 3]  = seg["mask"] * 0.5
        ax.imshow(overlay)
        src_tag = "P" if seg["source"] == "panoptic" else "R"   # P=panoptic, R=residual
        patches.append(mpatches.Patch(
            color=colors[i],
            label=f"[{src_tag}] {i}: {seg['label']} ({seg['area_frac']*100:.1f}%)"
        ))

    ax.axis("off")
    ax.set_title(f"{len(segments)} segments  (P=panoptic  R=residual SAM2)", fontsize=13)
    ax.legend(handles=patches[:25], loc="upper right", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Preview saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid panoptic + SAM2 residual segmentation for demo_multi3.py"
    )
    parser.add_argument("--image",       required=True, help="Path to input image")
    parser.add_argument("--checkpoint",  default=None,  help="SAM2 .pt checkpoint path")
    parser.add_argument("--model-cfg",   default=None,  help="SAM2 model config YAML")
    parser.add_argument("--output-dir",  default=None,  help="Output directory (default: notebook/images/<stem>/)")
    parser.add_argument(
        "--min-area", type=float, default=0.002,
        help="Discard segments covering < this fraction of image (default: 0.002 = 0.2%%)",
    )
    parser.add_argument(
        "--residual-points", type=int, default=48,
        help="SAM2 points_per_side for residual pass (default: 48 — denser than default to catch small objects)",
    )
    parser.add_argument(
        "--no-refine",   action="store_true", help="Skip SAM2 boundary refinement of panoptic masks"
    )
    parser.add_argument(
        "--no-residual", action="store_true", help="Skip SAM2 residual pass (panoptic only)"
    )
    parser.add_argument(
        "--no-preview",  action="store_true", help="Skip saving preview PNG"
    )
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        print(f"✗ Image not found: {image_path}")
        sys.exit(1)

    image_stem = Path(image_path).stem
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = (
        os.path.abspath(args.output_dir) if args.output_dir
        else os.path.join(script_dir, "notebook", "images", image_stem)
    )

    device = get_device()
    print(f"\nInput    : {image_path}")
    print(f"Output   : {output_dir}")
    print(f"Device   : {device}")
    print(f"Min area : {args.min_area*100:.2f}%")
    print(f"Refine   : {'off' if args.no_refine   else 'on'}")
    print(f"Residual : {'off' if args.no_residual else 'on'  } (points_per_side={args.residual_points})\n")

    image_pil = Image.open(image_path).convert("RGB")
    image_rgb = np.array(image_pil)
    print(f"✓ Image loaded: {image_rgb.shape[1]}x{image_rgb.shape[0]}")

    # ── Step 1: Panoptic ──────────────────────────────────────────────────
    segments = run_panoptic(image_pil, device)

    before = len(segments)
    segments = [s for s in segments if s["area_frac"] >= args.min_area]
    filtered = before - len(segments)
    if filtered:
        print(f"  Filtered {filtered} tiny panoptic segments → {len(segments)} kept")

    # ── Step 2: Refine panoptic masks with SAM2 ───────────────────────────
    if not args.no_refine and segments:
        predictor = load_sam2_predictor(args.checkpoint, args.model_cfg, device)
        predictor.set_image(image_rgb)
        print(f"\nRefining {len(segments)} panoptic masks with SAM2...")
        for i, seg in enumerate(segments):
            pct = (i + 1) / len(segments)
            bar = ("█" * int(pct * 20)).ljust(20)
            print(f"\r  [{bar}] {i+1}/{len(segments)}", end="", flush=True)
            seg["mask"] = refine_mask_with_sam2(predictor, seg["mask"])
        print("\n✓ Refinement complete")

    # ── Step 3: SAM2 residual pass ────────────────────────────────────────
    residual_segments = []
    if not args.no_residual:
        # Union of all panoptic masks = already-covered region
        h, w = image_rgb.shape[:2]
        covered = np.zeros((h, w), dtype=bool)
        for seg in segments:
            covered |= seg["mask"].astype(bool)

        uncovered_frac = (~covered).sum() / (h * w)
        print(f"\nStep 3: SAM2 residual pass")
        print(f"  Panoptic covers {covered.sum()/(h*w)*100:.1f}% of image")

        if uncovered_frac < 0.01:
            print("  Panoptic covered >99% — skipping residual pass")
        else:
            generator = load_sam2_generator(
                args.checkpoint, args.model_cfg, device,
                points_per_side=args.residual_points,
                min_mask_region_area=int(h * w * args.min_area * 0.5),
            )
            residual_segments = run_residual_sam2(
                generator, image_rgb, covered, args.min_area
            )

    # ── Step 4: Merge + deduplicate ───────────────────────────────────────
    all_segments = segments + residual_segments
    print(f"\nMerging: {len(segments)} panoptic + {len(residual_segments)} residual = {len(all_segments)} total")
    all_segments = deduplicate(all_segments, iou_threshold=0.5)
    print(f"After deduplication: {len(all_segments)} segments")

    if not all_segments:
        print("✗ No segments remaining. Try lowering --min-area.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────
    save_output(image_path, image_rgb, all_segments, output_dir)

    if not args.no_preview:
        save_preview(image_rgb, all_segments, os.path.join(output_dir, "_preview.png"))

    print(f"\n{'='*60}")
    print("✓ Done! Run the 3D model with:")
    print(f"    python demo_multi3.py --image {os.path.join(output_dir, 'image.png')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
