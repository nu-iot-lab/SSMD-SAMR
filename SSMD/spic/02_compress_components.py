"""
SPIC Script 2: Component Compression
Implements the two-stream SPIC transmission pipeline from the paper.

Stream 1 – Segmentation Map (SSM):
    RGB seg map → grayscale (class-to-gray mapping) → FLIF lossless
    (Falls back to PNG level-9 if FLIF is not installed)
    Paper target: ~0.112 BPP

Stream 2 – Coarse Image:
    Original → 4× downscale (2048×1024 → 512×256) → BPG lossy
    (Falls back to JPEG if BPG is not installed)

Input (produced by 01_generate_components.py):
    spic_data/{image_name}/{image_name}.png
    spic_data/{image_name}/{image_name}_seg_map.png

Output (written into the same per-image folder):
    {image_name}_seg_gray.png              <- grayscale seg map
    {image_name}_seg_compressed.png/.flif  <- lossless compressed seg
    {image_name}_coarse.png                <- downscaled original
    {image_name}_coarse_compressed.jpg/.bpg <- lossy compressed coarse
    {image_name}_spic_stats.json           <- compression statistics
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR        = Path("spic_data")
DOWNSCALE_FACTOR = 4       # 2048×1024 → 512×256
BPG_QUALITY      = 28      # BPG quality (0–51, lower = higher quality)
                           # Maps to JPEG ~46 when BPG unavailable

# Cityscapes train-ID → evenly distributed grayscale value
_N_CLASSES = 19
CLASS_TO_GRAY: dict[int, int] = {
    i: int((i / (_N_CLASSES - 1)) * 255) for i in range(_N_CLASSES)
}
CLASS_TO_GRAY[255] = 0   # void → black

# Cityscapes RGB palette (index = train-ID)
CITYSCAPES_PALETTE = [
    (128,  64, 128), (244,  35, 232), ( 70,  70,  70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170,  30), (220, 220,   0),
    (107, 142,  35), (152, 251, 152), ( 70, 130, 180), (220,  20,  60),
    (255,   0,   0), (  0,   0, 142), (  0,   0,  70), (  0,  60, 100),
    (  0,  80, 100), (  0,   0, 230), (119,  11,  32),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flif_available() -> bool:
    try:
        return subprocess.run(["flif", "--version"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _bpg_available() -> bool:
    try:
        return subprocess.run(["bpgenc", "-h"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _file_stats(input_path: Path, output_path: Path) -> dict:
    in_size  = input_path.stat().st_size
    out_size = output_path.stat().st_size
    w, h     = Image.open(input_path).size
    bpp      = (out_size * 8) / (w * h)
    return {
        "input_kb":  round(in_size  / 1024, 2),
        "output_kb": round(out_size / 1024, 2),
        "ratio":     round(in_size  / out_size, 2) if out_size else 0,
        "bpp":       round(bpp, 4),
    }


def rgb_to_grayscale(seg_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB colored seg map → class IDs → grayscale values."""
    h, w = seg_rgb.shape[:2]
    class_map = np.zeros((h, w), dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        mask = np.all(seg_rgb == color, axis=2)
        class_map[mask] = class_id

    gray = np.zeros((h, w), dtype=np.uint8)
    for class_id, gray_val in CLASS_TO_GRAY.items():
        gray[class_map == class_id] = gray_val
    return gray


def compress_segmentation(folder: Path, base_name: str) -> dict:
    seg_path  = folder / f"{base_name}_seg_map.png"
    gray_path = folder / f"{base_name}_seg_gray.png"
    comp_stem = folder / f"{base_name}_seg_compressed"

    # RGB → grayscale
    seg_rgb = np.array(Image.open(seg_path).convert("RGB"))
    gray    = rgb_to_grayscale(seg_rgb)
    Image.fromarray(gray).save(gray_path)

    # Lossless compression
    if _flif_available():
        out_path = comp_stem.with_suffix(".flif")
        subprocess.run(["flif", "-e", str(gray_path), str(out_path)],
                       capture_output=True, check=True)
    else:
        out_path = comp_stem.with_suffix(".png")
        Image.open(gray_path).save(out_path, "PNG", compress_level=9, optimize=True)

    return _file_stats(gray_path, out_path)


def compress_coarse(folder: Path, base_name: str) -> dict:
    orig_path   = folder / f"{base_name}.png"
    coarse_path = folder / f"{base_name}_coarse.png"
    comp_stem   = folder / f"{base_name}_coarse_compressed"

    # Downscale
    img = Image.open(orig_path).convert("RGB")
    w, h = img.size
    coarse = img.resize((w // DOWNSCALE_FACTOR, h // DOWNSCALE_FACTOR),
                        Image.Resampling.LANCZOS)
    coarse.save(coarse_path, "PNG")

    # Lossy compression
    if _bpg_available():
        out_path = comp_stem.with_suffix(".bpg")
        subprocess.run(
            ["bpgenc", "-q", str(BPG_QUALITY), "-o", str(out_path), str(coarse_path)],
            capture_output=True, check=True,
        )
    else:
        out_path = comp_stem.with_suffix(".jpg")
        jpeg_q = max(5, min(95, int(95 - (BPG_QUALITY / 51) * 90)))
        coarse.save(out_path, "JPEG", quality=jpeg_q, optimize=True)

    return _file_stats(coarse_path, out_path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_items() -> list[dict]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    items = []
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if (folder / f"{name}.png").exists() and (folder / f"{name}_seg_map.png").exists():
            items.append({"base_name": name, "folder": folder})

    print(f"Found {len(items)} items ready for compression")
    return items


def already_compressed(folder: Path, base_name: str) -> bool:
    gray = (folder / f"{base_name}_seg_gray.png").exists()
    coarse = (folder / f"{base_name}_coarse.png").exists()
    return gray and coarse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    items = discover_items()
    all_stats = []

    print(f"\nCompressing {len(items)} items (downscale ×{DOWNSCALE_FACTOR}, BPG q={BPG_QUALITY})")
    print(f"FLIF available: {_flif_available()} | BPG available: {_bpg_available()}")
    print("=" * 60)

    for i, item in enumerate(items, 1):
        name   = item["base_name"]
        folder = item["folder"]
        print(f"\n[{i}/{len(items)}] {name}")

        if already_compressed(folder, name):
            print("  Skipping (already compressed)")
            continue

        t0 = time.time()
        try:
            seg_stats    = compress_segmentation(folder, name)
            coarse_stats = compress_coarse(folder, name)

            total_orig = (folder / f"{name}.png").stat().st_size
            total_comp = seg_stats["output_kb"] * 1024 + coarse_stats["output_kb"] * 1024

            combined = {
                "image_name":      name,
                "segmentation":    seg_stats,
                "coarse":          coarse_stats,
                "original_kb":     round(total_orig / 1024, 2),
                "total_compressed_kb": round((seg_stats["output_kb"] + coarse_stats["output_kb"]), 2),
                "overall_ratio":   round(total_orig / total_comp, 2) if total_comp else 0,
            }
            with open(folder / f"{name}_spic_stats.json", "w") as f:
                json.dump(combined, f, indent=2)

            all_stats.append(combined)

            print(f"  Seg:    {seg_stats['input_kb']}KB → {seg_stats['output_kb']}KB "
                  f"({seg_stats['bpp']} BPP, ratio {seg_stats['ratio']}×)")
            print(f"  Coarse: {coarse_stats['input_kb']}KB → {coarse_stats['output_kb']}KB "
                  f"({coarse_stats['bpp']} BPP, ratio {coarse_stats['ratio']}×)")
            print(f"  Total:  {combined['original_kb']}KB → {combined['total_compressed_kb']}KB "
                  f"(ratio {combined['overall_ratio']}×) | {time.time()-t0:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    if all_stats:
        avg_seg_bpp    = np.mean([s["segmentation"]["bpp"] for s in all_stats])
        avg_coarse_bpp = np.mean([s["coarse"]["bpp"] for s in all_stats])
        avg_ratio      = np.mean([s["overall_ratio"] for s in all_stats])
        print("\n" + "=" * 60)
        print(f"Compressed {len(all_stats)} items")
        print(f"Avg seg BPP:    {avg_seg_bpp:.4f}  (paper target ~0.112)")
        print(f"Avg coarse BPP: {avg_coarse_bpp:.4f}")
        print(f"Avg overall compression ratio: {avg_ratio:.2f}×")


if __name__ == "__main__":
    main()
