"""
Script 2: Component Preprocessing (Compression Pipeline)
Applies WebP-based compression to segmentation maps and edge maps,
simulating the semantic communication channel.

Compression strategies:
    Segmentation map : RGB → downsample to 1024×512 → WebP lossless → upsample to 2048×1024
    Edge map         : grayscale → binary threshold → WebP lossless at 2048×1024 → RGB

Input (produced by 01_generate_components.py):
    all_data_generated/{image_name}/{image_name}_seg_map.png
    all_data_generated/{image_name}/{image_name}_edge_map.png
    all_data_generated/{image_name}/{image_name}_captions.json

Output (written into the same per-image folder):
    {image_name}_seg_map_compressed.webp
    {image_name}_seg_map_decompressed.png
    {image_name}_edge_map_compressed.webp
    {image_name}_edge_map_decompressed.png
    {image_name}_compression_stats.json
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("all_data_generated")
SEG_DOWNSAMPLE_SIZE = (1024, 512)   # intermediate size for seg compression
TARGET_SIZE = (2048, 1024)          # final (width, height) for both maps
EDGE_BINARY_THRESHOLD = 128

# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def compress_segmentation(seg_path: Path, base_name: str) -> dict:
    """
    Compress segmentation map: RGB downsample → WebP → upsample.
    Returns dict with file sizes and paths.
    """
    seg_image = Image.open(seg_path).convert("RGB")
    if seg_image.size != TARGET_SIZE:
        seg_image = seg_image.resize(TARGET_SIZE, Image.NEAREST)

    # Compress: downsample to half resolution
    compressed = seg_image.resize(SEG_DOWNSAMPLE_SIZE, Image.NEAREST)

    # Save WebP (simulates channel transmission)
    compressed_path = seg_path.parent / f"{base_name}_seg_map_compressed.webp"
    compressed.save(compressed_path, format="WebP", lossless=True, quality=100)
    webp_size = compressed_path.stat().st_size

    # Decompress: reload and upsample back
    decompressed = Image.open(compressed_path).convert("RGB")
    decompressed = decompressed.resize(TARGET_SIZE, Image.NEAREST)
    decompressed_path = seg_path.parent / f"{base_name}_seg_map_decompressed.png"
    decompressed.save(decompressed_path)

    original_size = seg_path.stat().st_size
    return {
        "original_size_bytes": original_size,
        "compressed_size_bytes": webp_size,
        "reduction_pct": round((1 - webp_size / original_size) * 100, 1),
        "compressed_path": str(compressed_path),
        "decompressed_path": str(decompressed_path),
    }


def compress_edge(edge_path: Path, base_name: str) -> dict:
    """
    Compress edge map: grayscale → binary → WebP → RGB.
    Returns dict with file sizes and paths.
    """
    edge_image = Image.open(edge_path)
    if edge_image.size != TARGET_SIZE:
        edge_image = edge_image.resize(TARGET_SIZE, Image.LANCZOS)

    # Binarize
    gray = np.array(edge_image.convert("L"))
    binary = ((gray > EDGE_BINARY_THRESHOLD).astype(np.uint8) * 255)
    binary_pil = Image.fromarray(binary, mode="L")

    # Save WebP
    compressed_path = edge_path.parent / f"{base_name}_edge_map_compressed.webp"
    binary_pil.save(compressed_path, format="WebP", lossless=True, quality=100)
    webp_size = compressed_path.stat().st_size

    # Decompress: reload and convert to RGB
    decompressed = Image.open(compressed_path).convert("RGB")
    decompressed_path = edge_path.parent / f"{base_name}_edge_map_decompressed.png"
    decompressed.save(decompressed_path)

    original_size = edge_path.stat().st_size
    return {
        "original_size_bytes": original_size,
        "compressed_size_bytes": webp_size,
        "reduction_pct": round((1 - webp_size / original_size) * 100, 1),
        "compressed_path": str(compressed_path),
        "decompressed_path": str(decompressed_path),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_items() -> list[dict]:
    """Find all image folders that have seg, edge, and captions."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    items = []
    for folder in sorted(DATA_DIR.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        seg = folder / f"{name}_seg_map.png"
        edge = folder / f"{name}_edge_map.png"
        caps = folder / f"{name}_captions.json"
        if seg.exists() and edge.exists() and caps.exists():
            items.append({"base_name": name, "folder": folder,
                          "seg": seg, "edge": edge})

    print(f"Found {len(items)} items ready for preprocessing")
    return items


def already_preprocessed(folder: Path, base_name: str) -> bool:
    return (
        (folder / f"{base_name}_seg_map_decompressed.png").exists() and
        (folder / f"{base_name}_edge_map_decompressed.png").exists()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    items = discover_items()
    stats_all = []

    print(f"\nPreprocessing {len(items)} items")
    print("=" * 60)

    for i, item in enumerate(items, 1):
        name = item["base_name"]
        print(f"\n[{i}/{len(items)}] {name}")

        if already_preprocessed(item["folder"], name):
            print("  Skipping (already preprocessed)")
            continue

        t0 = time.time()
        try:
            seg_stats = compress_segmentation(item["seg"], name)
            edge_stats = compress_edge(item["edge"], name)

            total_original = seg_stats["original_size_bytes"] + edge_stats["original_size_bytes"]
            total_compressed = seg_stats["compressed_size_bytes"] + edge_stats["compressed_size_bytes"]

            combined = {
                "image_name": name,
                "segmentation": seg_stats,
                "edge": edge_stats,
                "total_original_bytes": total_original,
                "total_compressed_bytes": total_compressed,
                "overall_reduction_pct": round((1 - total_compressed / total_original) * 100, 1),
            }

            # Save per-image stats
            stats_path = item["folder"] / f"{name}_compression_stats.json"
            with open(stats_path, "w") as f:
                json.dump(combined, f, indent=2)

            stats_all.append(combined)

            print(
                f"  Seg:  {seg_stats['original_size_bytes']//1024}KB → "
                f"{seg_stats['compressed_size_bytes']//1024}KB "
                f"({seg_stats['reduction_pct']}% reduction)"
            )
            print(
                f"  Edge: {edge_stats['original_size_bytes']//1024}KB → "
                f"{edge_stats['compressed_size_bytes']//1024}KB "
                f"({edge_stats['reduction_pct']}% reduction)"
            )
            print(f"  Done in {time.time()-t0:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    if stats_all:
        avg_reduction = sum(s["overall_reduction_pct"] for s in stats_all) / len(stats_all)
        avg_total_compressed_kb = sum(s["total_compressed_bytes"] for s in stats_all) / len(stats_all) / 1024
        print("\n" + "=" * 60)
        print(f"Preprocessed {len(stats_all)} items")
        print(f"Average total reduction: {avg_reduction:.1f}%")
        print(f"Average compressed payload: {avg_total_compressed_kb:.1f} KB per image")


if __name__ == "__main__":
    main()
