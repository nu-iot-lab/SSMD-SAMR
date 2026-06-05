"""
SPIC Script 1: Component Generation
Generates segmentation maps for all input images and copies the original
image into the per-image output folder. Both are needed downstream.

Unlike the semantic-communication pipeline, SPIC does NOT use captions or
edge maps. The two transmitted components are:
    1. Segmentation map  (grayscale → lossless compressed)
    2. Coarse image      (downscaled original → lossy compressed)

Input:
    data/leftImg8bit/{split}/{city}/{image}_leftImg8bit.png

Output structure (one folder per image):
    spic_data/{image_name}/
        {image_name}.png               <- copy of original image
        {image_name}_seg_map.png       <- RGB colored segmentation map

Model used:
    Mask2Former (facebook/mask2former-swin-large-cityscapes-semantic)

Configuration:
    SELECTION_STRATEGY  – 'balanced' | 'random' | 'sequential' | 'val_only' | 'all'
    MAX_IMAGES          – int or None (None = all)
    SPLITS              – which Cityscapes splits to scan
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np
import random
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR          = Path("data/leftImg8bit")
OUTPUT_DIR         = Path("spic_data")
SPLITS             = ["val", "test", "train"]
SELECTION_STRATEGY = "balanced"   # 'balanced' | 'random' | 'sequential' | 'val_only' | 'all'
MAX_IMAGES         = None         # None = process all discovered images
RANDOM_SEED        = 42
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Cityscapes class → RGB color (for visualization)
CITYSCAPES_COLORS = {
    0:  (128,  64, 128),  # road
    1:  (244,  35, 232),  # sidewalk
    2:  ( 70,  70,  70),  # building
    3:  (102, 102, 156),  # wall
    4:  (190, 153, 153),  # fence
    5:  (153, 153, 153),  # pole
    6:  (250, 170,  30),  # traffic light
    7:  (220, 220,   0),  # traffic sign
    8:  (107, 142,  35),  # vegetation
    9:  (152, 251, 152),  # terrain
    10: ( 70, 130, 180),  # sky
    11: (220,  20,  60),  # person
    12: (255,   0,   0),  # rider
    13: (  0,   0, 142),  # car
    14: (  0,   0,  70),  # truck
    15: (  0,  60, 100),  # bus
    16: (  0,  80, 100),  # train
    17: (  0,   0, 230),  # motorcycle
    18: (119,  11,  32),  # bicycle
    255: (0,    0,   0),  # ignore
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def colorize_segmentation(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CITYSCAPES_COLORS.items():
        colored[mask == class_id] = color
    return colored


def discover_images() -> list[dict]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    images = []
    for split in SPLITS:
        split_dir = INPUT_DIR / split
        if not split_dir.exists():
            continue
        for city_dir in sorted(split_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                base_name = img_path.stem.replace("_leftImg8bit", "")
                images.append({
                    "base_name": base_name,
                    "path": img_path,
                    "split": split,
                    "city": city_dir.name,
                })

    print(f"Discovered {len(images)} images across {SPLITS}")
    return images


def select_images(images: list[dict]) -> list[dict]:
    random.seed(RANDOM_SEED)

    if SELECTION_STRATEGY == "all":
        selected = images
    elif SELECTION_STRATEGY == "val_only":
        selected = [i for i in images if i["split"] == "val"]
    elif SELECTION_STRATEGY == "test_only":
        selected = [i for i in images if i["split"] == "test"]
    elif SELECTION_STRATEGY == "sequential":
        selected = images[:MAX_IMAGES] if MAX_IMAGES else images
    elif SELECTION_STRATEGY == "random":
        selected = random.sample(images, min(MAX_IMAGES or len(images), len(images)))
    elif SELECTION_STRATEGY == "balanced":
        by_city: dict[str, list] = {}
        for img in images:
            by_city.setdefault(img["city"], []).append(img)
        per_city = max(1, (MAX_IMAGES or len(images)) // len(by_city)) if by_city else 1
        selected = []
        for city_imgs in by_city.values():
            selected.extend(random.sample(city_imgs, min(per_city, len(city_imgs))))
        if MAX_IMAGES and len(selected) > MAX_IMAGES:
            selected = random.sample(selected, MAX_IMAGES)
    else:
        selected = images[:MAX_IMAGES] if MAX_IMAGES else images

    print(f"Selected {len(selected)} images (strategy: {SELECTION_STRATEGY})")
    return selected


def already_processed(base_name: str) -> bool:
    folder = OUTPUT_DIR / base_name
    return (
        (folder / f"{base_name}.png").exists() and
        (folder / f"{base_name}_seg_map.png").exists()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -- Discover and select images --
    all_images = discover_images()
    selected = select_images(all_images)

    # -- Load Mask2Former --
    print("Loading Mask2Former...")
    seg_processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    ).to(DEVICE).eval()

    print(f"\nGenerating components for {len(selected)} images → {OUTPUT_DIR}/")
    print("=" * 60)

    stats = {"success": 0, "skipped": 0, "errors": 0}

    for i, img_info in enumerate(selected, 1):
        base_name = img_info["base_name"]
        print(f"\n[{i}/{len(selected)}] {base_name}")

        if already_processed(base_name):
            print("  Skipping (already processed)")
            stats["skipped"] += 1
            continue

        try:
            t0 = time.time()
            folder = OUTPUT_DIR / base_name
            folder.mkdir(exist_ok=True)

            # Copy original image into folder
            dest_orig = folder / f"{base_name}.png"
            if not dest_orig.exists():
                shutil.copy2(img_info["path"], dest_orig)

            # Generate segmentation map
            image = Image.open(img_info["path"]).convert("RGB")
            inputs = seg_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = seg_model(**inputs)
            pred_mask = seg_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0].cpu().numpy().astype(np.uint8)

            colored = colorize_segmentation(pred_mask)
            Image.fromarray(colored).save(folder / f"{base_name}_seg_map.png")

            print(f"  Done in {time.time()-t0:.1f}s")
            stats["success"] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            stats["errors"] += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save manifest
    manifest = {
        "strategy": SELECTION_STRATEGY,
        "total_selected": len(selected),
        "stats": stats,
        "images": [{"base_name": i["base_name"], "split": i["split"], "city": i["city"]}
                   for i in selected],
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Done. Success: {stats['success']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
