"""
Ablation: No Caption (Empty Prompt)
Reconstructs images using edge map + segmentation map with ControlNet,
but passes an EMPTY string as the text prompt.

Purpose:
    Isolates the contribution of the caption component. Comparing these
    results against the full pipeline (03_reconstruct_images.py) shows
    how much the BLIP/Gemma caption improves reconstruction quality.

Components used:      seg map + edge map  (same WebP compression as full pipeline)
Components removed:   caption             (replaced with empty string "")

Input (produced by main pipeline scripts 01 + 02):
    all_data_generated/{image_name}/{image_name}_seg_map_decompressed.png
    all_data_generated/{image_name}/{image_name}_edge_map_decompressed.png

Output:
    all_data_generated/{image_name}/{image_name}_no_caption_reconstructed.png
    ablation_no_caption/{image_name}_no_caption_reconstructed.png
"""

import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR         = Path("all_data_generated")
OUTPUT_DIR       = Path("ablation_no_caption")
TARGET_SIZE      = (2048, 1024)
NUM_STEPS        = 20
GUIDANCE_SCALE   = 7.5
SEED             = 42
MAX_IMAGES       = None
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE      = torch.float16 if torch.cuda.is_available() else torch.float32

EMPTY_PROMPT     = ""   # ablation: no text guidance

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_pipeline() -> StableDiffusionControlNetPipeline:
    print("Loading ControlNet models (canny + seg)...")
    edge_cn = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny", torch_dtype=MODEL_DTYPE
    )
    seg_cn = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg", torch_dtype=MODEL_DTYPE
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[edge_cn, seg_cn],
        safety_checker=None,
        torch_dtype=MODEL_DTYPE,
        variant="fp16" if MODEL_DTYPE == torch.float16 else None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.vae = pipe.vae.to(MODEL_DTYPE)
    pipe.unet = pipe.unet.to(MODEL_DTYPE)
    pipe.text_encoder = pipe.text_encoder.to(MODEL_DTYPE)
    print(f"Pipeline ready on {DEVICE} ({MODEL_DTYPE})")
    return pipe


def generate(pipe, edge_map: Image.Image, seg_map: Image.Image) -> Image.Image:
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    with torch.no_grad():
        result = pipe(
            prompt=EMPTY_PROMPT,
            image=[edge_map, seg_map],   # order: edge first, then seg
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=TARGET_SIZE[1],
            width=TARGET_SIZE[0],
            generator=generator,
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result.images[0]


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
        seg  = folder / f"{name}_seg_map_decompressed.png"
        edge = folder / f"{name}_edge_map_decompressed.png"
        if seg.exists() and edge.exists():
            items.append({"base_name": name, "folder": folder, "seg": seg, "edge": edge})

    print(f"Found {len(items)} items ready for ablation reconstruction")
    return items


def already_done(folder: Path, name: str) -> bool:
    return (folder / f"{name}_no_caption_reconstructed.png").exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    items = discover_items()
    if MAX_IMAGES:
        items = items[:MAX_IMAGES]

    pipe = load_pipeline()

    print(f"\nAblation: No Caption | {len(items)} images")
    print(f"Prompt: empty string  |  ControlNets: edge + seg")
    print("=" * 60)

    stats = {"success": 0, "errors": 0}

    for i, item in enumerate(items, 1):
        name   = item["base_name"]
        folder = item["folder"]
        print(f"\n[{i}/{len(items)}] {name}")

        if already_done(folder, name):
            print("  Skipping (already done)")
            continue

        t0 = time.time()
        try:
            seg_map  = Image.open(item["seg"]).convert("RGB")
            edge_map = Image.open(item["edge"]).convert("RGB")

            if seg_map.size  != TARGET_SIZE:
                seg_map  = seg_map.resize(TARGET_SIZE, Image.NEAREST)
            if edge_map.size != TARGET_SIZE:
                edge_map = edge_map.resize(TARGET_SIZE, Image.LANCZOS)

            out_img = generate(pipe, edge_map, seg_map)
            out_img.save(folder    / f"{name}_no_caption_reconstructed.png")
            out_img.save(OUTPUT_DIR / f"{name}_no_caption_reconstructed.png")

            print(f"  Done in {time.time()-t0:.1f}s")
            stats["success"] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            stats["errors"] += 1

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"Done. Success: {stats['success']} | Errors: {stats['errors']}")
    print(f"Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
