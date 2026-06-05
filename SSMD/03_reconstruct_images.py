"""
Script 3: Image Reconstruction
Reconstructs original images from the three semantic components using ControlNet.

Uses:
    - Decompressed segmentation map  → sd-controlnet-seg
    - Decompressed edge map          → sd-controlnet-canny
    - Text caption (BLIP or Gemma)   → Stable Diffusion prompt

Input (produced by scripts 01 and 02):
    all_data_generated/{image_name}/{image_name}_seg_map_decompressed.png
    all_data_generated/{image_name}/{image_name}_edge_map_decompressed.png
    all_data_generated/{image_name}/{image_name}_captions.json

Output:
    all_data_generated/{image_name}/{image_name}_blip_reconstructed.png
    all_data_generated/{image_name}/{image_name}_gemma_reconstructed.png
    generated_final_images/{image_name}_blip_reconstructed.png
    generated_final_images/{image_name}_gemma_reconstructed.png
"""

import json
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
DATA_DIR = Path("all_data_generated")
OUTPUT_DIR = Path("generated_final_images")
TARGET_SIZE = (2048, 1024)          # (width, height) for generation
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5
SEED = 42
MAX_IMAGES = None                   # Set to int to limit, None = all
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def load_pipeline() -> StableDiffusionControlNetPipeline:
    print("Loading ControlNet models...")
    edge_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=MODEL_DTYPE,
    )
    seg_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg",
        torch_dtype=MODEL_DTYPE,
    )

    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=[edge_controlnet, seg_controlnet],
        safety_checker=None,
        torch_dtype=MODEL_DTYPE,
        variant="fp16" if MODEL_DTYPE == torch.float16 else None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.vae = pipe.vae.to(MODEL_DTYPE)
    pipe.unet = pipe.unet.to(MODEL_DTYPE)
    pipe.text_encoder = pipe.text_encoder.to(MODEL_DTYPE)
    print(f"Pipeline loaded on {DEVICE} ({MODEL_DTYPE})")
    return pipe


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    pipe: StableDiffusionControlNetPipeline,
    edge_map: Image.Image,
    seg_map: Image.Image,
    prompt: str,
) -> Image.Image:
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=[edge_map, seg_map],   # order: edge first, then seg
            num_inference_steps=NUM_INFERENCE_STEPS,
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
        seg = folder / f"{name}_seg_map_decompressed.png"
        edge = folder / f"{name}_edge_map_decompressed.png"
        caps = folder / f"{name}_captions.json"

        if not (seg.exists() and edge.exists() and caps.exists()):
            continue

        try:
            with open(caps, encoding="utf-8") as f:
                data = json.load(f)
            blip = data["captions"]["blip"]["text"]
            gemma = data["captions"]["gemma"]["text"]
            if blip and gemma:
                items.append({
                    "base_name": name,
                    "folder": folder,
                    "seg": seg,
                    "edge": edge,
                    "blip_caption": blip,
                    "gemma_caption": gemma,
                })
        except Exception:
            pass

    print(f"Found {len(items)} items ready for reconstruction")
    return items


def already_reconstructed(folder: Path, base_name: str) -> bool:
    return (
        (folder / f"{base_name}_blip_reconstructed.png").exists() and
        (folder / f"{base_name}_gemma_reconstructed.png").exists()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    items = discover_items()
    if MAX_IMAGES:
        items = items[:MAX_IMAGES]

    pipe = load_pipeline()

    print(f"\nReconstructing {len(items)} images")
    print(f"Individual results: {DATA_DIR}/{{name}}/")
    print(f"Centralized results: {OUTPUT_DIR}/")
    print("=" * 60)

    stats = {"success": 0, "errors": 0}

    for i, item in enumerate(items, 1):
        name = item["base_name"]
        print(f"\n[{i}/{len(items)}] {name}")

        if already_reconstructed(item["folder"], name):
            print("  Skipping (already reconstructed)")
            continue

        t0 = time.time()
        try:
            seg_map = Image.open(item["seg"]).convert("RGB")
            edge_map = Image.open(item["edge"]).convert("RGB")

            if seg_map.size != TARGET_SIZE:
                seg_map = seg_map.resize(TARGET_SIZE, Image.NEAREST)
            if edge_map.size != TARGET_SIZE:
                edge_map = edge_map.resize(TARGET_SIZE, Image.LANCZOS)

            for caption_type in ("blip", "gemma"):
                caption = item[f"{caption_type}_caption"]
                print(f"  [{caption_type}] {caption[:70]}")

                out_img = generate(pipe, edge_map, seg_map, caption)

                # Save to individual folder
                local_path = item["folder"] / f"{name}_{caption_type}_reconstructed.png"
                out_img.save(local_path)

                # Save to centralized folder
                central_path = OUTPUT_DIR / f"{name}_{caption_type}_reconstructed.png"
                out_img.save(central_path)

            print(f"  Done in {time.time()-t0:.1f}s")
            stats["success"] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            stats["errors"] += 1

    # Cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"Done. Success: {stats['success']} | Errors: {stats['errors']}")
    print(f"Centralized results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
