"""
SPIC Script 3: Image Reconstruction (Receiver/Decoder)
Reconstructs high-quality images from the two transmitted SPIC components.

What is received:
    1. Compressed segmentation map  (lossless) → decompressed → RGB for ControlNet
    2. Compressed coarse image      (lossy)    → decompressed → upscaled as init

Reconstruction modes:
    use_coarse_init=True  (default)
        StableDiffusionControlNetImg2ImgPipeline
        Coarse image initialises the diffusion process (strength=0.7).
        Segmentation map conditions ControlNet.

    use_coarse_init=False
        StableDiffusionControlNetPipeline
        Segmentation map conditions ControlNet.
        Coarse image is NOT used (ablation mode).

No caption is transmitted. A fixed generic prompt is used locally at the
receiver to satisfy the diffusion model text input requirement.

Input (produced by 01 and 02):
    spic_data/{image_name}/{image_name}_seg_compressed.png/.flif
    spic_data/{image_name}/{image_name}_coarse_compressed.jpg/.bpg

Output:
    spic_data/{image_name}/{image_name}_spic_reconstructed.png
    spic_reconstructed/{image_name}_spic_reconstructed.png
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR         = Path("spic_data")
OUTPUT_DIR       = Path("spic_reconstructed")
TARGET_SIZE      = (2048, 1024)   # (width, height) for output
USE_COARSE_INIT  = True           # True = Img2Img; False = seg-only (ablation)
NUM_STEPS        = 30
GUIDANCE_SCALE   = 7.5
STRENGTH         = 0.7            # Img2Img strength (only used when USE_COARSE_INIT=True)
SEED             = 42
MAX_IMAGES       = None
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DTYPE      = torch.float16 if torch.cuda.is_available() else torch.float32

# Fixed receiver-side prompt (not transmitted)
RECEIVER_PROMPT = "a photorealistic urban street scene with buildings, roads, and vehicles"

# ---------------------------------------------------------------------------
# Decompression helpers
# ---------------------------------------------------------------------------

def decompress_segmentation(folder: Path, base_name: str) -> Image.Image | None:
    """Load compressed seg map. Handles both FLIF and PNG fallback."""
    for ext in [".png", ".flif"]:
        path = folder / f"{base_name}_seg_compressed{ext}"
        if not path.exists():
            continue
        if ext == ".flif":
            out = path.with_suffix(".flif_dec.png")
            result = subprocess.run(["flif", "-d", str(path), str(out)],
                                    capture_output=True)
            if result.returncode == 0:
                img = Image.open(out)
                out.unlink()
                return img
        else:
            return Image.open(path)
    return None


def decompress_coarse(folder: Path, base_name: str) -> Image.Image | None:
    """Load compressed coarse image. Handles BPG and JPEG fallback."""
    for ext in [".jpg", ".jpeg", ".bpg"]:
        path = folder / f"{base_name}_coarse_compressed{ext}"
        if not path.exists():
            continue
        if ext == ".bpg":
            out = path.with_suffix(".bpg_dec.png")
            result = subprocess.run(["bpgdec", "-o", str(out), str(path)],
                                    capture_output=True)
            if result.returncode == 0:
                img = Image.open(out)
                out.unlink()
                return img
        else:
            return Image.open(path)
    return None


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def load_pipeline():
    print("Loading ControlNet (sd-controlnet-seg)...")
    seg_controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg",
        torch_dtype=MODEL_DTYPE,
    )

    if USE_COARSE_INIT:
        print("Loading Img2Img pipeline (coarse image as init)...")
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=seg_controlnet,
            safety_checker=None,
            torch_dtype=MODEL_DTYPE,
            variant="fp16" if MODEL_DTYPE == torch.float16 else None,
        )
    else:
        print("Loading ControlNet pipeline (seg-only mode)...")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=seg_controlnet,
            safety_checker=None,
            torch_dtype=MODEL_DTYPE,
            variant="fp16" if MODEL_DTYPE == torch.float16 else None,
        )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(DEVICE)
    pipe.vae          = pipe.vae.to(MODEL_DTYPE)
    pipe.unet         = pipe.unet.to(MODEL_DTYPE)
    pipe.text_encoder = pipe.text_encoder.to(MODEL_DTYPE)

    print(f"Pipeline ready on {DEVICE} ({MODEL_DTYPE})")
    return pipe


def reconstruct(pipe, seg_map: Image.Image, coarse_img: Image.Image) -> Image.Image:
    seg_ctrl  = seg_map.convert("RGB").resize(TARGET_SIZE, Image.NEAREST)
    coarse_up = coarse_img.convert("RGB").resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    with torch.no_grad():
        if USE_COARSE_INIT:
            result = pipe(
                prompt=RECEIVER_PROMPT,
                image=coarse_up,
                control_image=seg_ctrl,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                controlnet_conditioning_scale=1.0,
                strength=STRENGTH,
                generator=generator,
            )
        else:
            result = pipe(
                prompt=RECEIVER_PROMPT,
                image=seg_ctrl,
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
        has_seg    = any((folder / f"{name}_seg_compressed{e}").exists()
                        for e in [".png", ".flif"])
        has_coarse = any((folder / f"{name}_coarse_compressed{e}").exists()
                        for e in [".jpg", ".jpeg", ".bpg"])
        if has_seg and has_coarse:
            items.append({"base_name": name, "folder": folder})

    print(f"Found {len(items)} items ready for reconstruction")
    return items


def already_reconstructed(folder: Path, base_name: str) -> bool:
    return (folder / f"{base_name}_spic_reconstructed.png").exists()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    items = discover_items()
    if MAX_IMAGES:
        items = items[:MAX_IMAGES]

    pipe = load_pipeline()

    mode_str = "Img2Img + ControlNet" if USE_COARSE_INIT else "ControlNet seg-only"
    print(f"\nReconstructing {len(items)} images | Mode: {mode_str}")
    print(f"Individual: {DATA_DIR}/{{name}}/  |  Centralized: {OUTPUT_DIR}/")
    print("=" * 60)

    stats = {"success": 0, "errors": 0}

    for i, item in enumerate(items, 1):
        name   = item["base_name"]
        folder = item["folder"]
        print(f"\n[{i}/{len(items)}] {name}")

        if already_reconstructed(folder, name):
            print("  Skipping (already reconstructed)")
            continue

        t0 = time.time()
        try:
            seg_map    = decompress_segmentation(folder, name)
            coarse_img = decompress_coarse(folder, name)

            if seg_map is None or coarse_img is None:
                missing = [k for k, v in [("seg", seg_map), ("coarse", coarse_img)] if v is None]
                print(f"  ERROR: could not decompress {missing}")
                stats["errors"] += 1
                continue

            out_img = reconstruct(pipe, seg_map, coarse_img)

            # Save to per-image folder and centralized output
            out_img.save(folder  / f"{name}_spic_reconstructed.png")
            out_img.save(OUTPUT_DIR / f"{name}_spic_reconstructed.png")

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
    print(f"Centralized results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
