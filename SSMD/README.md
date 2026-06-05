# Semantic Communication Pipeline

End-to-end pipeline for semantic image compression and reconstruction using
segmentation maps, edge maps, and text captions as the transmitted representation.

---

## Overview

The pipeline simulates a semantic communication channel in three stages:

```
[Original Image]
       │
       ▼
01_generate_components.py     ← segmentation map, edge map, captions
       │
       ▼
02_preprocess_components.py   ← WebP compression (simulates channel transmission)
       │
       ▼
03_reconstruct_images.py      ← ControlNet reconstruction at receiver
       │
       ▼
[Reconstructed Image]
```

Each image is transmitted as three compact components:
| Component | Compression | Approx. size |
|-----------|-------------|-------------|
| Segmentation map | RGB → downsample 1024×512 → WebP lossless → upsample | ~3–4 KB |
| Edge map | binary → WebP lossless at 2048×1024 | ~3–4 KB |
| Caption | UTF-8 text string | ~100–200 B |

---

## Requirements

### Python packages

```bash
pip install torch torchvision
pip install transformers diffusers accelerate
pip install opencv-python pillow numpy
```

> Gemma 3n requires `transformers >= 4.50`. Upgrade if needed:
> ```bash
> pip install --upgrade transformers
> ```

### Models downloaded automatically on first run

- `facebook/mask2former-swin-large-cityscapes-semantic`
- `Salesforce/blip-image-captioning-base`
- `google/gemma-3n-e2b-it`
- `lllyasviel/sd-controlnet-canny`
- `lllyasviel/sd-controlnet-seg`
- `runwayml/stable-diffusion-v1-5`

### Hardware

- GPU strongly recommended (RTX 3060+ or equivalent)
- ~16 GB VRAM for all three models loaded simultaneously
- Script 01 loads Mask2Former + BLIP + Gemma 3n sequentially and clears GPU memory between images

---

## Dataset

The pipeline expects the **Cityscapes** dataset in its standard layout:

```
data/
└── leftImg8bit/
    ├── train/
    │   └── {city}/{city}_{seq}_{frame}_leftImg8bit.png
    ├── val/
    │   └── {city}/{city}_{seq}_{frame}_leftImg8bit.png
    └── test/
        └── {city}/{city}_{seq}_{frame}_leftImg8bit.png
```

---

## Running the Pipeline

Run all three scripts from the `SSMD/` directory (or adjust paths in each script).

### Step 1 — Generate components

```bash
python 01_generate_components.py
```

Produces for every image:

```
all_data_generated/{image_name}/
    {image_name}_seg_map.png        ← RGB colored segmentation map (2048×1024)
    {image_name}_edge_map.png       ← Canny edge map as RGB (2048×1024)
    {image_name}_captions.json      ← BLIP and Gemma 3n captions
```

**Configuration options** (top of the file):

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `data/leftImg8bit` | Path to Cityscapes images |
| `OUTPUT_DIR` | `all_data_generated` | Output root folder |
| `TARGET_SIZE` | `(2048, 1024)` | Output resolution (width, height) |
| `MAX_IMAGES` | `None` | Limit number of images; `None` = all |

Already-processed images are automatically skipped on re-runs.

---

### Step 2 — Compress components

```bash
python 02_preprocess_components.py
```

Reads from `all_data_generated/` and writes into the same per-image folders:

```
{image_name}_seg_map_compressed.webp      ← transmitted segmentation
{image_name}_seg_map_decompressed.png     ← reconstructed at receiver
{image_name}_edge_map_compressed.webp     ← transmitted edge map
{image_name}_edge_map_decompressed.png    ← reconstructed at receiver
{image_name}_compression_stats.json      ← file sizes, reduction %
```

**Compression strategies applied:**

- **Segmentation**: RGB → downsample to 1024×512 → WebP lossless → upsample back to 2048×1024 (~75–80% size reduction)
- **Edge map**: grayscale → binary threshold (>128) → WebP lossless at full 2048×1024 → RGB (~90–93% size reduction)

**Configuration options** (top of the file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `all_data_generated` | Input/output root folder |
| `SEG_DOWNSAMPLE_SIZE` | `(1024, 512)` | Intermediate size for seg compression |
| `TARGET_SIZE` | `(2048, 1024)` | Final resolution after decompression |
| `EDGE_BINARY_THRESHOLD` | `128` | Pixel threshold for edge binarization |

---

### Step 3 — Reconstruct images

```bash
python 03_reconstruct_images.py
```

Reads the decompressed maps and captions, runs ControlNet inference, and saves:

```
all_data_generated/{image_name}/
    {image_name}_blip_reconstructed.png
    {image_name}_gemma_reconstructed.png

generated_final_images/
    {image_name}_blip_reconstructed.png    ← centralized copy
    {image_name}_gemma_reconstructed.png   ← centralized copy
```

Two reconstructions are produced per image — one conditioned on the BLIP caption and one on the Gemma 3n caption.

**Configuration options** (top of the file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `all_data_generated` | Input folder |
| `OUTPUT_DIR` | `generated_final_images` | Centralized output folder |
| `TARGET_SIZE` | `(2048, 1024)` | Generation resolution |
| `NUM_INFERENCE_STEPS` | `20` | Diffusion steps (higher = better quality, slower) |
| `GUIDANCE_SCALE` | `7.5` | ControlNet guidance strength |
| `SEED` | `42` | Random seed for reproducibility |
| `MAX_IMAGES` | `None` | Limit number of images; `None` = all |

---

## Output Structure (after all three steps)

```
all_data_generated/
└── {image_name}/
    ├── {image_name}_seg_map.png
    ├── {image_name}_seg_map_compressed.webp
    ├── {image_name}_seg_map_decompressed.png
    ├── {image_name}_edge_map.png
    ├── {image_name}_edge_map_compressed.webp
    ├── {image_name}_edge_map_decompressed.png
    ├── {image_name}_captions.json
    ├── {image_name}_compression_stats.json
    ├── {image_name}_blip_reconstructed.png
    └── {image_name}_gemma_reconstructed.png

generated_final_images/
    ├── {image_name}_blip_reconstructed.png
    └── {image_name}_gemma_reconstructed.png
```

---

## Tips

**Test on a small subset first** — set `MAX_IMAGES = 5` in scripts 01 and 03 before a full run.

**Resuming interrupted runs** — all three scripts skip images that already have their output files, so you can safely re-run after an interruption.

**Checking compression stats** — each `_compression_stats.json` contains original size, compressed size, and reduction percentage for both the seg map and edge map.
