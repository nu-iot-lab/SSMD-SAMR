"""
Script 1: Component Generation
Generates segmentation maps, edge maps, and captions for all input images.

Input:
    data/leftImg8bit/{split}/{city}/{image}_leftImg8bit.png

Output structure (one folder per image):
    all_data_generated/{image_name}/
        {image_name}_seg_map.png        <- RGB colored segmentation map
        {image_name}_edge_map.png       <- Canny edge map (RGB)
        {image_name}_captions.json      <- BLIP and Gemma captions

Models used:
    - Mask2Former (facebook/mask2former-swin-large-cityscapes-semantic)
    - BLIP (Salesforce/blip-image-captioning-base)
    - Gemma 3n (google/gemma-3n-e2b-it) via HuggingFace Transformers
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    Mask2FormerForUniversalSegmentation,
    BlipProcessor,
    BlipForConditionalGeneration,
    Gemma3nForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_DIR = Path("data/leftImg8bit")
OUTPUT_DIR = Path("all_data_generated")
TARGET_SIZE = (2048, 1024)   # (width, height) for all output maps
MAX_IMAGES = None             # Set to an int to limit processing, None = all
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cityscapes class → RGB color
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


def generate_edge_map(image: Image.Image, target_size: tuple) -> Image.Image:
    """Canny edge detection on the original image, returned as RGB PIL image."""
    img_resized = image.resize(target_size, Image.LANCZOS)
    gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


class Gemma3VisionCaptioner:
    """Local vision captioner using google/gemma-3n-e2b-it."""

    MODEL_ID = "google/gemma-3n-e2b-it"

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self):
        print("Loading Gemma 3n...")
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        print("Gemma 3n loaded")

    def generate_caption(self, image: Image.Image, max_tokens: int = 60) -> str:
        if self.model is None:
            self.load()
        if self.model is None:
            return "A street scene with various urban elements."

        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Describe this street scene briefly in under {max_tokens} words."},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        caption = self.processor.decode(generation, skip_special_tokens=True).strip()
        return caption if len(caption) >= 5 else "A street scene with various urban elements."


def discover_images() -> list[dict]:
    """Recursively find all *_leftImg8bit.png files under INPUT_DIR."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    image_files = []
    for path in sorted(INPUT_DIR.rglob("*_leftImg8bit.png")):
        base_name = path.stem.replace("_leftImg8bit", "")
        relative = path.relative_to(INPUT_DIR)
        split = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        image_files.append({
            "base_name": base_name,
            "path": path,
            "split": split,
        })

    print(f"Found {len(image_files)} images")
    return image_files


def already_processed(base_name: str) -> bool:
    folder = OUTPUT_DIR / base_name
    required = [
        folder / f"{base_name}_seg_map.png",
        folder / f"{base_name}_edge_map.png",
        folder / f"{base_name}_captions.json",
    ]
    return all(f.exists() for f in required)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---- Load models ----
    print("Loading Mask2Former...")
    seg_processor = AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    )
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-large-cityscapes-semantic"
    ).to(DEVICE).eval()

    print("Loading BLIP...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)

    gemma = Gemma3VisionCaptioner()
    gemma.load()

    # ---- Discover images ----
    images = discover_images()
    if MAX_IMAGES:
        images = images[:MAX_IMAGES]

    print(f"\nProcessing {len(images)} images → {OUTPUT_DIR}/")
    print("=" * 60)

    stats = {"success": 0, "skipped": 0, "errors": 0}

    for i, img_info in enumerate(images, 1):
        base_name = img_info["base_name"]
        print(f"\n[{i}/{len(images)}] {base_name}")

        if already_processed(base_name):
            print("  Skipping (already processed)")
            stats["skipped"] += 1
            continue

        try:
            t0 = time.time()
            image = Image.open(img_info["path"]).convert("RGB")
            folder = OUTPUT_DIR / base_name
            folder.mkdir(exist_ok=True)

            # -- Segmentation map --
            inputs = seg_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = seg_model(**inputs)
            pred_mask = seg_processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0].cpu().numpy().astype(np.uint8)

            colored = colorize_segmentation(pred_mask)
            seg_pil = Image.fromarray(colored).resize(TARGET_SIZE, Image.NEAREST)
            seg_pil.save(folder / f"{base_name}_seg_map.png")

            # -- Edge map --
            edge_pil = generate_edge_map(image, TARGET_SIZE)
            edge_pil.save(folder / f"{base_name}_edge_map.png")

            # -- Captions --
            image_for_caption = image.resize((384, 384))
            blip_inputs = blip_processor(image_for_caption, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                blip_ids = blip_model.generate(**blip_inputs)
            blip_text = blip_processor.decode(blip_ids[0], skip_special_tokens=True)

            gemma_text = gemma.generate_caption(image)

            captions = {
                "image_name": base_name,
                "captions": {
                    "blip": {"text": blip_text},
                    "gemma": {"text": gemma_text},
                },
            }
            with open(folder / f"{base_name}_captions.json", "w", encoding="utf-8") as f:
                json.dump(captions, f, indent=2, ensure_ascii=False)

            print(f"  Done in {time.time()-t0:.1f}s | BLIP: {blip_text[:60]}")
            stats["success"] += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            stats["errors"] += 1

        # Free GPU memory each iteration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"Done. Success: {stats['success']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
