"""
Local test runner — no Triton needed.
Supports both SAU (full-body sketch) and FRU (face sketch) pipelines.

Usage:
    # SAU — full-body sketch (default)
    python local_test.py --image person.jpg --id A1234 --gender female

    # FRU — face sketch
    python local_test.py --image person.jpg --id A1234 --gender female --place FRU

    # SAU with gender auto-detection
    python local_test.py --image person.jpg --id A1234

    # FRU with gender auto-detection
    python local_test.py --image person.jpg --id A1234 --place FRU
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

# ── Make sure project root is on path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


OUTPUT_DIR = Path("local_test_output")


# ============================================================================
# SAU — full-body sketch
# ============================================================================

def run_sau(image_path: str, person_id: str, gender: str):
    from pipeline.preprocessor import preprocess_image_in_memory
    from pipeline.generator import generate_sketch_in_memory

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Pipeline : SAU (full-body sketch)")
    print(f"Input    : {image_path}")
    print(f"ID       : {person_id}")
    print(f"Gender   : {gender or 'not provided — will auto-detect'}")
    print(f"{'='*60}\n")

    img = Image.open(image_path)

    # ── Preprocess ─────────────────────────────────────────────────────────
    print("[STEP 1/2] Preprocessing...")
    gender_override = gender if gender in ("male", "female") else None
    data = preprocess_image_in_memory(img, gender_override=gender_override)

    print(f"\n  Resolved gender : {data.gender}")
    print(f"  Original size   : {data.original_size}")
    print(f"  Face detected   : {data.primary_face is not None}")
    print(f"  Face embedding  : {data.has_face_embedding}")

    # Save preprocessing intermediates
    preprocess_dir = OUTPUT_DIR / f"{person_id}_sau_preprocess"
    preprocess_dir.mkdir(exist_ok=True)
    data.enhanced.save(preprocess_dir / "enhanced.jpg", quality=95)
    data.body_edges.save(preprocess_dir / "body_edges.png")
    if data.face_img is not None:
        data.face_img.save(preprocess_dir / "face_cropped.jpg", quality=95)
    if data.face_edges is not None:
        data.face_edges.save(preprocess_dir / "face_edges.png")
    print(f"\n  Preprocessing intermediates saved to: {preprocess_dir}/")

    # ── Generate ───────────────────────────────────────────────────────────
    print("\n[STEP 2/2] Generating sketches...")
    result = generate_sketch_in_memory(data)

    sketch_path = OUTPUT_DIR / f"{person_id}_sau_final_sketch.jpg"
    result.final_sketch.save(sketch_path, quality=95)
    print(f"\n  Final sketch saved : {sketch_path}")

    if result.face_sketch is not None:
        face_path = OUTPUT_DIR / f"{person_id}_sau_face_sketch.jpg"
        result.face_sketch.save(face_path, quality=95)
        print(f"  Face sketch saved  : {face_path}")

    if result.scene_images:
        for i, scene_img in enumerate(result.scene_images):
            scene_path = OUTPUT_DIR / f"{person_id}_sau_scene_{i:03d}.jpg"
            scene_img.save(scene_path, quality=95)
        print(f"  Scenes saved       : {len(result.scene_images)} image(s)")
    else:
        print("  No scenes (check inputs/scenes/ and crops.json)")


# ============================================================================
# FRU — face sketch
# ============================================================================

def run_fru(image_path: str, person_id: str, gender: str):
    from pipeline.fru_preprocessor import preprocess_fru_image_in_memory
    from pipeline.fru_generator import generate_fru_sketch_in_memory

    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Pipeline : FRU (face sketch)")
    print(f"Input    : {image_path}")
    print(f"ID       : {person_id}")
    print(f"Gender   : {gender or 'not provided — will auto-detect'}")
    print(f"{'='*60}\n")

    img = Image.open(image_path)

    # ── Preprocess ─────────────────────────────────────────────────────────
    print("[STEP 1/2] Preprocessing...")
    gender_override = gender if gender in ("male", "female") else None
    data = preprocess_fru_image_in_memory(img, gender_override=gender_override)

    print(f"\n  Resolved gender     : {data.gender}")
    print(f"  Original size       : {data.original_size}")
    print(f"  Face detected       : {data.primary_face is not None}")
    print(f"  Face embedding      : {data.has_face_embedding}")
    print(f"  BiSeNet crop box    : {data.face_crop_box}")
    print(f"  Face box in sketch  : {data.face_box_in_sketch}")
    print(f"  BiSeNet mask        : {'available' if data.face_mask_crop is not None else 'not available (rembg fallback will be used)'}")

    # Save preprocessing intermediates
    preprocess_dir = OUTPUT_DIR / f"{person_id}_fru_preprocess"
    preprocess_dir.mkdir(exist_ok=True)

    if data.face_img is not None:
        data.face_img.save(preprocess_dir / "face_cropped.jpg", quality=95)
    if data.face_edges is not None:
        data.face_edges.save(preprocess_dir / "face_edges.png")
    if data.face_mask_crop is not None:
        import numpy as np
        from PIL import Image as _Image
        _Image.fromarray(data.face_mask_crop).save(preprocess_dir / "face_mask.png")

    print(f"\n  Preprocessing intermediates saved to: {preprocess_dir}/")

    if data.face_img is None:
        print("\n  [ERROR] No face crop produced — check that a face is visible in the image.")
        print("          Also verify BiSeNet model is at models/bisenet/bisenet_face_parsing.pth")
        sys.exit(1)

    # ── Generate ───────────────────────────────────────────────────────────
    print("\n[STEP 2/2] Generating face sketch...")
    result = generate_fru_sketch_in_memory(data)

    sketch_path = OUTPUT_DIR / f"{person_id}_fru_final_sketch.jpg"
    result.final_sketch.save(sketch_path, quality=95)
    print(f"\n  Final sketch saved : {sketch_path}")

    if result.scene_images:
        for i, scene_img in enumerate(result.scene_images):
            scene_path = OUTPUT_DIR / f"{person_id}_fru_scene_{i:03d}.jpg"
            scene_img.save(scene_path, quality=95)
        print(f"  Scenes saved       : {len(result.scene_images)} image(s)")
    else:
        print("  No scenes (check inputs/scenes/face_bg/ and crops-face-meta.json)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Local test runner for SAU (full-body) and FRU (face) sketch pipelines."
    )
    parser.add_argument("--image",   required=True,  help="Path to input image")
    parser.add_argument("--id",      required=True,  help="Person ID string")
    parser.add_argument(
        "--gender", default="",
        choices=["male", "female", ""],
        help="Gender override (leave blank to use auto-detection)",
    )
    parser.add_argument(
        "--place", default="SAU",
        choices=["SAU", "FRU"],
        help="Pipeline to run: SAU = full-body sketch, FRU = face sketch (default: SAU)",
    )
    args = parser.parse_args()

    try:
        if args.place == "FRU":
            run_fru(args.image, args.id, args.gender)
        else:
            run_sau(args.image, args.id, args.gender)

        print(f"\n{'='*60}")
        print(f"Done. All outputs in: {OUTPUT_DIR}/")
        print(f"{'='*60}\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)