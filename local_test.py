"""
Local test runner — no Triton needed.
Runs preprocessor -> generator directly and saves all outputs to ./local_test_output/

Usage:
    python run_local_test.py --image person.jpg --id A1234 --gender female
    python run_local_test.py --image person.jpg --id A1234 --gender male
    python run_local_test.py --image person.jpg --id A1234  # no gender, uses detection
"""

import argparse
import sys
import io
from pathlib import Path
from PIL import Image

# ── Make sure project root is on path ─────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.preprocessor import preprocess_image_in_memory
from pipeline.generator import generate_sketch_in_memory

OUTPUT_DIR = Path("local_test_output")


def run(image_path: str, person_id: str, gender: str):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Load image ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Input : {image_path}")
    print(f"ID    : {person_id}")
    print(f"Gender: {gender or 'not provided — will auto-detect'}")
    print(f"{'='*60}\n")

    img = Image.open(image_path)

    # ── Preprocess ─────────────────────────────────────────────────────────────
    print("[STEP 1/2] Preprocessing...")
    gender_override = gender if gender in ("male", "female") else None
    data = preprocess_image_in_memory(img, gender_override=gender_override)

    print(f"\n  Resolved gender : {data.gender}")
    print(f"  Original size   : {data.original_size}")
    print(f"  Face detected   : {data.primary_face is not None}")
    print(f"  Face embedding  : {data.has_face_embedding}")

    # Save intermediate outputs so you can inspect preprocessing quality
    preprocess_dir = OUTPUT_DIR / f"{person_id}_preprocess"
    preprocess_dir.mkdir(exist_ok=True)

    data.enhanced.save(preprocess_dir / "enhanced.jpg", quality=95)
    data.body_edges.save(preprocess_dir / "body_edges.png")
    if data.face_img is not None:
        data.face_img.save(preprocess_dir / "face_cropped.jpg", quality=95)
    if data.face_edges is not None:
        data.face_edges.save(preprocess_dir / "face_edges.png")

    print(f"\n  Preprocessing intermediates saved to: {preprocess_dir}/")

    # ── Generate ───────────────────────────────────────────────────────────────
    print("\n[STEP 2/2] Generating sketches...")
    result = generate_sketch_in_memory(data)

    # Save final sketch
    sketch_path = OUTPUT_DIR / f"{person_id}_final_sketch.jpg"
    result.final_sketch.save(sketch_path, quality=95)
    print(f"\n  Final sketch saved : {sketch_path}")

    # Save face sketch if generated
    if result.face_sketch is not None:
        face_path = OUTPUT_DIR / f"{person_id}_face_sketch.jpg"
        result.face_sketch.save(face_path, quality=95)
        print(f"  Face sketch saved  : {face_path}")

    # Save scene compositions
    if result.scene_images:
        for i, scene_img in enumerate(result.scene_images):
            scene_path = OUTPUT_DIR / f"{person_id}_scene_{i:03d}.jpg"
            scene_img.save(scene_path, quality=95)
        print(f"  Scenes saved       : {len(result.scene_images)} image(s)")
    else:
        print("  No scenes (check inputs/scenes/ and crops.json)")

    print(f"\n{'='*60}")
    print(f"Done. All outputs in: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True,  help="Path to input image")
    parser.add_argument("--id",     required=True,  help="Person ID string")
    parser.add_argument("--gender", default="",
                        choices=["male", "female", ""],
                        help="Gender override (leave blank to use auto-detection)")
    args = parser.parse_args()

    try:
        run(args.image, args.id, args.gender)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)