"""
Test client for the sketch_pipeline Triton ensemble.
Supports both SAU (full-body sketch) and FRU (face sketch) pipelines.

Usage:
    # SAU — full-body sketch
    python client_test.py --image person.jpg --id A1234 --gender female --place SAU

    # FRU — face sketch
    python client_test.py --image person.jpg --id A1234 --gender female --place FRU

    # Remote server
    python client_test.py --image person.jpg --id A1234 --gender female --place SAU --url 192.168.1.10:8000

    # Unknown gender (auto-detect)
    python client_test.py --image person.jpg --id A1234 --place SAU
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tritonclient.http as httpclient
except ImportError:
    print("ERROR: tritonclient not installed.")
    print("       pip install tritonclient[http]")
    sys.exit(1)

OUTPUT_DIR = Path("test_output")


# ============================================================================
# HELPERS
# ============================================================================

def make_bytes_input(name: str, value: bytes) -> httpclient.InferInput:
    inp = httpclient.InferInput(name, [1], "BYTES")
    inp.set_data_from_numpy(np.array([value], dtype=object))
    return inp


def decode_scalar(arr) -> str:
    val = arr.flat[0]
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val)


def check_server(client, url: str):
    print(f"[INFO] Connecting to Triton at {url} ...")
    try:
        if not client.is_server_live():
            print(f"[ERROR] Server is not live at {url}")
            print("        Check that the container is running: docker compose ps")
            sys.exit(1)
        print("[INFO] Server is live")
    except Exception as e:
        print(f"[ERROR] Cannot reach server at {url}")
        print(f"        {e}")
        print("        Is the container running? Is the port correct?")
        sys.exit(1)

    try:
        if not client.is_server_ready():
            print("[ERROR] Server is not ready — models may still be loading")
            print("        Wait ~60s and retry, or check logs: docker compose logs -f sketch-triton")
            sys.exit(1)
        print("[INFO] Server is ready")
    except Exception as e:
        print(f"[ERROR] Server ready check failed: {e}")
        sys.exit(1)


def check_model(client, model_name: str):
    try:
        if not client.is_model_ready(model_name):
            print(f"[ERROR] Model '{model_name}' is not ready")
            print(f"        Check model status: curl http://<host>:8000/v2/models/{model_name}/ready")
            print(f"        Check logs: docker compose logs -f sketch-triton")
            sys.exit(1)
        print(f"[INFO] Model '{model_name}' is ready")
    except Exception as e:
        print(f"[ERROR] Model ready check failed: {e}")
        sys.exit(1)


# ============================================================================
# MAIN RUN
# ============================================================================

def run(image_path: str, person_id: str, gender: str, place: str, url: str):
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ── Client setup ──────────────────────────────────────────────────────────
    client = httpclient.InferenceServerClient(
        url=url,
        connection_timeout=600,
        network_timeout=600,
    )

    check_server(client, url)
    check_model(client, "sketch_pipeline")

    # ── Print request summary ─────────────────────────────────────────────────
    print("")
    print("=" * 60)
    print(f"  Pipeline : {place}")
    print(f"  Image    : {image_path}")
    print(f"  ID       : {person_id}")
    print(f"  Gender   : {gender if gender else 'not provided (auto-detect)'}")
    print(f"  Server   : {url}")
    print("=" * 60)
    print("")

    # ── Read image ────────────────────────────────────────────────────────────
    if not Path(image_path).exists():
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    print(f"[INFO] Image loaded: {len(image_bytes) / 1024:.1f} KB")

    # ── Build inputs ──────────────────────────────────────────────────────────
    inputs = [
        make_bytes_input("image_bytes", image_bytes),
        make_bytes_input("person_id",   person_id.encode("utf-8")),
        make_bytes_input("gender",      (gender or "unknown").encode("utf-8")),
        make_bytes_input("place",       place.encode("utf-8")),
    ]

    outputs = [
        httpclient.InferRequestedOutput("scene_images"),
        httpclient.InferRequestedOutput("scene_names"),
        httpclient.InferRequestedOutput("person_id"),
        httpclient.InferRequestedOutput("status"),
    ]

    # ── Send request ──────────────────────────────────────────────────────────
    print("[INFO] Sending inference request ...")
    start = time.time()

    try:
        response = client.infer(
            model_name="sketch_pipeline",
            inputs=inputs,
            outputs=outputs,
        )
    except Exception as e:
        print(f"[ERROR] Inference request failed: {e}")
        print("")
        print("Debugging steps:")
        print("  1. Check server logs: docker compose logs -f sketch-triton")
        print("  2. Check model status: curl http://<host>:8000/v2/models/sketch_pipeline/ready")
        print("  3. Check GPU memory: nvidia-smi")
        sys.exit(1)

    elapsed = time.time() - start
    print(f"[INFO] Response received in {elapsed:.1f}s")

    # ── Parse response ────────────────────────────────────────────────────────
    status  = decode_scalar(response.as_numpy("status"))
    ret_id  = decode_scalar(response.as_numpy("person_id"))
    scenes  = response.as_numpy("scene_images")
    names   = response.as_numpy("scene_names")

    print(f"[INFO] Response: id={ret_id}  status={status}")

    if status != "ok":
        print(f"[ERROR] Pipeline returned error status: {status}")
        print("")
        print("Debugging steps:")
        print("  1. Check server logs: docker compose logs -f sketch-triton")
        print("  2. Verify input image contains a clearly visible person")
        print("  3. For FRU: verify input image contains a clearly visible face")
        sys.exit(1)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if scenes is None or scenes.flat[0] == b"":
        print("[WARNING] Pipeline returned no scene images")
        print("")
        print("Possible reasons:")
        print("  SAU: check that inputs/scenes/ contains scene images and crops.json is correct")
        print("  FRU: check that inputs/scenes/face_bg/ contains scene images and crops-face-meta.json is correct")
        print("  FRU: check that gender in crops-face-meta.json matches the requested gender")
        return

    saved = 0
    for scene_bytes, name_raw in zip(scenes.flat, names.flat):
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)

        if not scene_bytes or scene_bytes == b"":
            continue

        try:
            img = Image.open(io.BytesIO(bytes(scene_bytes)))
            out_path = OUTPUT_DIR / f"{ret_id}_{place.lower()}_{name}.jpg"
            img.save(out_path, quality=95)
            print(f"[INFO] Saved: {out_path}  ({img.width}x{img.height})")
            saved += 1
        except Exception as e:
            print(f"[WARNING] Could not save scene '{name}': {e}")

    print("")
    print("=" * 60)
    print(f"  Done. {saved} scene(s) saved to {OUTPUT_DIR}/")
    print("=" * 60)
    print("")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triton test client for SAU (full-body) and FRU (face) sketch pipelines."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to input person image (JPG or PNG)",
    )
    parser.add_argument(
        "--id",
        required=True,
        help="Person ID string (used in output filenames)",
    )
    parser.add_argument(
        "--gender",
        default="",
        choices=["male", "female", ""],
        help="Gender override. Leave blank to use auto-detection from InsightFace.",
    )
    parser.add_argument(
        "--place",
        default="SAU",
        choices=["SAU", "FRU"],
        help="Pipeline to run: SAU = full-body sketch, FRU = face sketch (default: SAU)",
    )
    parser.add_argument(
        "--url",
        default="localhost:8000",
        help="Triton server HTTP endpoint (default: localhost:8000)",
    )
    args = parser.parse_args()

    run(
        image_path=args.image,
        person_id=args.id,
        gender=args.gender,
        place=args.place,
        url=args.url,
    )