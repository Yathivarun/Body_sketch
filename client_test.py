"""
Test client for the sketch_pipeline Triton ensemble.
Sends one image + metadata and saves the returned scene images to ./test_output/.

Usage:
    python client_test.py --image path/to/person.jpg --id A1234 --gender female
    python client_test.py --image path/to/person.jpg --id A1234 --gender male --place SAU
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tritonclient.http as httpclient
except ImportError:
    print("ERROR: Install tritonclient — pip install tritonclient[http]")
    sys.exit(1)


# ── Defaults ──────────────────────────────────────────────────────────────────
TRITON_URL   = os.getenv("TRITON_URL", "localhost:8000")
MODEL_NAME   = "sketch_pipeline"
OUTPUT_DIR   = Path("test_output")


def load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def run(image_path: str, person_id: str, gender: str, place: str):
    OUTPUT_DIR.mkdir(exist_ok=True)

    client = httpclient.InferenceServerClient(url=TRITON_URL)

    # ── Check server health ───────────────────────────────────────────────────
    if not client.is_server_live():
        print(f"ERROR: Triton server not live at {TRITON_URL}")
        sys.exit(1)
    if not client.is_model_ready(MODEL_NAME):
        print(f"ERROR: Model '{MODEL_NAME}' not ready")
        sys.exit(1)

    print(f"Server live. Sending request: id={person_id} gender={gender} place={place}")

    # ── Prepare inputs ────────────────────────────────────────────────────────
    image_bytes = load_image_bytes(image_path)

    def bytes_input(name: str, value: bytes) -> httpclient.InferInput:
        inp = httpclient.InferInput(name, [1, 1], "BYTES")
        inp.set_data_from_numpy(np.array([[value]], dtype=object))
        return inp

    inputs = [
        bytes_input("image_bytes",  image_bytes),
        bytes_input("person_id",    person_id.encode()),
        bytes_input("gender",       gender.encode()),
        bytes_input("place",        place.encode()),
    ]

    outputs = [
        httpclient.InferRequestedOutput("scene_images"),
        httpclient.InferRequestedOutput("scene_names"),
        httpclient.InferRequestedOutput("person_id"),
        httpclient.InferRequestedOutput("status"),
    ]

    # ── Send request ──────────────────────────────────────────────────────────
    response = client.infer(
        model_name=MODEL_NAME,
        inputs=inputs,
        outputs=outputs,
    )

    # ── Parse outputs ─────────────────────────────────────────────────────────
    status    = response.as_numpy("status")[0].decode("utf-8")
    ret_id    = response.as_numpy("person_id")[0].decode("utf-8")
    scenes    = response.as_numpy("scene_images")   # array of bytes
    names     = response.as_numpy("scene_names")    # array of bytes

    print(f"\nResponse: id={ret_id} status={status}")

    if status != "ok":
        print(f"Pipeline returned non-ok status: {status}")
        return

    if scenes is None or len(scenes) == 0:
        print("No scene images returned (check scenes/ directory and crops.json)")
        return

    # ── Save outputs ──────────────────────────────────────────────────────────
    for scene_bytes, name_bytes in zip(scenes, names):
        name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else name_bytes
        img  = Image.open(io.BytesIO(bytes(scene_bytes)))
        out  = OUTPUT_DIR / f"{ret_id}_{name}.jpg"
        img.save(out, quality=95)
        print(f"  Saved: {out}")

    print(f"\nDone — {len(scenes)} scene(s) saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True,  help="Path to input person image")
    parser.add_argument("--id",     required=True,  help="Person ID string")
    parser.add_argument("--gender", default="unknown",
                        choices=["male", "female", "unknown"], help="Gender from JSON")
    parser.add_argument("--place",  default="SAU",
                        choices=["SAU", "FRU"], help="Image source place code")
    parser.add_argument("--url",    default=TRITON_URL, help="Triton server URL")
    args = parser.parse_args()

    TRITON_URL = args.url
    run(args.image, args.id, args.gender, args.place)
