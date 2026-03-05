"""
Test client for the sketch_pipeline Triton ensemble.
Usage:
    python client_test.py --image person.jpg --id A1234 --gender female --place SAU
    python client_test.py --image person.jpg --id A1234 --gender female --place SAU --url 192.168.1.10:8000
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
    print("ERROR: pip install tritonclient[http]")
    sys.exit(1)

OUTPUT_DIR = Path("test_output")


import time

def run(image_path, person_id, gender, place, url):
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n[DEBUG] Creating Triton client...")
    client = httpclient.InferenceServerClient(
        url=url,
        connection_timeout=300,
        network_timeout=300
    )

    print("[DEBUG] Checking if server is live...")
    if not client.is_server_live():
        print(f"[ERROR] Server not live at {url}")
        sys.exit(1)
    print("[DEBUG] Server is live")

    print("[DEBUG] Checking if model is ready...")
    if not client.is_model_ready("sketch_pipeline"):
        print("[ERROR] sketch_pipeline model not ready")
        sys.exit(1)
    print("[DEBUG] Model sketch_pipeline is ready")

    print(f"\n[INFO] Sending request")
    print(f"       id={person_id}")
    print(f"       gender={gender}")
    print(f"       place={place}")
    print(f"       image={image_path}")

    print("[DEBUG] Reading image file...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"[DEBUG] Image size: {len(image_bytes)/1024:.2f} KB")

    def make_input(name, value: bytes):
        print(f"[DEBUG] Preparing input: {name}")
        inp = httpclient.InferInput(name, [1], "BYTES")
        inp.set_data_from_numpy(np.array([value], dtype=object))
        return inp

    print("[DEBUG] Building inputs...")
    inputs = [
        make_input("image_bytes", image_bytes),
        make_input("person_id", person_id.encode()),
        make_input("gender", gender.encode()),
        make_input("place", place.encode()),
    ]

    print("[DEBUG] Building requested outputs...")
    outputs = [
        httpclient.InferRequestedOutput("scene_images"),
        httpclient.InferRequestedOutput("scene_names"),
        httpclient.InferRequestedOutput("person_id"),
        httpclient.InferRequestedOutput("status"),
    ]

    print("\n[DEBUG] Sending inference request to Triton...")
    start_time = time.time()

    try:
        response = client.infer(
            model_name="sketch_pipeline",
            inputs=inputs,
            outputs=outputs,
        )
    except Exception as e:
        print("[ERROR] Exception during infer()")
        print(e)
        sys.exit(1)

    end_time = time.time()
    print(f"[DEBUG] Response received in {end_time - start_time:.2f} sec")

    print("[DEBUG] Parsing response...")

    status = response.as_numpy("status").flat[0]
    ret_id = response.as_numpy("person_id").flat[0]

    status = status.decode() if isinstance(status, bytes) else str(status)
    ret_id = ret_id.decode() if isinstance(ret_id, bytes) else str(ret_id)

    print(f"[INFO] Response: id={ret_id} status={status}")

    if status != "ok":
        print(f"[WARNING] Pipeline returned status: {status}")
        return

    print("[DEBUG] Extracting scene outputs...")

    scenes = response.as_numpy("scene_images")
    names = response.as_numpy("scene_names")

    if scenes is None or len(scenes) == 0 or scenes.flat[0] == b"":
        print("[WARNING] No scene images returned")
        return

    print(f"[DEBUG] Number of scenes returned: {len(list(scenes.flat))}")

    for scene_bytes, name_raw in zip(scenes.flat, names.flat):
        name = name_raw.decode() if isinstance(name_raw, bytes) else str(name_raw)

        print(f"[DEBUG] Processing scene: {name}")

        img = Image.open(io.BytesIO(bytes(scene_bytes)))
        out = OUTPUT_DIR / f"{ret_id}_{name}.jpg"

        img.save(out, quality=95)
        print(f"[INFO] Saved: {out}")

    print(f"\n[INFO] Done — {len(list(scenes.flat))} scenes saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  required=True)
    parser.add_argument("--id",     required=True)
    parser.add_argument("--gender", default="unknown",
                        choices=["male", "female", "unknown"])
    parser.add_argument("--place",  default="SAU",
                        choices=["SAU", "FRU"])
    parser.add_argument("--url",    default="localhost:8000")
    args = parser.parse_args()
    run(args.image, args.id, args.gender, args.place, args.url)