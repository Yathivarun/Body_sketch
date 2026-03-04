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


def run(image_path, person_id, gender, place, url):
    OUTPUT_DIR.mkdir(exist_ok=True)
    client = httpclient.InferenceServerClient(url=url)

    if not client.is_server_live():
        print(f"ERROR: Server not live at {url}")
        sys.exit(1)
    if not client.is_model_ready("sketch_pipeline"):
        print("ERROR: sketch_pipeline not ready")
        sys.exit(1)

    print(f"Server ready. Sending: id={person_id} gender={gender} place={place}")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    def make_input(name, value: bytes):
        # max_batch_size=0 means NO batch dim — shape is exactly [1]
        inp = httpclient.InferInput(name, [1], "BYTES")
        inp.set_data_from_numpy(np.array([value], dtype=object))
        return inp

    inputs = [
        make_input("image_bytes", image_bytes),
        make_input("person_id",   person_id.encode()),
        make_input("gender",      gender.encode()),
        make_input("place",       place.encode()),
    ]

    outputs = [
        httpclient.InferRequestedOutput("scene_images"),
        httpclient.InferRequestedOutput("scene_names"),
        httpclient.InferRequestedOutput("person_id"),
        httpclient.InferRequestedOutput("status"),
    ]

    response = client.infer(
        model_name="sketch_pipeline",
        inputs=inputs,
        outputs=outputs,
    )

    status   = response.as_numpy("status").flat[0]
    ret_id   = response.as_numpy("person_id").flat[0]
    status   = status.decode() if isinstance(status, bytes) else str(status)
    ret_id   = ret_id.decode() if isinstance(ret_id, bytes) else str(ret_id)

    print(f"Response: id={ret_id}  status={status}")

    if status != "ok":
        print(f"Pipeline returned: {status}")
        return

    scenes = response.as_numpy("scene_images")
    names  = response.as_numpy("scene_names")

    if scenes is None or len(scenes) == 0 or scenes.flat[0] == b"":
        print("No scene images returned — check inputs/scenes/ and crops.json")
        return

    for scene_bytes, name_raw in zip(scenes.flat, names.flat):
        name = name_raw.decode() if isinstance(name_raw, bytes) else str(name_raw)
        img  = Image.open(io.BytesIO(bytes(scene_bytes)))
        out  = OUTPUT_DIR / f"{ret_id}_{name}.jpg"
        img.save(out, quality=95)
        print(f"  Saved: {out}")

    print(f"\nDone — {len(list(scenes.flat))} scene(s) in {OUTPUT_DIR}/")


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