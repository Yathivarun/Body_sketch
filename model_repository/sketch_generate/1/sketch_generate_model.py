"""
Triton Python Backend - sketch_generate (Triton 24.01 compatible)
"""

import io
import pickle
import sys
import traceback

import numpy as np
from PIL import Image


def _decode_string(tensor):
    """Safely extract a string from a Triton input tensor regardless of shape."""
    arr = tensor.as_numpy()
    val = arr.flat[0]
    if isinstance(val, bytes):
        return val.decode("utf-8").strip()
    if isinstance(val, np.ndarray):
        return val.flat[0].decode("utf-8").strip()
    return str(val).strip()

def _decode_bytes(tensor):
    """Safely extract raw bytes from a Triton input tensor."""
    arr = tensor.as_numpy()
    val = arr.flat[0]
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, np.ndarray):
        return bytes(val.flat[0])
    return bytes(val)


class TritonPythonModel:

    def initialize(self, args):
        import triton_python_backend_utils as pb_utils
        self.pb_utils = pb_utils
        sys.path.insert(0, "/app")
        from pipeline.generator import generate_sketch_in_memory
        self._generate = generate_sketch_in_memory
        print("[sketch_generate] Initialised")

    def execute(self, requests):
        pb_utils = self.pb_utils
        responses = []

        for request in requests:
            try:
                upstream_status = _decode_string(
                    pb_utils.get_input_tensor_by_name(request, "status")
                )
                person_id = _decode_string(
                    pb_utils.get_input_tensor_by_name(request, "person_id")
                )

                if upstream_status != "ok":
                    print(f"[sketch_generate] Skipping id={person_id}: {upstream_status}")
                    responses.append(self._empty_response(pb_utils, person_id, upstream_status))
                    continue

                raw = _decode_bytes(
                    pb_utils.get_input_tensor_by_name(request, "preprocessed_data")
                )
                preprocessed = pickle.loads(raw)

                if preprocessed is None:
                    responses.append(self._empty_response(
                        pb_utils, person_id, "error: empty preprocessed_data"
                    ))
                    continue

                print(f"[sketch_generate] id={person_id} gender={preprocessed.gender}")
                result = self._generate(preprocessed)

                scene_image_bytes = []
                scene_names = []
                for i, scene_img in enumerate(result.scene_images):
                    buf = io.BytesIO()
                    scene_img.save(buf, format="JPEG", quality=95)
                    scene_image_bytes.append(buf.getvalue())
                    scene_names.append(f"scene_{i:03d}".encode("utf-8"))

                if not scene_image_bytes:
                    scene_image_bytes = [b""]
                    scene_names = [b""]

                responses.append(self._make_response(
                    pb_utils, scene_image_bytes, scene_names, person_id, "ok"
                ))

            except Exception as e:
                traceback.print_exc()
                responses.append(self._empty_response(pb_utils, "", f"error: {e}"))

        return responses

    def _make_response(self, pb_utils, scene_image_bytes, scene_names, person_id, status):
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("scene_images",
                            np.array(scene_image_bytes, dtype=object)),
            pb_utils.Tensor("scene_names",
                            np.array(scene_names, dtype=object)),
            pb_utils.Tensor("person_id",
                            np.array([person_id.encode("utf-8")], dtype=object)),
            pb_utils.Tensor("status",
                            np.array([status.encode("utf-8")], dtype=object)),
        ])

    def _empty_response(self, pb_utils, person_id, status):
        return self._make_response(pb_utils, [b""], [b""], person_id, status)

    def finalize(self):
        print("[sketch_generate] Finalised")
