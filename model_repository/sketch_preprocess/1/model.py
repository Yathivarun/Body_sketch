"""
Triton Python Backend - sketch_preprocess (Triton 24.01 compatible)
Routes to SAU (full-body) or FRU (face) preprocessing based on the 'place' input.
"""

import io
import pickle
import sys
import traceback

import numpy as np
from PIL import Image


def _decode_string(tensor, name):
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

        from pipeline.preprocessor import preprocess_image_in_memory
        self._preprocess_sau = preprocess_image_in_memory

        from pipeline.fru_preprocessor import preprocess_fru_image_in_memory
        self._preprocess_fru = preprocess_fru_image_in_memory

        print("[sketch_preprocess] Initialised (SAU + FRU)")

    def execute(self, requests):
        pb_utils = self.pb_utils
        responses = []

        for request in requests:
            try:
                image_bytes = _decode_bytes(
                    pb_utils.get_input_tensor_by_name(request, "image_bytes")
                )
                person_id = _decode_string(
                    pb_utils.get_input_tensor_by_name(request, "person_id"), "person_id"
                )
                gender_raw = _decode_string(
                    pb_utils.get_input_tensor_by_name(request, "gender"), "gender"
                ).lower()
                place = _decode_string(
                    pb_utils.get_input_tensor_by_name(request, "place"), "place"
                ).upper()

                img = Image.open(io.BytesIO(image_bytes))
                gender_override = gender_raw if gender_raw in ("male", "female") else None

                print(
                    f"[sketch_preprocess] id={person_id} place={place} "
                    f"gender_override={gender_override}"
                )

                if place == "FRU":
                    preprocessed = self._preprocess_fru(
                        img=img,
                        gender_override=gender_override,
                    )
                else:
                    # SAU (default) — also handles any unknown place value
                    preprocessed = self._preprocess_sau(
                        img=img,
                        gender_override=gender_override,
                    )

                responses.append(self._make_response(
                    pb_utils, pickle.dumps(preprocessed), person_id, "ok"
                ))

            except Exception as e:
                traceback.print_exc()
                safe_error = ("error: " + str(e)).encode("ascii", "ignore").decode()
                responses.append(self._error_response(pb_utils, "", safe_error))

        return responses

    def _make_response(self, pb_utils, serialised_data, person_id, status):
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("preprocessed_data",
                            np.array([serialised_data], dtype=object)),
            pb_utils.Tensor("person_id",
                            np.array([person_id.encode("utf-8")], dtype=object)),
            pb_utils.Tensor("status",
                            np.array([status.encode("utf-8")], dtype=object)),
        ])

    def _error_response(self, pb_utils, person_id, status):
        return self._make_response(pb_utils, pickle.dumps(None), person_id, status)

    def finalize(self):
        import torch

        # Clear the shared pipeline model cache if it was populated
        try:
            from pipeline import preprocessor as _pre_mod
            if hasattr(_pre_mod, "_MODEL_CACHE"):
                _pre_mod._MODEL_CACHE.clear()
                print("[sketch_preprocess] _MODEL_CACHE cleared (SAU preprocessor)")
        except Exception:
            pass

        try:
            from pipeline import fru_preprocessor as _fru_mod
            if hasattr(_fru_mod, "_MODEL_CACHE"):
                _fru_mod._MODEL_CACHE.clear()
                print("[sketch_preprocess] _MODEL_CACHE cleared (FRU preprocessor)")
        except Exception:
            pass

        # Release any VRAM that PyTorch allocated for preprocessing ops
        torch.cuda.empty_cache()
        print("[sketch_preprocess] Finalised — VRAM released")
