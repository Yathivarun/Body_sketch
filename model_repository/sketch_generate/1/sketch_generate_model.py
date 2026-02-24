"""
Triton Python Backend — sketch_generate
Receives serialised PreprocessedData from sketch_preprocess,
runs the generation + scene composition pipeline, and returns
the composed scene images as JPEG bytes.

Inputs:
    preprocessed_data — pickle-serialised PreprocessedData (bytes)
    person_id         — string identifier (passed through)
    status            — upstream status; non-'ok' values are short-circuited

Outputs:
    scene_images — array of JPEG image bytes, one per scene
    scene_names  — array of scene name strings matching scene_images order
    person_id    — passed through unchanged
    status       — 'ok' | propagated upstream status | 'error: <msg>'
"""

import io
import pickle
import sys
import traceback

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image


class TritonPythonModel:

    def initialize(self, args):
        """
        Called once when Triton loads the model.
        Heavy SD pipeline imports happen here so the GPU is ready
        before any request arrives (warm start).
        """
        sys.path.insert(0, "/app")

        from pipeline.generator import generate_sketch_in_memory
        self._generate = generate_sketch_in_memory

        # Warm up: load pipeline into GPU memory on startup
        # so the first real request is not penalised by model loading time.
        # This is intentionally a no-op call — no input needed.
        print("[sketch_generate] Model initialised — pipeline will load on first request")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                # ── Decode inputs ─────────────────────────────────────────────
                upstream_status = pb_utils.get_input_tensor_by_name(
                    request, "status"
                ).as_numpy()[0][0].decode("utf-8").strip()

                person_id = pb_utils.get_input_tensor_by_name(
                    request, "person_id"
                ).as_numpy()[0][0].decode("utf-8").strip()

                # ── Short-circuit non-ok upstream status ──────────────────────
                # e.g. FRU_NOT_IMPLEMENTED or preprocessing error
                if upstream_status != "ok":
                    print(f"[sketch_generate] Skipping id={person_id}: {upstream_status}")
                    responses.append(self._empty_response(person_id, upstream_status))
                    continue

                # ── Deserialise PreprocessedData ──────────────────────────────
                raw = pb_utils.get_input_tensor_by_name(
                    request, "preprocessed_data"
                ).as_numpy()[0][0]

                preprocessed = pickle.loads(bytes(raw))

                if preprocessed is None:
                    responses.append(self._empty_response(
                        person_id, "error: empty preprocessed_data"
                    ))
                    continue

                # ── Run generation ────────────────────────────────────────────
                print(f"[sketch_generate] Generating for id={person_id} "
                      f"gender={preprocessed.gender}")

                result = self._generate(preprocessed)

                # ── Encode scene images to JPEG bytes ─────────────────────────
                scene_image_bytes = []
                scene_names       = []

                for i, scene_img in enumerate(result.scene_images):
                    buf = io.BytesIO()
                    scene_img.save(buf, format="JPEG", quality=95)
                    scene_image_bytes.append(buf.getvalue())
                    scene_names.append(f"scene_{i:03d}")

                if not scene_image_bytes:
                    # No scenes — still a valid result (no scenes configured)
                    print(f"[sketch_generate] No scenes produced for id={person_id}")

                responses.append(self._make_response(
                    scene_image_bytes=scene_image_bytes,
                    scene_names=scene_names,
                    person_id=person_id,
                    status="ok",
                ))

            except Exception as e:
                traceback.print_exc()
                responses.append(self._empty_response(
                    person_id="",
                    status=f"error: {e}",
                ))

        return responses

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_response(self, scene_image_bytes, scene_names,
                       person_id, status) -> pb_utils.InferenceResponse:
        # scene_images: 1-D array of bytes objects, one per scene
        t_scenes = pb_utils.Tensor(
            "scene_images",
            np.array(scene_image_bytes, dtype=object),
        )
        t_names = pb_utils.Tensor(
            "scene_names",
            np.array([n.encode("utf-8") for n in scene_names], dtype=object),
        )
        t_id = pb_utils.Tensor(
            "person_id",
            np.array([[person_id.encode("utf-8")]], dtype=object),
        )
        t_status = pb_utils.Tensor(
            "status",
            np.array([[status.encode("utf-8")]], dtype=object),
        )
        return pb_utils.InferenceResponse(
            output_tensors=[t_scenes, t_names, t_id, t_status]
        )

    def _empty_response(self, person_id: str,
                        status: str) -> pb_utils.InferenceResponse:
        return self._make_response(
            scene_image_bytes=[],
            scene_names=[],
            person_id=person_id,
            status=status,
        )

    def finalize(self):
        print("[sketch_generate] Model finalised")
