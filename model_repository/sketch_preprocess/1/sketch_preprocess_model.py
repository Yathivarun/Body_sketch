"""
Triton Python Backend — sketch_preprocess
Runs the full preprocessing pipeline and passes a serialised
PreprocessedData object to sketch_generate via the ensemble.

Inputs:
    image_bytes  — raw image file bytes
    person_id    — string identifier (maps to MinIO ID later)
    gender       — 'male' | 'female' | 'unknown'  (from person JSON)
    place        — 'SAU' | 'FRU'  (routing flag)

Outputs:
    preprocessed_data — pickle-serialised PreprocessedData object (bytes)
    person_id         — passed through unchanged for generate stage
    status            — 'ok' | 'FRU_NOT_IMPLEMENTED' | 'error: <msg>'
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
        We import the pipeline here (not at module level) so Triton's
        process is fully up before heavy imports run.
        """
        import os
        # /app is on PYTHONPATH via Dockerfile ENV
        sys.path.insert(0, "/app")

        from pipeline.preprocessor import preprocess_image_in_memory
        self._preprocess = preprocess_image_in_memory
        print("[sketch_preprocess] Model initialised")

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                # ── Decode inputs ─────────────────────────────────────────────
                image_bytes = pb_utils.get_input_tensor_by_name(
                    request, "image_bytes"
                ).as_numpy()[0][0]

                person_id = pb_utils.get_input_tensor_by_name(
                    request, "person_id"
                ).as_numpy()[0][0].decode("utf-8").strip()

                gender_raw = pb_utils.get_input_tensor_by_name(
                    request, "gender"
                ).as_numpy()[0][0].decode("utf-8").strip().lower()

                place = pb_utils.get_input_tensor_by_name(
                    request, "place"
                ).as_numpy()[0][0].decode("utf-8").strip().upper()

                # ── FRU stub ──────────────────────────────────────────────────
                # Face-swap sketching not yet implemented.
                # Modular hook: replace this block when FRU pipeline is ready.
                if place == "FRU":
                    responses.append(self._stub_response(
                        person_id, "FRU_NOT_IMPLEMENTED"
                    ))
                    continue

                # ── Decode image ──────────────────────────────────────────────
                img = Image.open(io.BytesIO(bytes(image_bytes)))

                # ── Gender override logic ─────────────────────────────────────
                # Pass valid override; preprocessor falls back to detection
                # if override is missing or not one of male/female.
                gender_override = gender_raw if gender_raw in ("male", "female") else None

                # ── Run preprocessing ─────────────────────────────────────────
                print(f"[sketch_preprocess] Processing id={person_id} "
                      f"place={place} gender_override={gender_override}")

                preprocessed = self._preprocess(
                    img=img,
                    gender_override=gender_override,
                )

                # ── Serialise PreprocessedData for handoff to generate stage ──
                serialised = pickle.dumps(preprocessed)

                # ── Build response ────────────────────────────────────────────
                responses.append(self._make_response(
                    serialised_data=serialised,
                    person_id=person_id,
                    status="ok",
                ))

            except Exception as e:
                traceback.print_exc()
                responses.append(self._stub_response(
                    person_id="",
                    status=f"error: {e}",
                ))

        return responses

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_response(self, serialised_data: bytes, person_id: str,
                       status: str) -> pb_utils.InferenceResponse:
        t_data = pb_utils.Tensor(
            "preprocessed_data",
            np.array([[serialised_data]], dtype=object),
        )
        t_id = pb_utils.Tensor(
            "person_id",
            np.array([[person_id.encode("utf-8")]], dtype=object),
        )
        t_status = pb_utils.Tensor(
            "status",
            np.array([[status.encode("utf-8")]], dtype=object),
        )
        return pb_utils.InferenceResponse(output_tensors=[t_data, t_id, t_status])

    def _stub_response(self, person_id: str,
                       status: str) -> pb_utils.InferenceResponse:
        """Returns empty preprocessed_data with a non-ok status."""
        return self._make_response(
            serialised_data=pickle.dumps(None),
            person_id=person_id,
            status=status,
        )

    def finalize(self):
        print("[sketch_preprocess] Model finalised")
