"""
Triton Python Backend — sketch_preprocess (Triton 24.01 / API 1.17)
"""

import io
import pickle
import sys
import traceback
import numpy as np


class TritonPythonModel:

    def initialize(self, args):
        import triton_python_backend_utils as pb_utils
        self.pb_utils = pb_utils

        sys.path.insert(0, "/app")
        from pipeline.preprocessor import preprocess_image_in_memory
        self._preprocess = preprocess_image_in_memory
        print("[sketch_preprocess] Initialized")

    def execute(self, requests):
        pb_utils = self.pb_utils
        responses = []

        for request in requests:
            try:
                # ── Decode inputs ─────────────────────────────────────────────
                image_bytes = pb_utils.get_input_tensor_by_name(
                    request, "image_bytes"
                ).as_numpy()[0].tobytes()

                person_id = pb_utils.get_input_tensor_by_name(
                    request, "person_id"
                ).as_numpy()[0].decode("utf-8").strip()

                gender_raw = pb_utils.get_input_tensor_by_name(
                    request, "gender"
                ).as_numpy()[0].decode("utf-8").strip().lower()

                place = pb_utils.get_input_tensor_by_name(
                    request, "place"
                ).as_numpy()[0].decode("utf-8").strip().upper()

                # ── FRU stub ──────────────────────────────────────────────────
                if place == "FRU":
                    responses.append(self._make_response(
                        pb_utils,
                        serialised_data=pickle.dumps(None),
                        person_id=person_id,
                        status="FRU_NOT_IMPLEMENTED",
                    ))
                    continue

                # ── Decode image ──────────────────────────────────────────────
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))

                gender_override = gender_raw if gender_raw in ("male", "female") else None

                print(f"[sketch_preprocess] id={person_id} place={place} gender_override={gender_override}")

                preprocessed = self._preprocess(
                    img=img,
                    gender_override=gender_override,
                )

                serialised = pickle.dumps(preprocessed)

                responses.append(self._make_response(
                    pb_utils,
                    serialised_data=serialised,
                    person_id=person_id,
                    status="ok",
                ))

            except Exception as e:
                traceback.print_exc()
                responses.append(self._make_response(
                    pb_utils,
                    serialised_data=pickle.dumps(None),
                    person_id="",
                    status=f"error: {e}",
                ))

        return responses

    def _make_response(self, pb_utils, serialised_data, person_id, status):
        t_data = pb_utils.Tensor(
            "preprocessed_data",
            np.array([serialised_data], dtype=object),
        )
        t_id = pb_utils.Tensor(
            "person_id",
            np.array([person_id.encode("utf-8")], dtype=object),
        )
        t_status = pb_utils.Tensor(
            "status",
            np.array([status.encode("utf-8")], dtype=object),
        )
        return pb_utils.InferenceResponse(output_tensors=[t_data, t_id, t_status])

    def finalize(self):
        print("[sketch_preprocess] Finalized")