"""
Triton Python Backend — sketch_generate (Triton 24.01 / API 1.17)
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
        from pipeline.generator import generate_sketch_in_memory
        self._generate = generate_sketch_in_memory
        print("[sketch_generate] Initialized — pipeline will load on first request")

    def execute(self, requests):
        pb_utils = self.pb_utils
        responses = []

        for request in requests:
            try:
                # ── Decode inputs ─────────────────────────────────────────────
                upstream_status = pb_utils.get_input_tensor_by_name(
                    request, "status"
                ).as_numpy()[0].decode("utf-8").strip()

                person_id = pb_utils.get_input_tensor_by_name(
                    request, "person_id"
                ).as_numpy()[0].decode("utf-8").strip()

                # ── Short-circuit non-ok upstream status ──────────────────────
                if upstream_status != "ok":
                    print(f"[sketch_generate] Skipping id={person_id}: {upstream_status}")
                    responses.append(self._empty_response(pb_utils, person_id, upstream_status))
                    continue

                # ── Deserialise PreprocessedData ──────────────────────────────
                raw = pb_utils.get_input_tensor_by_name(
                    request, "preprocessed_data"
                ).as_numpy()[0]

                preprocessed = pickle.loads(bytes(raw))

                if preprocessed is None:
                    responses.append(self._empty_response(
                        pb_utils, person_id, "error: empty preprocessed_data"
                    ))
                    continue

                print(f"[sketch_generate] Generating id={person_id} gender={preprocessed.gender}")

                result = self._generate(preprocessed)

                # ── Encode scene images to JPEG bytes ─────────────────────────
                scene_image_bytes = []
                scene_names = []
                for i, scene_img in enumerate(result.scene_images):
                    buf = io.BytesIO()
                    scene_img.save(buf, format="JPEG", quality=95)
                    scene_image_bytes.append(buf.getvalue())
                    scene_names.append(f"scene_{i:03d}".encode("utf-8"))

                if not scene_image_bytes:
                    print(f"[sketch_generate] No scenes produced for id={person_id}")

                responses.append(self._make_response(
                    pb_utils,
                    scene_image_bytes=scene_image_bytes,
                    scene_names=scene_names,
                    person_id=person_id,
                    status="ok",
                ))

            except Exception as e:
                traceback.print_exc()
                responses.append(self._empty_response(pb_utils, "", f"error: {e}"))

        return responses

    def _make_response(self, pb_utils, scene_image_bytes, scene_names, person_id, status):
        # Use empty arrays when no scenes — dtype=object handles variable-length bytes
        scenes_arr = np.array(scene_image_bytes if scene_image_bytes else [b""], dtype=object)
        names_arr  = np.array(scene_names if scene_names else [b""],  dtype=object)

        t_scenes = pb_utils.Tensor("scene_images", scenes_arr)
        t_names  = pb_utils.Tensor("scene_names",  names_arr)
        t_id     = pb_utils.Tensor("person_id",
                       np.array([person_id.encode("utf-8")], dtype=object))
        t_status = pb_utils.Tensor("status",
                       np.array([status.encode("utf-8")], dtype=object))

        return pb_utils.InferenceResponse(output_tensors=[t_scenes, t_names, t_id, t_status])

    def _empty_response(self, pb_utils, person_id, status):
        return self._make_response(
            pb_utils,
            scene_image_bytes=[],
            scene_names=[],
            person_id=person_id,
            status=status,
        )

    def finalize(self):
        print("[sketch_generate] Finalized")