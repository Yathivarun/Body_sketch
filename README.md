# Portrait Sketch Pipeline — Triton Server

Generates pencil-style full-body sketches from person images and composes them into scene backgrounds. Served via NVIDIA Triton Inference Server with a Python backend.

---

## How it works

```
Client → sketch_pipeline (ensemble)
              ├── sketch_preprocess  (CPU)  — background removal, edge maps, gender
              └── sketch_generate    (GPU)  — SD1.5 + ControlNet + IP-Adapter + scenes
```

Input: person image + gender + ID. Output: N composed scene images (JPEG bytes).  
FRU (face images) returns a stub — not yet implemented.

---

## Project structure

```
project/
├── pipeline/
│   ├── preprocessor.py       # preprocessing logic (memory-only)
│   └── generator.py          # generation + scene composition logic (memory-only)
├── model_repository/
│   ├── sketch_preprocess/    # Triton CPU backend
│   ├── sketch_generate/      # Triton GPU backend
│   └── sketch_pipeline/      # Triton ensemble (chains the two above)
├── ip_adapter/               # tencent-ailab IP-Adapter (copied, not pip-installed)
├── config.py                 # all paths and tunable defaults
├── client_test.py            # test client
├── crops.json                # scene crop bounding boxes
├── Dockerfile
└── docker-compose.yml
```

---

## What you must provide (volume-mounted, not in image)

```
models/
├── stable-diffusion-v1-5/
├── controlnet/control_v11p_sd15_lineart/
├── annotators/lineart/
├── annotators/hed/
├── insightface/models/antelopev2/
├── ip_adapter/ip-adapter-faceid_sd15.bin
├── loras/Pencil_Sketch_by_vizsumit.safetensors
├── loras/lcm-lora-sdv1-5/          # optional, falls back to HF hub
└── taesd/                          # optional, falls back to standard VAE

inputs/
└── scenes/                         # scene background images (PNG/JPG)

crops.json                          # bounding boxes per scene
```

---

## What a `crops.json` looks like

Each key is the scene image filename (without extension). Value is a list of bounding boxes (can be more than one slot per scene — pipeline uses slot `[0]`).

```json
{
  "park_bench": [[120, 80, 420, 750]],
  "street":     [[200, 50, 500, 800]],
  "library":    [[90, 100, 380, 720]]
}
```

Bounding box format: `[x1, y1, x2, y2]` in pixels relative to the scene image.

---

## Setup

**Prerequisites:** Docker, NVIDIA Container Toolkit, `nvidia-smi` accessible.

```bash
# 1. Clone / copy project
git clone <your-repo> && cd <your-repo>

# 2. Place model weights at ./models/ (see structure above)
# 3. Place scene images at ./inputs/scenes/
# 4. Edit crops.json to match your scene images

# 5. Build image
docker compose build

# 6. Start server
docker compose up -d

# 7. Check server is ready (wait ~60s for model load)
curl http://localhost:8000/v2/health/ready
```

---

## Running a test

Install test dependencies on the host (not inside Docker):

```bash
pip install tritonclient[http] pillow
```

Send a request:

```bash
python client_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --gender female \
  --place  SAU
```

Outputs saved to `./test_output/A1234_scene_*.jpg`.

---

## Useful commands

```bash
# View server logs live
docker compose logs -f sketch-triton

# Check model status
curl http://localhost:8000/v2/models/sketch_pipeline/ready

# Stop server
docker compose down

# Rebuild after code changes (pipeline/ or config.py)
docker compose build && docker compose up -d
```

---

## Adding FRU (face sketch) support later

In `model_repository/sketch_preprocess/1/model.py`, find:

```python
if place == "FRU":
    responses.append(self._stub_response(person_id, "FRU_NOT_IMPLEMENTED"))
    continue
```

Replace with a call to the FRU preprocessing function. No other file needs changing.

---

## MinIO integration (when ready)

The pipeline already uses `person_id` and `place` as first-class inputs, matching the `ID_place` MinIO key structure. To integrate:

1. Write a watcher service that listens to MinIO bucket events
2. For each new object, extract `person_id`, `place`, read `{id}.json` for gender
3. Call `sketch_pipeline` via HTTP or gRPC with those fields
4. Upload the returned scene images back to MinIO as `ID_place_sketch_N.jpg`

No changes to Triton backends required.
