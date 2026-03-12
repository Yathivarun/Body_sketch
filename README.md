# Portrait Sketch Pipeline — Triton Inference Server

Generates pencil-style sketches from person images and composes them into scene backgrounds. Served via NVIDIA Triton Inference Server with a Python backend. Supports two pipelines: SAU (full-body sketch) and FRU (face-only sketch).

---

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Model weights required](#model-weights-required)
- [Scene backgrounds and crop configuration](#scene-backgrounds-and-crop-configuration)
- [Server setup](#server-setup)
- [Running the server](#running-the-server)
- [Testing](#testing)
- [Debugging](#debugging)
- [Remote client access](#remote-client-access)
- [Local testing without Triton](#local-testing-without-triton)
- [Adding FRU support notes](#adding-fru-support-notes)
- [MinIO integration](#minio-integration)

---

## How it works

```
Client
  |
  v
sketch_pipeline  (Triton ensemble)
  |
  |-- sketch_preprocess  (CPU backend)
  |     reads: image_bytes, person_id, gender, place
  |     routes to SAU or FRU preprocessor based on place value
  |     outputs: preprocessed_data (pickle), person_id, status
  |
  |-- sketch_generate  (GPU backend)
        reads: preprocessed_data, person_id, status
        routes to SAU or FRU generator based on data type
        outputs: scene_images, scene_names, person_id, status
```

Input fields: person image bytes, person ID string, gender string, place string.
Output: N composed scene images as JPEG bytes.

SAU (place=SAU): full-body sketch using SD1.5 + ControlNet lineart + IP-Adapter FaceID + LoRA, composed into scene backgrounds via bbox crop placement.

FRU (place=FRU): face-only sketch using the same SD stack, face extracted via BiSeNet face parsing, composed into scene backgrounds via face anchor point alignment.

---

## Project structure

```
project/
|-- pipeline/
|   |-- preprocessor.py          SAU preprocessing (rembg, face detection, edge maps)
|   |-- generator.py             SAU generation + scene composition
|   |-- fru_preprocessor.py      FRU preprocessing (BiSeNet face crop, edge maps)
|   |-- fru_generator.py         FRU generation + face anchor scene composition
|-- model_repository/
|   |-- sketch_pipeline/
|   |   |-- config.pbtxt         Triton ensemble config
|   |-- sketch_preprocess/
|   |   |-- config.pbtxt         Triton CPU backend config
|   |   |-- 1/
|   |       |-- model.py         Triton preprocess backend
|   |-- sketch_generate/
|       |-- config.pbtxt         Triton GPU backend config
|       |-- 1/
|           |-- model.py         Triton generate backend
|-- ip_adapter/                  IP-Adapter source (not pip-installed)
|-- config.py                    All paths and tunable defaults
|-- crops.json                   SAU scene crop bounding boxes
|-- crops-face-meta.json         FRU face anchor points per scene
|-- client_test.py               Test client (SAU + FRU)
|-- local_test.py                Local test runner without Triton
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
```

---

## Prerequisites

- Docker with NVIDIA Container Toolkit installed
- nvidia-smi accessible on the host
- GPU with at least 8GB VRAM (16GB recommended)
- At least 4GB of available shared memory

Verify GPU access:
```
nvidia-smi
```

Verify Docker GPU access:
```
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Model weights required

All model weights are volume-mounted at runtime. Nothing is baked into the Docker image. Place all weights under a local `./models/` directory before starting the server.

Required directory structure:

```
models/
|-- stable-diffusion-v1-5/           SD1.5 base model (HuggingFace format)
|-- controlnet/
|   |-- control_v11p_sd15_lineart/   ControlNet lineart model
|-- annotators/
|   |-- lineart/                     Lineart annotator weights
|   |-- hed/                         HED annotator weights (fallback)
|-- insightface/
|   |-- models/
|       |-- antelopev2/              InsightFace antelopev2 model pack
|-- ip_adapter/
|   |-- ip-adapter-faceid_sd15.bin   IP-Adapter FaceID weights
|-- loras/
|   |-- Pencil_Sketch_by_vizsumit.safetensors   Style LoRA
|   |-- lcm-lora-sdv1-5/             LCM-LoRA weights (HuggingFace format)
|-- taesd/                           Tiny AutoEncoder for fast LCM decoding
|-- bisenet/
    |-- bisenet_face_parsing.pth     BiSeNet face parsing weights (FRU only)
```

Notes:
- `stable-diffusion-v1-5` must include fp16 variant files (the Dockerfile loads `variant="fp16"` on CUDA)
- `taesd` is optional but recommended for LCM speed. If missing the pipeline falls back to the standard VAE with a warning.
- `bisenet_face_parsing.pth` is required for FRU. Without it FRU falls back to a basic oval crop which produces lower quality results.
- `lcm-lora-sdv1-5` must contain `pytorch_lora_weights.safetensors`.

---

## Scene backgrounds and crop configuration

### SAU scenes

Place scene background images (PNG or JPG) in:
```
inputs/scenes/
```

Edit `crops.json` to define the placement bounding box for each scene. The key is the scene filename stem (without extension). The value is a list of bounding boxes — the pipeline uses index 0.

```json
{
  "park_bench": [[120, 80, 420, 750]],
  "street":     [[200, 50, 500, 800]]
}
```

Bounding box format: `[x1, y1, x2, y2]` in pixels relative to the scene image. The sketch is scaled and placed so it fills this box.

### FRU scenes

Place FRU scene background images in:
```
inputs/scenes/face_bg/
```

Edit `crops-face-meta.json` to define the face anchor point for each scene. The key is the scene filename stem.

```json
{
  "1": {
    "gender": ["male"],
    "face_anchor": {
      "center": [1368, 379],
      "face_w": 80,
      "face_h": 100,
      "bbox": [1328, 329, 1408, 429]
    }
  }
}
```

Fields:
- `gender`: list of allowed genders for this scene. Use `["male", "female", "unknown"]` to allow all.
- `face_anchor.center`: pixel coordinate `[cx, cy]` in the scene image where the face center should land.
- `face_anchor.face_h`: height in pixels of the target face placement zone. The sketch is scaled so the detected face height matches this value.
- `face_anchor.bbox`: reference bounding box, not used by pipeline, for documentation only.

---

## Server setup

Step 1 — Clone the repository and enter the project directory:
```
git clone <your-repo>
cd <your-repo>
```

Step 2 — Place model weights under `./models/` as described above.

Step 3 — Place scene images under `./inputs/scenes/` (SAU) and `./inputs/scenes/face_bg/` (FRU).

Step 4 — Edit `crops.json` and `crops-face-meta.json` to match your scene images.

Step 5 — Verify `docker-compose.yml` volume mounts include:
```yaml
volumes:
  - ./models:/app/models:ro
  - ./inputs:/app/inputs:ro
  - ./crops.json:/app/crops.json:ro
  - ./crops-face-meta.json:/app/crops-face-meta.json:ro
```

Step 6 — Build the Docker image:
```
docker compose build
```

This step takes several minutes on first build due to PyTorch and dependency installation. Subsequent builds are faster due to layer caching.

---

## Running the server

Start the server:
```
docker compose up -d
```

Watch the startup logs (model loading takes 30-90 seconds):
```
docker compose logs -f sketch-triton
```

The server is ready when you see a line containing:
```
Started GRPCInferenceService
```

Verify readiness via HTTP health check:
```
curl http://localhost:8000/v2/health/ready
```

Expected response:
```json
{"live":true}
```

Check individual model status:
```
curl http://localhost:8000/v2/models/sketch_pipeline/ready
curl http://localhost:8000/v2/models/sketch_preprocess/ready
curl http://localhost:8000/v2/models/sketch_generate/ready
```

Stop the server:
```
docker compose down
```

Restart after code changes to `pipeline/` or `config.py`:
```
docker compose build && docker compose up -d
```

---

## Testing

Install test dependencies on the host machine (not inside Docker):
```
pip install tritonclient[http] pillow numpy
```

Run a SAU test (full-body sketch):
```
python client_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --gender female \
  --place  SAU
```

Run a FRU test (face sketch):
```
python client_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --gender female \
  --place  FRU
```

Run with gender auto-detection (leave --gender blank):
```
python client_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --place  SAU
```

Output images are saved to `./test_output/` with filenames in the format:
```
{id}_{place}_{scene_name}.jpg
```

Client arguments:

| Argument   | Required | Default          | Description                                      |
|------------|----------|------------------|--------------------------------------------------|
| --image    | yes      |                  | Path to input image (JPG or PNG)                 |
| --id       | yes      |                  | Person ID string used in output filenames        |
| --gender   | no       | (auto-detect)    | male, female, or leave blank for auto-detection  |
| --place    | no       | SAU              | SAU for full-body, FRU for face sketch           |
| --url      | no       | localhost:8000   | Triton HTTP endpoint                             |

---

## Debugging

### Server will not start

Check Docker logs:
```
docker compose logs sketch-triton
```

Check GPU is visible to Docker:
```
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Check shared memory:
```
df -h /dev/shm
```
If available space is less than 4GB, reduce `shm_size` in `docker-compose.yml`.

### Models fail to load at startup

The server runs with `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` which means all models must be present on disk. Any missing model will cause a hard failure at startup, not a silent fallback.

Verify model paths inside the running container:
```
docker exec -it sketch-triton ls /app/models/stable-diffusion-v1-5
docker exec -it sketch-triton ls /app/models/controlnet/control_v11p_sd15_lineart
docker exec -it sketch-triton ls /app/models/bisenet
docker exec -it sketch-triton ls /app/models/insightface/models/antelopev2
```

### Pipeline returns error status

Check the detailed server logs immediately after sending a request:
```
docker compose logs -f sketch-triton
```

The Python backends print detailed tracebacks to stdout which appear in the Docker logs.

### Scene images are empty or missing

SAU empty scenes:
- Verify `./inputs/scenes/` contains at least one PNG or JPG file
- Verify `crops.json` contains a key matching the scene filename stem
- Check that rembg background removal is working — if the sketch background is not removed the person will not be cleanly composited

FRU empty scenes:
- Verify `./inputs/scenes/face_bg/` contains scene images
- Verify `crops-face-meta.json` has entries for each scene file
- Verify the `gender` field in `crops-face-meta.json` matches the gender being requested
- If BiSeNet is unavailable the face crop falls back to a basic oval crop — this is logged as a warning

### GPU out of memory

If generation fails with CUDA out of memory errors:
- Reduce concurrent requests (max_batch_size is 0 which means one at a time, this should not be an issue)
- Verify no other processes are using the GPU: `nvidia-smi`
- Restart the container to clear any leaked GPU memory: `docker compose restart sketch-triton`

### Request timeout on slow hardware

The client uses a 600 second timeout. First request after server start is slowest due to model warm-up. Subsequent requests are significantly faster due to model caching.

If timeouts occur on a remote client with high network latency, increase the timeout values in `client_test.py`:
```python
connection_timeout=600,
network_timeout=600,
```

### Checking which pipeline ran

The server logs print the pipeline path for every request:
```
[sketch_preprocess] id=A1234 place=SAU gender_override=female
[sketch_generate] SAU id=A1234 gender=female
```
or for FRU:
```
[sketch_preprocess] id=A1234 place=FRU gender_override=female
[sketch_generate] FRU id=A1234 gender=female
```

---

## Remote client access

The server binds to `0.0.0.0` on ports 8000 (HTTP), 8001 (gRPC), and 8002 (Prometheus metrics).

For remote clients, pass the server IP and port via the `--url` argument:
```
python client_test.py \
  --image  person.jpg \
  --id     A1234 \
  --gender female \
  --place  SAU \
  --url    192.168.1.100:8000
```

Firewall requirements — open inbound TCP on the server:
- Port 8000 for HTTP inference (used by client_test.py)
- Port 8001 for gRPC inference (alternative transport)
- Port 8002 for Prometheus metrics (optional, monitoring only)

For cloud VMs (AWS, GCP, Azure) add an inbound security group or firewall rule allowing TCP on port 8000 from your client IP range.

Verify remote connectivity from the client machine:
```
curl http://<server-ip>:8000/v2/health/ready
```

---

## Local testing without Triton

For development and debugging of the pipeline code without running the full Triton stack, use `local_test.py` which calls the preprocessor and generator directly:

Install dependencies locally (requires GPU with CUDA or Apple Silicon with MPS):
```
pip install -r requirements.txt
pip install --no-deps rembg==2.0.69
```

Run SAU locally:
```
python local_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --gender female \
  --place  SAU
```

Run FRU locally:
```
python local_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --gender female \
  --place  FRU
```

Outputs are saved to `./local_test_output/` including intermediate preprocessing results (edge maps, face crops, masks) which are useful for diagnosing quality issues.

---

## Adding FRU support notes

FRU preprocessing uses BiSeNet face parsing to produce a precise face and hair crop with a per-pixel mask. This mask is then applied to the generated sketch so only the face region is used in scene composition.

If BiSeNet is unavailable (model file missing or load error), the pipeline falls back to a simple oval crop around the MTCNN-detected face bounding box. This fallback works but produces lower quality edge alignment in the final composited scenes.

The face anchor system in `crops-face-meta.json` aligns the sketch to the scene by scaling the generated face to match the target `face_h` value and placing it so the face center lands on the `center` coordinate. Hair extends naturally beyond the anchor oval — this is expected and correct.

---

## MinIO integration

The pipeline already accepts `person_id` and `place` as first-class inputs matching the `ID_place` MinIO key structure. To integrate:

1. Write a watcher service that subscribes to MinIO bucket events
2. For each new uploaded image, extract `person_id` and `place` from the object key, read `{id}.json` for gender
3. Call the Triton HTTP endpoint at `http://<server>:8000/v2/models/sketch_pipeline/infer` with those fields
4. Upload the returned scene images back to MinIO as `{id}_{place}_scene_{n}.jpg`

No changes to the Triton backends or pipeline code are required for this integration.

---

## Useful commands reference

```
# Build image
docker compose build

# Start server (detached)
docker compose up -d

# Watch live logs
docker compose logs -f sketch-triton

# Check server health
curl http://localhost:8000/v2/health/ready

# Check specific model readiness
curl http://localhost:8000/v2/models/sketch_pipeline/ready

# List loaded models
curl http://localhost:8000/v2/models

# GPU usage
nvidia-smi

# Shared memory
df -h /dev/shm

# Shell into running container
docker exec -it sketch-triton bash

# Stop server
docker compose down

# Stop and remove volumes
docker compose down -v

# Rebuild and restart after code changes
docker compose build && docker compose up -d
```