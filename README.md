# Portrait Sketch Pipeline

Generates pencil-style sketches from person images and composes them into scene backgrounds. Supports two pipelines:
* **SAU (Full-body sketch)**: Uses SD1.5 + ControlNet lineart + IP-Adapter FaceID + Style LoRA.
* **FRU (Face-only sketch)**: Uses the same SD stack but extracts the face via BiSeNet face parsing and aligns it using specific anchor points.

---

## Table of Contents

1. [Quick Start: Local Setup](#1-quick-start-local-setup)
2. [Downloading Required Models](#2-downloading-required-models)
3. [Folder Structure](#3-folder-structure)
4. [Scene & Crop Configuration](#4-scene--crop-configuration)
5. [Running Local Tests](#5-running-local-tests)
6. [Advanced: Triton Server Deployment](#6-advanced-triton-server-deployment)

---

## 1. Quick Start: Local Setup

For development, debugging, or just trying out the pipeline, you can run everything locally without setting up the Triton Docker container.

**Prerequisites:** Python 3.10+ and an NVIDIA GPU (or Apple Silicon with MPS).

**Step 1:** Clone the repository and navigate to the directory.
```bash
git clone https://github.com/Yathivarun/Body_sketch.git
cd Body_sketch-main
```

**Step 2:** Install dependencies. (Note: `rembg` is installed with `--no-deps` to prevent it from downgrading/conflicting with the GPU-enabled onnxruntime).

```bash
pip install -r requirements.txt
pip install --no-deps rembg==2.0.69
```

---

## 2. Downloading Required Models

The pipeline runs completely offline. You must download the required model weights and place them in the `./models/` directory before running anything.

### Specific Model Downloads

**BiSeNet (Required for FRU Face Parsing):**

- Download `bisenet_face_parsing.pth` from Google Drive Link.
- Place it in: `models/bisenet/bisenet_face_parsing.pth`

**TAESD (Required for fast LCM decoding):**

- Go to the TAESD HuggingFace Repo.
- Download `config.json` and `diffusion_pytorch_model.safetensors`.
- Place both files in: `models/taesd/`

### Full Model Checklist

Ensure your `./models/` directory looks exactly like this:

```
models/
|-- stable-diffusion-v1-5/           # SD1.5 base model (HuggingFace format, needs fp16)
|-- controlnet/
|   |-- control_v11p_sd15_lineart/   # ControlNet lineart model
|-- annotators/
|   |-- lineart/                     # Lineart annotator weights
|   |-- hed/                         # HED annotator weights (fallback)
|-- insightface/
|   |-- models/
|       |-- antelopev2/              # InsightFace antelopev2 model pack
|-- ip_adapter/
|   |-- ip-adapter-faceid_sd15.bin   # IP-Adapter FaceID weights
|-- loras/
|   |-- Pencil_Sketch_by_vizsumit.safetensors   # Style LoRA
|   |-- lcm-lora-sdv1-5/             # LCM-LoRA weights (requires pytorch_lora_weights.safetensors)
|-- taesd/                           # (From link above) config.json & .safetensors
|-- bisenet/
    |-- bisenet_face_parsing.pth     # (From link above)
```

---

## 3. Folder Structure

Here is how your project should be organized once inputs and models are added:

```
project/
|-- pipeline/                    # Core Python pipeline scripts
|   |-- preprocessor.py          # SAU preprocessing
|   |-- generator.py             # SAU generation + scene composition
|   |-- fru_preprocessor.py      # FRU preprocessing
|   |-- fru_generator.py         # FRU generation + composition
|-- models/                      # MUST contain all downloaded weights
|-- inputs/                      # MUST contain your background images
|   |-- scenes/                  # SAU backgrounds
|   |-- scenes/face_bg/          # FRU backgrounds
|-- config.py                    # All paths and tunable defaults
|-- crops.json                   # SAU scene configuration
|-- crops-face-meta.json         # FRU scene configuration
|-- local_test.py                # Script to run tests locally
|-- requirements.txt
```

---

## 4. Scene & Crop Configuration

Before generating, you need background scenes to paste the sketches onto.

### SAU (Full-Body) Scenes

Place background images (PNG/JPG) in `inputs/scenes/`.

Edit `crops.json` to define the scale and floor-anchor point for each scene. The key is the scene filename (without extension).

```json
{
  "park_bench": {
    "scale": 0.25,
    "anchor": [1901, 1159]
  }
}
```

Scale dictates how much the sketch shrinks; anchor `[x, y]` dictates exactly where the bottom-center of their feet touch the floor.

### FRU (Face-Only) Scenes

Place FRU background images in `inputs/scenes/face_bg/`.

Edit `crops-face-meta.json` to define the face anchor point.

```json
{
  "face_scene_1": {
    "gender": ["male", "female"],
    "face_anchor": {
      "center": [1368, 379],
      "face_h": 100
    }
  }
}
```

The sketch is scaled so the detected face height matches `face_h`, and placed so the center of the face lands on `center`.

---

## 5. Running Local Tests

You can test the pipeline end-to-end using `local_test.py`. Outputs (including intermediate preprocessing results like edge maps and masks) are saved to `./local_test_output/`.

**Run SAU (Full-body sketch):**

```bash
python local_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --gender female \
  --place  SAU
```

**Run FRU (Face sketch):**

```bash
python local_test.py \
  --image  /path/to/person.jpg \
  --id     A1234 \
  --gender male \
  --place  FRU
```

**Auto-detect Gender:**  
Leave the `--gender` flag blank and InsightFace will attempt to auto-detect it:

```bash
python local_test.py --image /path/to/person.jpg --id A1234 --place SAU
```

---

## 6. Advanced: Triton Server Deployment

If you are ready to deploy to production, this pipeline is fully configured for NVIDIA Triton Inference Server.

**1. Start the Server:**

```bash
docker compose build
docker compose up -d
```

Model loading takes 30-90 seconds. Watch logs with `docker compose logs -f sketch-triton`.

**2. Test Server Health:**

```bash
curl http://localhost:8000/v2/health/ready
```

**3. Test via Client script:**

```bash
python client_test.py --image person.jpg --id A1234 --gender female --place SAU
```

### MinIO Integration Note

The Triton pipeline accepts `person_id` and `place` as first-class inputs matching the `ID_place` MinIO key structure. You can write a watcher service to subscribe to MinIO events, trigger the Triton HTTP endpoint at `http://<server>:8000/v2/models/sketch_pipeline/infer`, and upload the results back to your bucket.