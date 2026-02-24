"""
Portrait Sketch Generation Pipeline — Memory-Only
All processing runs in RAM. No files written to disk.
Returns List[PIL.Image] of scene-composed images.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from concurrent.futures import ThreadPoolExecutor, as_completed

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    LCMScheduler,
    AutoencoderTiny,
)

try:
    from ip_adapter.ip_adapter_faceid import IPAdapterFaceID
    FACEID_AVAILABLE = True
except Exception:
    FACEID_AVAILABLE = False

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# ── Config + sibling module import ────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MODEL_PATHS, GENERATION_DEFAULTS, POSTPROCESS_PARAMS,
    SCENES_DIR, CROPS_CONFIG_PATH,
)
from pipeline.preprocessor import PreprocessedData

# ── TF32 optimisation (no xformers) ──────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── Model cache ───────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict = {
    "pipeline":        None,
    "pipeline_config": None,
    "lora_loaded":     False,   # tracks whether LoRA was baked in on last load
}
_REMBG_SESSION = None


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def _get_optimal_device() -> Tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        print("[INFO] Using MPS (Apple Silicon)")
        return "mps", torch.float32
    if torch.cuda.is_available():
        print("[INFO] Using CUDA")
        return "cuda", torch.float16
    print("[INFO] Using CPU (slow)")
    return "cpu", torch.float32


# ============================================================================
# PROMPT ENGINEERING
# Gender-specific, no beard logic anywhere.
# 'unknown' gender gets a neutral prompt — controlnet + embedding guide identity.
# ============================================================================

_BASE_SKETCH_LCM = (
    "(rough pencil sketch:1.3), (graphite texture:1.3), (visible strokes:1.2), "
    "hand-drawn portrait, detailed shading, realistic pencil art, "
    "(hatching:1.1), monochrome drawing, high contrast, white background"
)
_BASE_SKETCH_STANDARD = (
    "pencil sketch, graphite drawing, hand-drawn portrait, detailed shading, "
    "realistic pencil art, professional sketch, soft shadows, clean lines, "
    "artistic portrait, monochrome drawing, traditional art style, "
    "high quality pencil work, white background"
)
_BASE_SKETCH_NO_LORA_LCM = (
    "(graphite pencil portrait:1.3), smooth shading, bold outlines, "
    "clean facial features, 2B graphite texture, (rough sketch style:1.2), white background"
)
_BASE_SKETCH_NO_LORA_STANDARD = (
    "Hand-drawn graphite pencil portrait, smooth shading, bold outlines, "
    "clean facial features, dark eyelashes and eyebrows, soft skin shading, "
    "2B graphite texture, slightly stylized realism, no cross-hatching, "
    "high contrast highlights, thick pencil strokes around hair and jawline, "
    "clear edges, white background"
)

_GENDER_POSITIVE: Dict[str, str] = {
    "male": (
        ", masculine facial features, defined jawline, strong bone structure, "
        "sharp cheekbones, structured brow ridge"
    ),
    "female": (
        ", feminine facial features, soft contours, delicate features, "
        "graceful jawline, expressive eyes, soft skin shading"
    ),
    "unknown": "",  # neutral — let controlnet + embedding guide identity
}

_GENDER_NEGATIVE: Dict[str, str] = {
    "male": (
        ", feminine features, makeup, lipstick, soft delicate features, "
        "smooth rounded jaw"
    ),
    "female": (
        ", masculine features, beard, mustache, facial hair, "
        "strong heavy jawline, thick brow ridge"
    ),
    "unknown": "",
}

_BASE_NEGATIVE = (
    "color, colored, photorealistic, photo, blur, blurry, low quality, "
    "distorted, disfigured, deformed, ugly face, bad anatomy, watermark, "
    "signature, text, expressionless, blank stare, neutral face, noise, grainy, "
    "artifacts, pixelated, jpeg artifacts, wrong eyes, malformed mouth, bad teeth, "
    "extra fingers, over-shaded, too dark, muddy shadows, dirty appearance"
)
_BASE_NEGATIVE_LCM = (
    "color, colored, photorealistic, photo, (smooth:1.2), blur, blurry, "
    "low quality, distorted, deformed, ugly, watermark, bad anatomy, "
    "extra fingers, malformed limbs"
)


def _build_prompt(gender: str, use_lora: bool, is_lcm: bool) -> str:
    if use_lora:
        base = _BASE_SKETCH_LCM if is_lcm else _BASE_SKETCH_STANDARD
    else:
        base = _BASE_SKETCH_NO_LORA_LCM if is_lcm else _BASE_SKETCH_NO_LORA_STANDARD
    return base + _GENDER_POSITIVE.get(gender, "")


def _build_negative(gender: str, is_lcm: bool) -> str:
    base = _BASE_NEGATIVE_LCM if is_lcm else _BASE_NEGATIVE
    return base + _GENDER_NEGATIVE.get(gender, "")


# ============================================================================
# POST-PROCESSING
# ============================================================================

def _post_process_sketch(img: Image.Image, sharpness: float = 2.0,
                         saturation: float = 0.0, exposure: float = 0.75) -> Image.Image:
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Brightness(img).enhance(exposure)
    return img


# ============================================================================
# FACE BLENDING
# ============================================================================

def _blend_face_sketch(fullbody: Image.Image, face_sketch: Image.Image,
                       face_crop_box: Tuple, blend_margin: int = 20) -> Image.Image:
    cx1, cy1, cx2, cy2 = face_crop_box
    face_resized = face_sketch.resize((cx2 - cx1, cy2 - cy1), Image.LANCZOS)
    result = fullbody.copy()
    mask = np.ones((face_resized.height, face_resized.width), dtype=float) * 255
    for i in range(blend_margin):
        alpha = i / blend_margin
        if i < mask.shape[0]:
            mask[i, :] *= alpha
        if mask.shape[0] - i - 1 >= 0:
            mask[mask.shape[0] - i - 1, :] *= alpha
        if i < mask.shape[1]:
            mask[:, i] *= alpha
        if mask.shape[1] - i - 1 >= 0:
            mask[:, mask.shape[1] - i - 1] *= alpha
    result.paste(face_resized, (cx1, cy1), Image.fromarray(mask.astype(np.uint8)))
    return result


# ============================================================================
# MODEL LOADING
# ============================================================================

def _verify_local_models():
    missing = [
        f"  ✗ {n}: {MODEL_PATHS[n]}"
        for n in ["sd15", "controlnet"]
        if not Path(MODEL_PATHS[n]).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing))


def _load_pipeline(device: str, dtype: torch.dtype, load_lora: bool = False,
                   lora_scale: float = 1.0, use_lcm: bool = True) -> Tuple:
    """
    Loads and caches the SD pipeline.
    Returns (pipe, lora_loaded: bool).
    lora_loaded reflects what was actually baked in — from cache or fresh load.
    """
    config_key = f"{device}_{dtype}_{load_lora}_{lora_scale}_{use_lcm}"
    if (_MODEL_CACHE["pipeline"] is not None
            and _MODEL_CACHE["pipeline_config"] == config_key):
        print("[INFO] Using cached pipeline")
        return _MODEL_CACHE["pipeline"], _MODEL_CACHE["lora_loaded"]

    _verify_local_models()

    controlnet = ControlNetModel.from_pretrained(
        MODEL_PATHS["controlnet"], torch_dtype=dtype
    )

    if device == "mps":
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_PATHS["sd15"], controlnet=controlnet, torch_dtype=dtype,
            safety_checker=None, use_safetensors=True,
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_PATHS["sd15"], controlnet=controlnet, torch_dtype=dtype,
            safety_checker=None,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
        )

    # ── TAESD (tiny VAE for fast LCM decoding) ────────────────────────────────
    if use_lcm:
        try:
            pipe.vae = AutoencoderTiny.from_pretrained(
                MODEL_PATHS["taesd"], torch_dtype=dtype
            )
            print("[INFO] TAESD loaded")
        except Exception as e:
            print(f"[WARN] TAESD loading failed: {e}")

    # ── LoRA loading ──────────────────────────────────────────────────────────
    lora_loaded = False
    if use_lcm:
        try:
            lcm_path = (MODEL_PATHS["lcm_lora"]
                        if Path(MODEL_PATHS["lcm_lora"]).exists()
                        else "latent-consistency/lcm-lora-sdv1-5")
            pipe.load_lora_weights(lcm_path)
            try:
                pipe.fuse_lora(lora_scale=1.0)
            except Exception:
                pass

            if load_lora and Path(MODEL_PATHS["lora"]).exists():
                pipe.load_lora_weights(MODEL_PATHS["lora"])
                try:
                    pipe.fuse_lora(lora_scale=lora_scale * 1.3)
                    lora_loaded = True
                    print(f"[INFO] Style LoRA loaded and fused (scale={lora_scale * 1.3:.2f})")
                except Exception:
                    pass

            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            print("[INFO] LCM scheduler activated")
        except Exception as e:
            print(f"[WARN] LCM setup failed: {e}")

    pipe = pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing()
        print("[INFO] MPS attention slicing enabled")

    _MODEL_CACHE["pipeline"]        = pipe
    _MODEL_CACHE["pipeline_config"] = config_key
    _MODEL_CACHE["lora_loaded"]     = lora_loaded  # persist actual state for cache hits

    return pipe, lora_loaded


def _load_ip_adapter(pipe, device: str, dtype: torch.dtype):
    if not FACEID_AVAILABLE:
        return None
    try:
        return IPAdapterFaceID(
            pipe, MODEL_PATHS["ip_adapter"], device, num_tokens=4, torch_dtype=dtype
        )
    except Exception as e:
        print(f"[WARN] IP-Adapter loading failed: {e}")
        return None


# ============================================================================
# SCENE COMPOSITION
# ============================================================================

def _init_rembg():
    global _REMBG_SESSION
    if _REMBG_SESSION is None and REMBG_AVAILABLE:
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception:
            _REMBG_SESSION = new_session("u2netp")


def _remove_sketch_background(img: Image.Image) -> Image.Image:
    if not REMBG_AVAILABLE:
        return img.convert("RGBA")
    _init_rembg()
    if img.mode != "RGBA":
        img = img.convert("RGB")
    return remove(img, session=_REMBG_SESSION)


def _compose_single_scene(sketch_rgba: Image.Image, scene_path: Path,
                           bbox: List[int]) -> Image.Image:
    scene = Image.open(scene_path).convert("RGB")
    x1, y1, x2, y2 = bbox
    target_w, target_h = x2 - x1, y2 - y1
    sketch_w, sketch_h = sketch_rgba.size
    ratio = max(target_w / sketch_w, target_h / sketch_h)
    new_w, new_h = int(sketch_w * ratio), int(sketch_h * ratio)
    resized = sketch_rgba.resize((new_w, new_h), Image.LANCZOS)
    offset_x = x1 + (target_w - new_w) // 2
    offset_y = y1 + (target_h - new_h) // 2
    final_scene = scene.copy().convert("RGBA")
    final_scene.paste(resized, (offset_x, offset_y), resized)
    return final_scene.convert("RGB")


def run_scene_composition_in_memory(final_sketch: Image.Image) -> List[Image.Image]:
    """
    Composes the sketch into all available scene backgrounds.
    Returns a list of PIL Images — no files saved.
    """
    if not SCENES_DIR.exists():
        print(f"[INFO] No scenes directory at {SCENES_DIR}, skipping composition.")
        return []

    crops_data: Dict = {}
    for path in [CROPS_CONFIG_PATH, Path("config/crops.json")]:
        if path.exists():
            with open(path) as f:
                crops_data = json.load(f)
            break

    if not crops_data:
        print("[WARN] crops.json not found, skipping scene composition.")
        return []

    sketch_transparent = _remove_sketch_background(final_sketch)

    scene_files: List[Path] = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        scene_files.extend(SCENES_DIR.glob(ext))

    def process_scene(scene_file: Path) -> Optional[Image.Image]:
        scene_id = scene_file.stem
        if scene_id in crops_data:
            try:
                # Load scene fresh inside each thread (PIL lazy-open is not thread-safe)
                return _compose_single_scene(
                    sketch_transparent, scene_file, crops_data[scene_id][0]
                )
            except Exception as e:
                print(f"  ✗ Scene {scene_id} failed: {e}")
        return None

    results: List[Image.Image] = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_scene, sf): sf for sf in scene_files}
        for future in as_completed(futures):
            img = future.result()
            if img is not None:
                results.append(img)

    print(f"  ✓ {len(results)} scene(s) composed")
    return results


# ============================================================================
# CORE GENERATION
# ============================================================================

def _generate_single_sketch(
    img: Image.Image,
    pipe,
    ip_adapter,
    controlnet_img: Image.Image,
    faceid_embeds: Optional[torch.Tensor],
    faceid_strength: float,
    device: str,
    dtype: torch.dtype,
    seed: int,
    use_lora: bool,
    gender: str = "unknown",
    use_lcm: bool = True,
) -> Image.Image:
    is_lcm = isinstance(pipe.scheduler, LCMScheduler)

    prompt   = _build_prompt(gender, use_lora, is_lcm)
    negative = _build_negative(gender, is_lcm)
    print(f"  [sketch] gender={gender} | {prompt[:80]}...")

    generator = torch.Generator(
        device="cpu" if device == "mps" else device
    ).manual_seed(seed)

    if is_lcm:
        num_steps, guidance, ctrl_scale = 4, 2.5, 0.95
    else:
        num_steps, guidance, ctrl_scale = 20, 7.0, 0.9

    if ip_adapter and faceid_embeds is not None:
        try:
            images = ip_adapter.generate(
                faceid_embeds=faceid_embeds.to(device=device, dtype=dtype),
                prompt=prompt,
                negative_prompt=negative,
                num_samples=1,
                seed=seed,
                guidance_scale=guidance,
                num_inference_steps=num_steps,
                scale=faceid_strength,
                image=controlnet_img,
                controlnet_conditioning_scale=ctrl_scale,
                width=controlnet_img.width,
                height=controlnet_img.height,
            )
            return images[0]
        except Exception as e:
            print(f"[WARN] IP-Adapter failed, falling back to base pipe: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=controlnet_img,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            controlnet_conditioning_scale=ctrl_scale,
            generator=generator,
            width=controlnet_img.width,
            height=controlnet_img.height,
        )
    return result.images[0]


# ============================================================================
# GeneratedResult — output container
# ============================================================================

class GeneratedResult:
    """Holds all output images. No files written to disk."""

    def __init__(
        self,
        final_sketch: Image.Image,
        face_sketch: Optional[Image.Image],
        scene_images: List[Image.Image],
    ):
        self.final_sketch  = final_sketch
        self.face_sketch   = face_sketch
        self.scene_images  = scene_images   # these are what Triton returns to the caller


# ============================================================================
# MAIN GENERATION FUNCTION
# ============================================================================

def generate_sketch_in_memory(
    data: PreprocessedData,
    use_lora: bool     = GENERATION_DEFAULTS["use_lora"],
    lora_scale: float  = GENERATION_DEFAULTS["lora_scale"],
    faceid_strength: float = GENERATION_DEFAULTS["faceid_strength"],
    device: Optional[str] = None,
    use_lcm: bool      = GENERATION_DEFAULTS["use_lcm"],
) -> GeneratedResult:
    """
    Generate sketch from PreprocessedData entirely in RAM.

    Gender is read directly from data.gender, which is already resolved
    upstream (JSON override > InsightFace detection > 'unknown').
    No beard logic anywhere in this function.

    Returns GeneratedResult whose .scene_images is what gets sent back
    to the Triton client.
    """
    if device is None:
        device, dtype = _get_optimal_device()
    else:
        dtype = torch.float32 if device in ("cpu", "mps") else torch.float16

    faceid_embeds: Optional[torch.Tensor] = None
    if data.has_face_embedding:
        faceid_embeds = torch.from_numpy(data.faceid_embedding).unsqueeze(0)

    pipe, lora_loaded = _load_pipeline(
        device, dtype, load_lora=use_lora, lora_scale=lora_scale, use_lcm=use_lcm
    )
    ip_adapter = _load_ip_adapter(pipe, device, dtype) if faceid_embeds is not None else None

    print(f"\n[INFO] Generating body sketch (gender={data.gender})...")
    body_sketch = _generate_single_sketch(
        data.enhanced, pipe, ip_adapter, data.body_edges,
        faceid_embeds, faceid_strength, device, dtype,
        seed=1234, use_lora=lora_loaded,
        gender=data.gender, use_lcm=use_lcm,
    )
    body_sketch = body_sketch.resize(data.original_size, Image.LANCZOS)
    final_sketch = body_sketch

    face_sketch: Optional[Image.Image] = None
    if data.primary_face and data.face_img is not None and data.face_edges is not None:
        print(f"[INFO] Generating face sketch (gender={data.gender})...")
        face_sketch = _generate_single_sketch(
            data.face_img, pipe, ip_adapter, data.face_edges,
            faceid_embeds, faceid_strength, device, dtype,
            seed=5678, use_lora=lora_loaded,
            gender=data.gender, use_lcm=use_lcm,
        )
        final_sketch = _blend_face_sketch(
            body_sketch, face_sketch, data.face_crop_box, blend_margin=30
        )

    final_sketch = _post_process_sketch(
        final_sketch,
        sharpness=POSTPROCESS_PARAMS["sharpness"],
        saturation=POSTPROCESS_PARAMS["saturation"],
        exposure=POSTPROCESS_PARAMS["exposure"],
    )

    scene_images: List[Image.Image] = []
    if REMBG_AVAILABLE:
        scene_images = run_scene_composition_in_memory(final_sketch)
    else:
        print("[WARN] rembg not available, skipping scene composition")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return GeneratedResult(
        final_sketch=final_sketch,
        face_sketch=face_sketch,
        scene_images=scene_images,
    )
