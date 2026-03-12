"""
FRU Face Sketch Generation Pipeline — Memory-Only
All processing runs in RAM. No files written to disk.
Accepts FRUPreprocessedData, returns FRUGeneratedResult with scene images.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageEnhance

from diffusers import (
    AutoencoderTiny,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionControlNetPipeline,
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
    MODEL_PATHS,
    GENERATION_DEFAULTS,
    POSTPROCESS_PARAMS,
    FRU_SCENES_DIR,
    FRU_CROPS_CONFIG_PATH,
)
from pipeline.fru_preprocessor import (
    FRUPreprocessedData,
    _BiSeNet,
    _BISENET_FACE_ONLY,
    _get_torch_device,   # FIX #2: import instead of calling undefined local name
    _init_bisenet,
)

# ── TF32 optimisation ─────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── Model cache ───────────────────────────────────────────────────────────────
# FIX #2: removed dead "bisenet" key — BiSeNet is owned by fru_preprocessor cache
_MODEL_CACHE: Dict = {
    "pipeline":        None,
    "pipeline_config": None,
    "lora_loaded":     False,
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
# ============================================================================

def _build_prompt(gender: str, use_lora: bool, is_lcm: bool) -> str:
    if use_lora:
        if is_lcm:
            base = (
                "(manga style sketch), (Anime), (rough pencil sketch:1.3), "
                "(graphite texture:1.3), (visible strokes:1.2), hand-drawn portrait, "
                "detailed shading, realistic pencil art, (hatching:1.1), "
                "monochrome drawing, high contrast"
            )
        else:
            base = (
                "pencil sketch, graphite drawing, hand-drawn portrait, detailed shading, "
                "realistic pencil art, professional sketch, soft shadows, clean lines, "
                "artistic portrait, monochrome drawing, traditional art style, "
                "high quality pencil work"
            )
    else:
        if is_lcm:
            base = (
                "(graphite pencil portrait:1.3), smooth shading, bold outlines, "
                "clean facial features, 2B graphite texture, (rough sketch style:1.2), "
                "white background"
            )
        else:
            base = (
                "Hand-drawn graphite pencil portrait, smooth shading, bold outlines, "
                "clean facial features, dark eyelashes and eyebrows, soft skin shading, "
                "2B graphite texture, slightly stylized realism, no cross-hatching, "
                "high contrast highlights, thick pencil strokes around hair and jawline, "
                "clear edges, white background"
            )

    gender_additions = {
        "male":   ", masculine facial features, defined jawline, strong bone structure",
        "female": ", feminine facial features, soft contours, delicate features",
    }
    return base + gender_additions.get(gender, "")


def _build_negative(gender: str, is_lcm: bool) -> str:
    if is_lcm:
        base = (
            "color, colored, photorealistic, photo, (smooth:1.2), blur, blurry, "
            "low quality, distorted, deformed, ugly, watermark"
        )
    else:
        base = (
            "color, colored, photorealistic, photo, blur, blurry, low quality, "
            "distorted, disfigured, deformed, ugly face, bad anatomy, watermark, "
            "signature, text, expressionless, blank stare, neutral face, noise, grainy, "
            "artifacts, pixelated, jpeg artifacts, wrong eyes, malformed mouth, bad teeth, "
            "extra fingers, over-shaded, too dark, muddy shadows, dirty appearance"
        )

    gender_negatives = {
        "male":   ", feminine features, makeup, lipstick",
        "female": ", masculine features, beard, mustache, facial hair, stubble",
    }
    return base + gender_negatives.get(gender, "")


# ============================================================================
# POST-PROCESSING
# ============================================================================

def _post_process_sketch(img: Image.Image, sharpness: float = 3.5,
                          saturation: float = 0.0, exposure: float = 0.9) -> Image.Image:
    img = ImageEnhance.Sharpness(img).enhance(sharpness)
    img = ImageEnhance.Color(img).enhance(saturation)
    img = ImageEnhance.Brightness(img).enhance(exposure)
    return img


# ============================================================================
# FACE MASK APPLICATION
# ============================================================================

def _apply_face_mask(
    sketch_rgb: Image.Image,
    face_mask_crop: Optional[np.ndarray],
) -> Image.Image:
    """
    Apply BiSeNet mask to the generated sketch to produce an RGBA cutout.
    Falls back to rembg silhouette if mask is unavailable.
    Returns RGBA image.
    """
    sketch_w, sketch_h = sketch_rgb.size

    if face_mask_crop is not None:
        mask_resized = Image.fromarray(face_mask_crop).resize(
            (sketch_w, sketch_h), Image.LANCZOS
        )
        mask_arr = np.array(mask_resized)
        mask_arr = cv2.GaussianBlur(mask_arr, (5, 5), 0)
        mask_arr = (mask_arr > 128).astype(np.uint8) * 255
        final_rgba = sketch_rgb.convert("RGBA")
        final_rgba.putalpha(Image.fromarray(mask_arr))
        print("  [INFO] BiSeNet face mask applied to sketch")
        return final_rgba

    # Fallback: rembg → binary threshold → largest contour → solid filled mask
    print("  [WARN] No BiSeNet mask — falling back to rembg silhouette")
    global _REMBG_SESSION
    if _REMBG_SESSION is None and REMBG_AVAILABLE:
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception:
            _REMBG_SESSION = new_session("u2netp")

    sketch_rgba_raw = remove(sketch_rgb, session=_REMBG_SESSION)
    alpha_arr = np.array(sketch_rgba_raw.split()[3])
    binary = (alpha_arr > 10).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask = np.zeros_like(alpha_arr)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [largest], -1, 255, thickness=cv2.FILLED)
    final_rgba = sketch_rgb.convert("RGBA")
    final_rgba.putalpha(Image.fromarray(solid_mask))
    return final_rgba


# ============================================================================
# BISENET RE-DETECTION IN GENERATED SKETCH
# ============================================================================

def _redetect_face_in_sketch(
    sketch_rgb: Image.Image,
    fallback_box: Optional[Tuple],
) -> Optional[Tuple[int, int, int, int]]:
    """
    Re-run BiSeNet on the generated sketch to get the face-only bbox.
    Falls back to fallback_box if re-detection fails.
    """
    bisenet = _init_bisenet()  # delegates to fru_preprocessor cache
    if bisenet is None:
        return fallback_box

    try:
        # FIX #2: _get_torch_device is now imported from fru_preprocessor
        device_bs = _get_torch_device()
        transform_bs = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        orig_w, orig_h = sketch_rgb.size
        tensor_bs = transform_bs(sketch_rgb).unsqueeze(0).to(device_bs)
        with torch.no_grad():
            bs_out, _, _ = bisenet(tensor_bs)
        bs_out_full = F.interpolate(
            bs_out, (orig_h, orig_w), mode="bilinear", align_corners=True
        )
        parsing = bs_out_full.squeeze(0).argmax(0).cpu().numpy()

        fo_mask = np.zeros(parsing.shape, dtype=np.uint8)
        for lid in _BISENET_FACE_ONLY:
            fo_mask[parsing == lid] = 255

        rows = np.any(fo_mask > 0, axis=1)
        cols = np.any(fo_mask > 0, axis=0)
        if rows.any() and cols.any():
            box = (
                int(np.where(cols)[0][0]),
                int(np.where(rows)[0][0]),
                int(np.where(cols)[0][-1]),
                int(np.where(rows)[0][-1]),
            )
            fx1, fy1, fx2, fy2 = box
            print(
                f"  [INFO] BiSeNet re-detected face in sketch: {box} "
                f"(h={fy2 - fy1}px)"
            )
            return box
        else:
            print("  [WARN] BiSeNet found no face in sketch — using preprocessed box")
            return fallback_box

    except Exception as e:
        print(f"  [WARN] BiSeNet re-detection failed: {e} — using preprocessed box")
        return fallback_box


# ============================================================================
# SCENE COMPOSITION
# ============================================================================

def _compose_single_scene(
    sketch_rgba: Image.Image,
    scene_path: Path,
    face_anchor: Dict,
    face_box_in_sketch: Optional[Tuple],
) -> Image.Image:
    """
    Compose sketch into scene using face_anchor alignment.
    Scale is computed so face-only height in sketch matches face_anchor["face_h"].
    """
    scene = Image.open(scene_path).convert("RGBA")
    anchor_cx, anchor_cy = face_anchor["center"]
    target_face_h = face_anchor["face_h"]

    if face_box_in_sketch is not None:
        fx1, fy1, fx2, fy2 = face_box_in_sketch
        actual_face_h = max(fy2 - fy1, 1)
        face_cx_in_sketch = (fx1 + fx2) / 2.0
        face_cy_in_sketch = (fy1 + fy2) / 2.0
    else:
        print("    [WARN] No face_box_in_sketch — using fallback center alignment")
        actual_face_h = sketch_rgba.height // 2
        face_cx_in_sketch = sketch_rgba.width / 2.0
        face_cy_in_sketch = sketch_rgba.height / 2.0

    scale   = target_face_h / actual_face_h
    new_w   = int(sketch_rgba.width  * scale)
    new_h   = int(sketch_rgba.height * scale)
    resized = sketch_rgba.resize((new_w, new_h), Image.LANCZOS)

    offset_x = int(round(anchor_cx - face_cx_in_sketch * scale))
    offset_y = int(round(anchor_cy - face_cy_in_sketch * scale))

    final_scene = scene.copy()
    final_scene.paste(resized, (offset_x, offset_y), resized)
    return final_scene.convert("RGB")


def _run_scene_composition(
    sketch_rgba: Image.Image,
    gender: str,
    face_box_in_sketch: Optional[Tuple],
) -> List[Image.Image]:
    """
    Compose the face sketch into all FRU scene backgrounds.
    Returns list of PIL Images — no files saved.
    """
    if not FRU_SCENES_DIR.exists():
        print(f"  [INFO] No FRU scenes directory at {FRU_SCENES_DIR}, skipping.")
        return []

    crops_data: Dict = {}
    if FRU_CROPS_CONFIG_PATH.exists():
        with open(FRU_CROPS_CONFIG_PATH) as f:
            crops_data = json.load(f)
    else:
        print("[WARN] crops-face-meta.json not found, skipping scene composition.")
        return []

    scene_files: List[Path] = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        scene_files.extend(FRU_SCENES_DIR.glob(ext))

    def process_scene(scene_file: Path) -> Optional[Image.Image]:
        scene_id = scene_file.stem
        if scene_id not in crops_data:
            return None

        scene_meta = crops_data[scene_id]
        allowed_genders = scene_meta.get("gender", ["male", "female", "unknown"])
        if gender not in allowed_genders:
            print(f"    — Skipped scene {scene_id} (gender mismatch)")
            return None

        face_anchor = scene_meta.get("face_anchor")
        if not face_anchor:
            print(f"    — Skipped scene {scene_id} (no face_anchor in config)")
            return None

        try:
            return _compose_single_scene(
                sketch_rgba, scene_file, face_anchor, face_box_in_sketch
            )
        except Exception as e:
            print(f"    [WARN] Scene {scene_id} failed: {e}")
            return None

    results: List[Image.Image] = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_scene, sf): sf for sf in scene_files}
        for future in as_completed(futures):
            img = future.result()
            if img is not None:
                results.append(img)

    print(f"  [INFO] {len(results)} FRU scene(s) composed")
    return results


# ============================================================================
# MODEL LOADING
# ============================================================================

def _verify_local_models():
    missing = [
        f"  [ERROR] {n}: {MODEL_PATHS[n]}"
        for n in ["sd15", "controlnet"]
        if not Path(MODEL_PATHS[n]).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(missing))


def _load_pipeline(
    device: str,
    dtype: torch.dtype,
    load_lora: bool = False,
    lora_scale: float = 1.0,
    use_lcm: bool = True,
) -> Tuple:
    """
    Loads and caches the SD pipeline.
    Returns (pipe, lora_loaded: bool).
    """
    config_key = f"fru_{device}_{dtype}_{load_lora}_{lora_scale}_{use_lcm}"
    if (
        _MODEL_CACHE["pipeline"] is not None
        and _MODEL_CACHE["pipeline_config"] == config_key
    ):
        print("[INFO] Using cached FRU pipeline")
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

    # TAESD
    if use_lcm:
        try:
            pipe.vae = AutoencoderTiny.from_pretrained(
                MODEL_PATHS["taesd"], torch_dtype=dtype
            )
            print("[INFO] TAESD loaded")
        except Exception as e:
            print(f"[WARN] TAESD loading failed: {e}")

    # LCM-LoRA + optional style LoRA
    lora_loaded = False
    if use_lcm:
        lcm_path = MODEL_PATHS["lcm_lora"]
        pipe.load_lora_weights(
            lcm_path, weight_name="pytorch_lora_weights.safetensors"
        )
        pipe.fuse_lora(lora_scale=1.0)
        print("[INFO] LCM-LoRA loaded and fused")

        if load_lora and Path(MODEL_PATHS["lora"]).exists():
            pipe.load_lora_weights(
                MODEL_PATHS["lora"],
                weight_name="Pencil_Sketch_by_vizsumit.safetensors",
            )
            pipe.fuse_lora(lora_scale=lora_scale * 1.3)
            lora_loaded = True
            print("[INFO] Style LoRA loaded and fused")

        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        print("[INFO] LCM Scheduler activated")

    # FIX #2: pipe.to(device) called only once here — removed redundant
    # second .to(device) that was happening inside IPAdapterFaceID.__init__
    pipe = pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing()

    _MODEL_CACHE["pipeline"]        = pipe
    _MODEL_CACHE["pipeline_config"] = config_key
    _MODEL_CACHE["lora_loaded"]     = lora_loaded

    return pipe, lora_loaded


def _load_ip_adapter(pipe, device: str, dtype: torch.dtype):
    if not FACEID_AVAILABLE:
        return None
    try:
        # FIX #2: pass device string only; IPAdapterFaceID.__init__ will call
        # sd_pipe.to(device) internally but the pipe is already on the correct
        # device so it's a no-op move — harmless and unavoidable given the
        # IPAdapterFaceID API. We do NOT double-move manually here.
        return IPAdapterFaceID(
            pipe, MODEL_PATHS["ip_adapter"], device, num_tokens=4, torch_dtype=dtype
        )
    except Exception as e:
        print(f"[WARN] IP-Adapter loading failed: {e}")
        return None


# ============================================================================
# CORE GENERATION
# ============================================================================

def _generate_single_sketch(
    pipe,
    ip_adapter,
    controlnet_img: Image.Image,
    faceid_embeds: Optional[torch.Tensor],
    faceid_strength: float,
    device: str,
    dtype: torch.dtype,
    seed: int,
    use_lora: bool,
    gender: str,
    use_lcm: bool,
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
# FRUGeneratedResult — output container
# ============================================================================

class FRUGeneratedResult:
    """Holds all FRU output images. No files written to disk."""

    def __init__(
        self,
        final_sketch: Image.Image,
        scene_images: List[Image.Image],
    ):
        self.final_sketch = final_sketch
        self.scene_images = scene_images


# ============================================================================
# MAIN FRU GENERATION FUNCTION
# ============================================================================

def generate_fru_sketch_in_memory(
    data: FRUPreprocessedData,
    use_lora: bool        = GENERATION_DEFAULTS["use_lora"],
    lora_scale: float     = GENERATION_DEFAULTS["lora_scale"],
    faceid_strength: float = GENERATION_DEFAULTS["faceid_strength"],
    device: Optional[str] = None,
    use_lcm: bool         = GENERATION_DEFAULTS["use_lcm"],
) -> FRUGeneratedResult:
    """
    Generate face sketch from FRUPreprocessedData entirely in RAM.

    1. Generates face sketch via SD1.5 + ControlNet + IP-Adapter FaceID
    2. Applies BiSeNet mask to sketch (or rembg fallback) → RGBA cutout
    3. Re-detects face position in generated sketch via BiSeNet
    4. Composes into all FRU scene backgrounds using face_anchor alignment
    5. Returns FRUGeneratedResult whose .scene_images is sent back to Triton
    """
    if data.face_img is None or data.face_edges is None:
        raise RuntimeError(
            "FRUPreprocessedData has no face crop — cannot generate face sketch."
        )

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

    print(f"\n[INFO] Generating face sketch (gender={data.gender})...")
    face_sketch = _generate_single_sketch(
        pipe, ip_adapter, data.face_edges,
        faceid_embeds, faceid_strength,
        device, dtype,
        seed=5678,
        use_lora=lora_loaded,
        gender=data.gender,
        use_lcm=use_lcm,
    )

    # Post-process
    final_sketch_rgb = _post_process_sketch(
        face_sketch.copy(),
        sharpness=POSTPROCESS_PARAMS["sharpness"],
        saturation=POSTPROCESS_PARAMS["saturation"],
        exposure=POSTPROCESS_PARAMS["exposure"],
    )

    # Apply BiSeNet mask → RGBA cutout
    final_sketch_rgba = _apply_face_mask(final_sketch_rgb, data.face_mask_crop)

    # Re-detect face position in the generated sketch
    print("\n[INFO] Re-detecting face position in generated sketch...")
    face_box_final = _redetect_face_in_sketch(
        final_sketch_rgb, fallback_box=data.face_box_in_sketch
    )

    # Scene composition
    scene_images: List[Image.Image] = _run_scene_composition(
        final_sketch_rgba,
        gender=data.gender,
        face_box_in_sketch=face_box_final,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FRUGeneratedResult(
        final_sketch=final_sketch_rgb,
        scene_images=scene_images,
    )