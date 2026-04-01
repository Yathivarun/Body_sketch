"""
FRU Face Sketch Generation Pipeline - Memory-Only
All processing runs in RAM. No files written to disk.
Accepts FRUPreprocessedData, returns FRUGeneratedResult with scene images.
"""

import json
import logging
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
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
    # USE_PEFT_BACKEND is True only when diffusers has verified that peft is
    # installed at a compatible version and has wired its LoRA loader to use the
    # PEFT backend.  A bare `import peft` check is insufficient: older diffusers
    # builds keep the legacy monkey-patching path even when peft is present,
    # causing set_adapters() to raise "PEFT backend is required".
    from diffusers.utils import USE_PEFT_BACKEND as _USE_PEFT_BACKEND
    PEFT_AVAILABLE = bool(_USE_PEFT_BACKEND)
except ImportError:
    # diffusers < 0.22 doesn't expose USE_PEFT_BACKEND; at that age fuse_lora()
    # is always the right path anyway.
    PEFT_AVAILABLE = False

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# -- Config + sibling module import --------------------------------------------
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
    _BISENET_MEAN,
    _BISENET_STD,
    _get_torch_device,
    _MODEL_CACHE as _PREPROCESSOR_CACHE,
)

logger = logging.getLogger(__name__)

# -- TF32 optimisation ---------------------------------------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -- Model cache ---------------------------------------------------------------
_MODEL_CACHE: Dict = {
    "pipeline":        None,
    "pipeline_config": None,
    "lora_loaded":     False,
}
_REMBG_SESSION = None

# -- Face-only class IDs for tight re-detection bbox --------------------------
# Strict subset of _BISENET_FACE_IDS: excludes hair (17) and neck (14) so that
# the anchor box used for scene placement tracks the actual face geometry, not
# the full head silhouette.
_BISENET_FACE_ONLY_IDS = frozenset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def _get_optimal_device() -> Tuple[str, torch.dtype]:
    if torch.backends.mps.is_available():
        logger.info("Using MPS (Apple Silicon)")
        return "mps", torch.float32
    if torch.cuda.is_available():
        logger.info("Using CUDA")
        return "cuda", torch.float16
    logger.info("Using CPU (slow)")
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
        logger.info("BiSeNet face mask applied to sketch")
        return final_rgba

    # Fallback: rembg -> binary threshold -> largest contour -> solid filled mask
    logger.warning("No BiSeNet mask — falling back to rembg silhouette")
    global _REMBG_SESSION
    if _REMBG_SESSION is None and REMBG_AVAILABLE:
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception:
            _REMBG_SESSION = new_session("u2netp")

    sketch_rgba_raw = remove(sketch_rgb, session=_REMBG_SESSION)
    alpha_arr = np.array(sketch_rgba_raw.split()[3])
    binary    = (alpha_arr > 10).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid_mask  = np.zeros_like(alpha_arr)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid_mask, [largest], -1, 255, thickness=cv2.FILLED)
    final_rgba = sketch_rgb.convert("RGBA")
    final_rgba.putalpha(Image.fromarray(solid_mask))
    return final_rgba


# ============================================================================
# LANDMARK RE-DETECTION IN GENERATED SKETCH
# ============================================================================

def _redetect_landmarks_in_sketch(
    sketch_rgb: Image.Image,
    original_landmarks: Optional[np.ndarray],
    face_crop_box: Optional[Tuple],
) -> Optional[np.ndarray]:
    """
    Detect 5-point facial keypoints in the generated RGB sketch via InsightFace.

    Returns an (5, 2) float32 ndarray of keypoints in sketch-pixel coordinates,
    or None if detection fails and no fallback is available.

    Fallback strategy
    -----------------
    If InsightFace cannot detect a face in the (abstract) lineart sketch, the
    function re-uses ``original_landmarks`` from the preprocessor — which were
    recorded in the coordinate space of the *original cropped image* — and
    re-maps them into the sketch's coordinate space by subtracting the top-left
    corner of ``face_crop_box`` and then scaling proportionally to the sketch
    dimensions.

    Args:
        sketch_rgb:          Generated RGB sketch (PIL Image).
        original_landmarks:  FRUPreprocessedData.original_landmarks — (5,2)
                             float32 array in the original-crop coordinate space,
                             or None if InsightFace failed during preprocessing.
        face_crop_box:       FRUPreprocessedData.face_crop_box — (cx1, cy1, cx2, cy2)
                             bounding box used to cut the face crop from the full
                             image.  Required to offset landmarks correctly.
    """
    insightface_app = _PREPROCESSOR_CACHE.get("insightface_app")
    if insightface_app is not None:
        try:
            sketch_bgr = cv2.cvtColor(np.array(sketch_rgb.convert("RGB")), cv2.COLOR_RGB2BGR)
            faces = insightface_app.get(sketch_bgr)
            if faces:
                # Pick the most central face
                sw, sh = sketch_rgb.size
                cx, cy = sw / 2.0, sh / 2.0
                best_face = min(
                    faces,
                    key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2
                                  + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2,
                )
                if best_face.kps is not None:
                    logger.info(
                        "InsightFace re-detected landmarks in sketch (left_eye=%s)",
                        best_face.kps[0],
                    )
                    return best_face.kps.astype(np.float32)
        except Exception:
            logger.warning(
                "InsightFace re-detection in sketch failed — will try fallback",
                exc_info=True,
            )
    else:
        logger.warning(
            "InsightFace app not in preprocessor cache — skipping sketch re-detection"
        )

    # ---- Fallback: remap original_landmarks into sketch coordinate space ----
    if original_landmarks is None or face_crop_box is None:
        logger.warning(
            "No original_landmarks or face_crop_box available — "
            "landmark-based composition cannot proceed"
        )
        return None

    cx1, cy1, cx2, cy2 = face_crop_box
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1
    if crop_w <= 0 or crop_h <= 0:
        logger.warning("face_crop_box has zero area — landmark fallback aborted")
        return None

    sketch_w, sketch_h = sketch_rgb.size
    scale_x = sketch_w / crop_w
    scale_y = sketch_h / crop_h

    # Translate from full-image coords to crop-local coords, then scale to sketch
    fallback_kps = original_landmarks.copy().astype(np.float32)
    fallback_kps[:, 0] = (fallback_kps[:, 0] - cx1) * scale_x
    fallback_kps[:, 1] = (fallback_kps[:, 1] - cy1) * scale_y

    logger.info(
        "Using fallback landmarks (scaled from original): left_eye=%s, right_eye=%s",
        fallback_kps[0],
        fallback_kps[1],
    )
    return fallback_kps


# ============================================================================
# SCENE COMPOSITION  —  Landmark-Based Scaling
# Uses inter-ocular distance + eye midpoint + tilt angle for pixel-perfect
# face placement.  Scene JSON now provides target_eye_midpoint,
# target_eye_distance, and target_tilt_angle instead of the old face_anchor.
# ============================================================================

def _compose_single_scene(
    sketch_rgba: Image.Image,
    face_mask_crop: Optional[np.ndarray],
    scene_path: Path,
    target_eye_midpoint: Tuple[float, float],
    target_eye_distance: float,
    target_tilt_angle: float,
    sketch_kps: np.ndarray,
) -> Image.Image:
    """
    Compose the face sketch into a scene background using landmark-based scaling.

    Pipeline
    --------
    1.  Derive actual_eye_distance and actual_eye_midpoint from sketch_kps.
    2.  Calculate sketch_angle (current eye-line tilt) via atan2.
    3.  Scale the sketch (and its BiSeNet RGBA mask) so actual_eye_distance
        matches target_eye_distance.
    4.  Rotate the scaled sketch to align with target_tilt_angle; PIL's
        expand=True prevents clipping and the eye midpoint is recomputed in
        the rotated frame.
    5.  Paste onto the scene so the final eye midpoint lands exactly on
        target_eye_midpoint.

    Args:
        sketch_rgba:         RGBA sketch cutout (PIL Image).
        face_mask_crop:      Optional BiSeNet mask (np.ndarray) for resizing
                             in sync with the sketch — may be None.
        scene_path:          Path to the background scene image.
        target_eye_midpoint: (x, y) pixel position in the scene where the
                             mid-point between the two eyes should land.
        target_eye_distance: Desired Euclidean distance (px) between left and
                             right eye landmarks in the final composite.
        target_tilt_angle:   Desired rotation of the eye-line in degrees
                             (counter-clockwise positive, matching PIL convention).
        sketch_kps:          (5, 2) float32 array: [left_eye, right_eye, …]
                             in sketch-pixel coordinates.
    """
    scene = Image.open(scene_path).convert("RGBA")

    # ------------------------------------------------------------------
    # 1.  Actual eye geometry in the sketch
    # ------------------------------------------------------------------
    left_eye  = sketch_kps[0].astype(np.float64)   # (x, y)
    right_eye = sketch_kps[1].astype(np.float64)   # (x, y)

    actual_eye_distance: float = float(np.linalg.norm(right_eye - left_eye))
    if actual_eye_distance < 1.0:
        actual_eye_distance = 1.0  # guard against degenerate keypoints

    actual_eye_midpoint: Tuple[float, float] = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0,
    )

    # ------------------------------------------------------------------
    # 2.  Current eye-line tilt in the sketch (degrees, CCW positive)
    # ------------------------------------------------------------------
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    sketch_angle_deg: float = math.degrees(math.atan2(-dy, dx))   # CCW, screen coords

    # ------------------------------------------------------------------
    # 3.  Scale so actual_eye_distance → target_eye_distance
    # ------------------------------------------------------------------
    scale: float = float(target_eye_distance) / actual_eye_distance

    new_w = max(1, int(round(sketch_rgba.width  * scale)))
    new_h = max(1, int(round(sketch_rgba.height * scale)))
    scaled_sketch = sketch_rgba.resize((new_w, new_h), Image.LANCZOS)

    # Scaled eye midpoint (in the scaled-but-not-yet-rotated sketch space)
    scaled_mid_x: float = actual_eye_midpoint[0] * scale
    scaled_mid_y: float = actual_eye_midpoint[1] * scale

    # ------------------------------------------------------------------
    # 4.  Rotate to match target_tilt_angle
    #     PIL rotates counter-clockwise when angle > 0.
    #     We want the eye-line to end up at target_tilt_angle, so the
    #     rotation to apply is the difference between the two angles.
    # ------------------------------------------------------------------
    rotation_deg: float = target_tilt_angle - sketch_angle_deg

    # PIL Image.rotate with expand=True returns a larger canvas that
    # contains the full rotated image without clipping.
    rotated_sketch = scaled_sketch.rotate(rotation_deg, expand=True, resample=Image.BICUBIC)

    # Compute how PIL shifted the canvas origin so we can track the
    # eye midpoint into the new (expanded) coordinate frame.
    # The pivot is the centre of the *scaled* image.
    pivot_x = new_w / 2.0
    pivot_y = new_h / 2.0

    # Translate midpoint to pivot-relative coordinates
    rel_x = scaled_mid_x - pivot_x
    rel_y = scaled_mid_y - pivot_y

    # Apply 2-D rotation (CCW by rotation_deg)
    rad = math.radians(rotation_deg)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    rot_rel_x =  cos_r * rel_x + sin_r * rel_y   # PIL x-axis is left-to-right
    rot_rel_y = -sin_r * rel_x + cos_r * rel_y   # PIL y-axis is top-to-bottom

    # The expanded canvas is centred on the rotated image; its size is:
    rot_canvas_w = rotated_sketch.width
    rot_canvas_h = rotated_sketch.height

    # Midpoint in the expanded canvas coordinate frame
    final_mid_x: float = rot_canvas_w / 2.0 + rot_rel_x
    final_mid_y: float = rot_canvas_h / 2.0 + rot_rel_y

    # ------------------------------------------------------------------
    # 5.  Anchor: paste so final_mid lands on target_eye_midpoint
    # ------------------------------------------------------------------
    tgt_x, tgt_y = target_eye_midpoint
    paste_x: int = int(round(tgt_x - final_mid_x))
    paste_y: int = int(round(tgt_y - final_mid_y))

    final_scene = scene.copy()
    final_scene.paste(rotated_sketch, (paste_x, paste_y), rotated_sketch)
    return final_scene.convert("RGB")


def _run_scene_composition(
    sketch_rgba: Image.Image,
    face_mask_crop: Optional[np.ndarray],
    gender: str,
    sketch_kps: Optional[np.ndarray],
) -> List[Image.Image]:
    """
    Compose the face sketch into all FRU scene backgrounds.

    Reads ``target_eye_midpoint``, ``target_eye_distance``, and
    ``target_tilt_angle`` from each scene's entry in crops-face-meta.json.

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

    if sketch_kps is None:
        print("[WARN] No sketch keypoints available — scene composition skipped.")
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
            print(f"     Skipped scene {scene_id} (gender mismatch)")
            return None

        # -- Read landmark-based composition parameters from scene JSON -------
        try:
            face_anchor         = scene_meta["face_anchor"]           # sub-dict
            raw_mid             = face_anchor["target_eye_midpoint"]  # [x, y]
            target_eye_midpoint: Tuple[float, float] = (
                float(raw_mid[0]), float(raw_mid[1])
            )
            target_eye_distance: float = float(face_anchor["target_eye_distance"])
            target_tilt_angle:   float = float(face_anchor["target_tilt_angle"])
        except KeyError as exc:
            print(
                f"     Skipped scene {scene_id} — missing required key "
                f"in crops-face-meta.json: {exc}"
            )
            return None

        try:
            return _compose_single_scene(
                sketch_rgba=sketch_rgba,
                face_mask_crop=face_mask_crop,
                scene_path=scene_file,
                target_eye_midpoint=target_eye_midpoint,
                target_eye_distance=target_eye_distance,
                target_tilt_angle=target_tilt_angle,
                sketch_kps=sketch_kps,
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

    LoRA strategy:
      PEFT available  → named adapters ('lcm', 'style') + set_adapters().
                        Avoids the fuse_lora() conflict that breaks a second load.
      PEFT absent     → legacy fuse_lora() path for older diffusers installs.
    """
    config_key = f"fru_{device}_{dtype}_{load_lora}_{lora_scale}_{use_lcm}"
    if (
        _MODEL_CACHE["pipeline"] is not None
        and _MODEL_CACHE["pipeline_config"] == config_key
    ):
        logger.info("Using cached FRU pipeline")
        return _MODEL_CACHE["pipeline"], _MODEL_CACHE["lora_loaded"]

    _verify_local_models()

    # -- ControlNet ------------------------------------------------------------
    controlnet = ControlNetModel.from_pretrained(
        MODEL_PATHS["controlnet"],
        torch_dtype=dtype,
        local_files_only=True,
    )

    # -- Base SD1.5 pipeline ---------------------------------------------------
    if device == "mps":
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_PATHS["sd15"],
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_PATHS["sd15"],
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
            local_files_only=True,
        )

    # -- TAESD (tiny VAE for fast LCM decoding) --------------------------------
    if use_lcm:
        try:
            pipe.vae = AutoencoderTiny.from_pretrained(
                MODEL_PATHS["taesd"],
                torch_dtype=dtype,
                local_files_only=True,
            )
            logger.info("TAESD loaded")
        except (FileNotFoundError, OSError, RuntimeError):
            logger.warning("TAESD loading failed — falling back to default VAE", exc_info=True)

    # -- LoRA loading: PEFT adapter method when available, legacy fuse_lora fallback --
    lora_loaded = False
    if use_lcm:
        lcm_path = MODEL_PATHS["lcm_lora"]

        if PEFT_AVAILABLE:
            # Modern path: named adapters + set_adapters() (no fuse_lora).
            # Loading a second LoRA after fuse_lora() corrupts attention weights;
            # the PEFT adapter backend avoids this by keeping each LoRA separate.
            pipe.load_lora_weights(
                lcm_path,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="lcm",
            )
            logger.info("LCM LoRA loaded as PEFT adapter 'lcm'")

            active_adapters: List[str] = ["lcm"]
            adapter_weights: List[float] = [1.0]

            if load_lora and Path(MODEL_PATHS["lora"]).exists():
                pipe.load_lora_weights(
                    MODEL_PATHS["lora"],
                    weight_name="Pencil_Sketch_by_vizsumit.safetensors",
                    adapter_name="style",
                )
                active_adapters.append("style")
                adapter_weights.append(lora_scale * 1.3)
                lora_loaded = True
                logger.info("Style LoRA loaded as PEFT adapter 'style'")

            pipe.set_adapters(active_adapters, adapter_weights=adapter_weights)
            logger.info(
                "Active adapters: %s  weights: %s", active_adapters, adapter_weights
            )

        else:
            # Legacy path: fuse_lora() for diffusers installations without PEFT.
            logger.warning(
                "peft package not found — using legacy fuse_lora() path. "
                "Install peft to enable the modern adapter backend."
            )
            pipe.load_lora_weights(
                lcm_path, weight_name="pytorch_lora_weights.safetensors"
            )
            pipe.fuse_lora(lora_scale=1.0)
            logger.info("LCM LoRA loaded and fused (legacy)")

            if load_lora and Path(MODEL_PATHS["lora"]).exists():
                pipe.load_lora_weights(
                    MODEL_PATHS["lora"],
                    weight_name="Pencil_Sketch_by_vizsumit.safetensors",
                )
                pipe.fuse_lora(lora_scale=lora_scale * 1.3)
                lora_loaded = True
                logger.info("Style LoRA loaded and fused (legacy)")

        pipe.scheduler = LCMScheduler.from_config(
            pipe.scheduler.config,
            local_files_only=True,
        )
        logger.info("LCM Scheduler activated")

    pipe = pipe.to(device)

    if device == "mps":
        pipe.enable_attention_slicing()
        logger.info("MPS attention slicing enabled")

    _MODEL_CACHE["pipeline"]        = pipe
    _MODEL_CACHE["pipeline_config"] = config_key
    _MODEL_CACHE["lora_loaded"]     = lora_loaded

    return pipe, lora_loaded


def _load_ip_adapter(pipe, device: str, dtype: torch.dtype):
    if not FACEID_AVAILABLE:
        return None
    try:
        return IPAdapterFaceID(
            pipe, MODEL_PATHS["ip_adapter"], device, num_tokens=4, torch_dtype=dtype
        )
    except (FileNotFoundError, OSError, RuntimeError):
        logger.warning("IP-Adapter loading failed", exc_info=True)
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
    is_lcm   = isinstance(pipe.scheduler, LCMScheduler)
    prompt   = _build_prompt(gender, use_lora, is_lcm)
    negative = _build_negative(gender, is_lcm)
    logger.info("Sketch generation — gender=%s | seed=%d | %s...", gender, seed, prompt[:80])

    generator = torch.Generator(
        device="cpu" if device == "mps" else device
    ).manual_seed(seed)

    # Inference hyper-parameters — read from config if present, fall back to
    # the production-tested defaults used at launch time.
    if is_lcm:
        num_steps  = GENERATION_DEFAULTS.get("lcm_steps",      4)
        guidance   = GENERATION_DEFAULTS.get("lcm_guidance",   2.5)
        ctrl_scale = GENERATION_DEFAULTS.get("lcm_ctrl_scale", 0.95)
    else:
        num_steps  = GENERATION_DEFAULTS.get("steps",      20)
        guidance   = GENERATION_DEFAULTS.get("guidance",   7.0)
        ctrl_scale = GENERATION_DEFAULTS.get("ctrl_scale", 0.9)

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
        except RuntimeError:
            logger.warning(
                "IP-Adapter generation failed, falling back to base pipe", exc_info=True
            )

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
# FRUGeneratedResult - output container
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
    use_lora: bool         = GENERATION_DEFAULTS["use_lora"],
    lora_scale: float      = GENERATION_DEFAULTS["lora_scale"],
    faceid_strength: float = GENERATION_DEFAULTS["faceid_strength"],
    device: Optional[str]  = None,
    use_lcm: bool          = GENERATION_DEFAULTS["use_lcm"],
    seed: Optional[int]    = None,
) -> FRUGeneratedResult:
    """
    Generate face sketch from FRUPreprocessedData entirely in RAM.

    1. Generates face sketch via SD1.5 + ControlNet + IP-Adapter FaceID
    2. Applies BiSeNet mask to sketch (or rembg fallback) -> RGBA cutout
    3. Re-detects face position in generated sketch via BiSeNet
    4. Composes into all FRU scene backgrounds using landmark-based alignment
    5. Returns FRUGeneratedResult whose .scene_images is sent back to Triton

    Args:
        data:             FRUPreprocessedData from fru_preprocessor.
        use_lora:         Whether to load the style LoRA.
        lora_scale:       Style LoRA weight multiplier.
        faceid_strength:  IP-Adapter FaceID conditioning scale.
        device:           Force a specific device string; auto-detected if None.
        use_lcm:          Use LCM scheduler + LoRA for 4-step inference.
        seed:             RNG seed for the diffusion generator.  Defaults to a
                          random integer so each call produces a distinct result.
    """
    if data.face_img is None or data.face_edges is None:
        raise RuntimeError(
            "FRUPreprocessedData has no face crop — cannot generate face sketch."
        )

    # Resolve seed: use caller-supplied value, or draw a fresh random one.
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logger.info("No seed provided — using random seed %d", seed)

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

    logger.info("Generating face sketch (gender=%s, seed=%d)...", data.gender, seed)
    face_sketch = _generate_single_sketch(
        pipe, ip_adapter, data.face_edges,
        faceid_embeds, faceid_strength,
        device, dtype,
        seed=seed,
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

    # Apply BiSeNet mask -> RGBA cutout
    final_sketch_rgba = _apply_face_mask(final_sketch_rgb, data.face_mask_crop)

    # Re-detect (or fall back to) facial landmarks in the generated sketch
    logger.info("Re-detecting landmark keypoints in generated sketch...")
    sketch_kps: Optional[np.ndarray] = _redetect_landmarks_in_sketch(
        final_sketch_rgb,
        original_landmarks=data.original_landmarks,
        face_crop_box=data.face_crop_box,
    )

    # Scene composition
    scene_images: List[Image.Image] = _run_scene_composition(
        final_sketch_rgba,
        face_mask_crop=data.face_mask_crop,
        gender=data.gender,
        sketch_kps=sketch_kps,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FRUGeneratedResult(
        final_sketch=final_sketch_rgb,
        scene_images=scene_images,
    )