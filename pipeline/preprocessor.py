"""
Portrait Sketch Preprocessing Pipeline — Memory-Only
All processing runs in RAM. No files written to disk.
"""

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

# ── TF32 optimisation (no xformers) ──────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ── Optional heavy dependencies ───────────────────────────────────────────────
try:
    from controlnet_aux import LineartDetector, HEDdetector
except Exception:
    LineartDetector = None
    HEDdetector = None

try:
    from facenet_pytorch import MTCNN
    FACE_DETECTOR_AVAILABLE = True
except Exception:
    FACE_DETECTOR_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    import onnxruntime
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False
    onnxruntime = None

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except Exception:
    REMBG_AVAILABLE = False

# ── Config import — works whether run from project root or as pipeline package ─
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_PATHS, PREPROCESS_DEFAULTS

# ── Module-level model cache ──────────────────────────────────────────────────
_MODEL_CACHE: Dict = {
    "insightface_app": None,
    "edge_detector":   None,
    "mtcnn":           None,
}
_REMBG_SESSION = None


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def _get_onnx_providers() -> List[str]:
    if (onnxruntime and torch.cuda.is_available()
            and "CUDAExecutionProvider" in onnxruntime.get_available_providers()):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _get_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================================================
# REMBG
# ============================================================================

def _init_rembg():
    global _REMBG_SESSION
    if _REMBG_SESSION is None and REMBG_AVAILABLE:
        print("[INFO] Initializing RemBG session...")
        try:
            _REMBG_SESSION = new_session("u2net")
        except Exception:
            _REMBG_SESSION = new_session("u2netp")


# ============================================================================
# IMAGE CORRECTION UTILITIES
# ============================================================================

def _crop_to_content(img: Image.Image, padding_pct: float = 0.05) -> Tuple[Image.Image, Dict]:
    if img.mode != "RGBA":
        return img, {"crop_box": None, "original_size": img.size}
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return img, {"crop_box": None, "original_size": img.size}
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * padding_pct), int(h * padding_pct)
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(img.width, x2 + pad_w)
    cy2 = min(img.height, y2 + pad_h)
    return img.crop((cx1, cy1, cx2, cy2)), {
        "original_size": img.size,
        "crop_box": [cx1, cy1, cx2, cy2],
        "content_bbox": [x1, y1, x2, y2],
    }


def _apply_gamma_correction(img: Image.Image, gamma: float = 1.3) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    corrected = (np.power(arr, 1.0 / gamma) * 255).astype(np.uint8)
    return Image.fromarray(corrected)


def _adaptive_brighten(img: Image.Image, target_brightness: float = 0.55) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    current = np.mean(arr)
    if current < target_brightness:
        factor = min(target_brightness / (current + 0.01), 1.8)
        arr = np.clip(arr * factor, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def _enhance_local_contrast(img: Image.Image) -> Image.Image:
    try:
        img_cv = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(img_cv.shape) == 3:
            lab = cv2.cvtColor(img_cv, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = clahe.apply(img_cv)
        return Image.fromarray(enhanced)
    except Exception as e:
        print(f"[WARN] Local contrast enhancement failed: {e}")
        return img


def _preprocess_for_sketch(img: Image.Image, gamma: float = 1.3,
                            enhance_contrast: bool = True) -> Image.Image:
    processed = _apply_gamma_correction(img, gamma=gamma)
    processed = _adaptive_brighten(processed, target_brightness=0.55)
    if enhance_contrast:
        processed = _enhance_local_contrast(processed)
    return processed


# ============================================================================
# FACE DETECTION
# ============================================================================

def _detect_all_faces(img: Image.Image) -> List[Dict]:
    if not FACE_DETECTOR_AVAILABLE:
        return []
    try:
        if _MODEL_CACHE["mtcnn"] is None:
            _MODEL_CACHE["mtcnn"] = MTCNN(
                keep_all=True,
                device=_get_torch_device(),
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
            )
        boxes, probs = _MODEL_CACHE["mtcnn"].detect(img)
        if boxes is None or len(boxes) == 0:
            return []
        faces = []
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            x1, y1, x2, y2 = tuple(map(int, box.tolist()))
            faces.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(prob),
                "size": (x2 - x1, y2 - y1),
                "index": i,
            })
        faces.sort(key=lambda x: x["confidence"], reverse=True)
        return faces
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        return []


# ============================================================================
# GENDER DETECTION  (fallback only — JSON override is always preferred)
# ============================================================================

def _detect_gender(img: Image.Image, face_box: Optional[Tuple] = None) -> str:
    """
    Auto-detects gender using InsightFace.
    Only called when no JSON gender override is provided.
    Returns 'male', 'female', or 'unknown'.
    """
    if not INSIGHTFACE_AVAILABLE:
        return "unknown"
    try:
        if _MODEL_CACHE["insightface_app"] is None:
            app = FaceAnalysis(
                name="antelopev2",
                root=MODEL_PATHS["insightface"],
                providers=_get_onnx_providers(),
            )
            ctx_id = 0 if torch.cuda.is_available() else -1
            app.prepare(ctx_id=ctx_id, det_size=(1280, 1280))
            _MODEL_CACHE["insightface_app"] = app
        else:
            app = _MODEL_CACHE["insightface_app"]

        img_np = np.array(img)
        img_bgr = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGBA2BGR if img.mode == "RGBA" else cv2.COLOR_RGB2BGR,
        )
        faces = app.get(img_bgr)
        if not faces:
            return "unknown"

        if face_box:
            x1, y1, x2, y2 = face_box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best_face = min(
                faces,
                key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2
                              + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2,
            )
        else:
            best_face = faces[0]

        return "male" if best_face.gender == 1 else "female"
    except Exception as e:
        print(f"[WARN] Gender detection failed: {e}")
        return "unknown"


# ============================================================================
# FACE EMBEDDING
# ============================================================================

def _extract_face_embedding(img: Image.Image, face_box: Tuple) -> Optional[np.ndarray]:
    if not INSIGHTFACE_AVAILABLE:
        return None
    try:
        # Reuse the same InsightFace app instance already loaded for gender detection
        if _MODEL_CACHE["insightface_app"] is None:
            app = FaceAnalysis(
                name="antelopev2",
                root=MODEL_PATHS["insightface"],
                providers=_get_onnx_providers(),
            )
            ctx_id = 0 if torch.cuda.is_available() else -1
            app.prepare(ctx_id=ctx_id, det_size=(1280, 1280))
            _MODEL_CACHE["insightface_app"] = app
        else:
            app = _MODEL_CACHE["insightface_app"]

        img_np = np.array(img)
        img_bgr = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGBA2BGR if img.mode == "RGBA" else cv2.COLOR_RGB2BGR,
        )
        faces = app.get(img_bgr)
        if not faces:
            return None

        x1, y1, x2, y2 = face_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best_face = min(
            faces,
            key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - cx) ** 2
                          + ((f.bbox[1] + f.bbox[3]) / 2 - cy) ** 2,
        )
        return best_face.normed_embedding
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        return None


# ============================================================================
# FACE & EDGE PROCESSING
# ============================================================================

def _enhance_face_region(img: Image.Image, face: Dict,
                         padding: float = 0.2,
                         sharpen_strength: float = 1.5) -> Image.Image:
    """Uniform face sharpening — no gender/beard variants."""
    x1, y1, x2, y2 = face["box"]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)
    rx1 = max(0, x1 - px)
    ry1 = max(0, y1 - py)
    rx2 = min(img.width, x2 + px)
    ry2 = min(img.height, y2 + py)
    face_region = img.crop((rx1, ry1, rx2, ry2))
    face_region = ImageEnhance.Contrast(face_region).enhance(1.15)
    sharp = face_region.filter(
        ImageFilter.UnsharpMask(radius=1.2, percent=int(150 * sharpen_strength))
    )
    img.paste(sharp, (rx1, ry1))
    return img


def _make_edges_enhanced(img: Image.Image, preserve_details: bool = True) -> Image.Image:
    """Generate edge map. Tries Lineart → HED → Canny fallback."""
    if LineartDetector:
        try:
            if _MODEL_CACHE["edge_detector"] is None:
                _MODEL_CACHE["edge_detector"] = LineartDetector.from_pretrained(
                    MODEL_PATHS["annotators_lineart"]
                )
            det = _MODEL_CACHE["edge_detector"]
            result = det(img, coarse=not preserve_details)
            if result is not None:
                result = ImageEnhance.Contrast(result).enhance(1.2)
                return result.convert("RGB")
        except Exception:
            pass

    if HEDdetector:
        try:
            det = HEDdetector.from_pretrained(MODEL_PATHS["annotators_hed"])
            result = det(img)
            if result is not None:
                return result.convert("RGB")
        except Exception:
            pass

    # Canny fallback
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def _crop_face_with_padding(img: Image.Image,
                            face_box: Tuple) -> Tuple[Image.Image, Tuple]:
    """Uniform face crop with consistent padding for all genders."""
    x1, y1, x2, y2 = face_box
    w, h = x2 - x1, y2 - y1
    base_padding = 0.4
    px       = int(w * (base_padding + 0.05))
    py_top   = int(h * (base_padding + 0.30))
    py_bottom = int(h * base_padding)
    shift    = int(h * 0.07)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py_top - shift)
    cx2 = min(img.width, x2 + px)
    cy2 = min(img.height, y2 + py_bottom - shift)
    return img.crop((cx1, cy1, cx2, cy2)), (cx1, cy1, cx2, cy2)


# ============================================================================
# SEGMENTATION
# ============================================================================

def segment_and_crop(img: Image.Image) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    Background removal, content crop, and white-BG composite.

    Returns:
        cropped_rgba  — subject on transparent background (for gender/ID detection)
        img_white_bg  — subject on white background (for sketch/edges)
        crop_info     — crop metadata dict
    """
    if not REMBG_AVAILABLE:
        raise ImportError("rembg not installed: pip install rembg")

    _init_rembg()

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    segmented = remove(img, session=_REMBG_SESSION)           # always returns RGBA
    cropped_rgba, crop_info = _crop_to_content(segmented, padding_pct=0.05)

    white_bg = Image.new("RGB", cropped_rgba.size, (255, 255, 255))
    white_bg.paste(cropped_rgba, mask=cropped_rgba.split()[3])

    return cropped_rgba, white_bg, crop_info


# ============================================================================
# PreprocessedData — in-memory result container
# ============================================================================

class PreprocessedData:
    """In-memory container holding all artifacts needed by the generation pipeline."""

    def __init__(
        self,
        enhanced: Image.Image,
        body_edges: Image.Image,
        face_img: Optional[Image.Image],
        face_edges: Optional[Image.Image],
        faceid_embedding: Optional[np.ndarray],
        gender: str,
        primary_face: Optional[Dict],
        face_crop_box: Optional[Tuple],
        original_size: Tuple[int, int],
        crop_info: Dict,
    ):
        self.enhanced         = enhanced
        self.body_edges       = body_edges
        self.face_img         = face_img
        self.face_edges       = face_edges
        self.faceid_embedding = faceid_embedding
        self.gender           = gender
        self.primary_face     = primary_face
        self.face_crop_box    = face_crop_box
        self.original_size    = original_size
        self.crop_info        = crop_info

    @property
    def has_face_embedding(self) -> bool:
        return self.faceid_embedding is not None


# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_image_in_memory(
    img: Image.Image,
    gender_override: Optional[str] = None,
    gamma: float = PREPROCESS_DEFAULTS["gamma"],
    preserve_details: bool = PREPROCESS_DEFAULTS["preserve_details"],
    enhance_faces: bool = PREPROCESS_DEFAULTS["enhance_faces"],
) -> PreprocessedData:
    """
    Full preprocessing pipeline operating entirely in RAM.

    Args:
        img:             Input PIL Image (any mode).
        gender_override: 'male' or 'female' sourced from the person's JSON file.
                         When provided, skips InsightFace detection entirely.
                         Falls back to detection if None, then to 'unknown'.
        gamma:           Gamma correction value.
        preserve_details: Use fine lineart detection (True) or coarse (False).
        enhance_faces:   Apply face region sharpening.

    Returns:
        PreprocessedData
    """

    # 1. Background removal + auto-crop
    cropped_rgba, img_white_bg, crop_info = segment_and_crop(img)
    current_size = img_white_bg.size

    # 2. Corrections on white-BG version (used for sketch/edges)
    img_corrected = _preprocess_for_sketch(img_white_bg, gamma=gamma, enhance_contrast=True)

    # 3. Face detection
    all_faces = _detect_all_faces(img_corrected)
    primary_face = all_faces[0] if all_faces else None

    # 4. Gender resolution
    #    Priority: JSON override → InsightFace detection → 'unknown'
    override = gender_override.strip().lower() if gender_override else None
    if override in ("male", "female"):
        gender = override
        print(f"  ✓ Gender from JSON override: {gender}")
    elif primary_face:
        gender = _detect_gender(cropped_rgba, primary_face["box"])
        print(f"  ✓ Gender from detection: {gender}")
    else:
        gender = "unknown"
        print("  ! Gender unknown — no face detected and no JSON override provided")

    # 5. Face enhancement — uniform sharpening, no gender/beard variants
    enhanced = img_corrected.copy()
    if primary_face and enhance_faces:
        enhanced = _enhance_face_region(enhanced, primary_face)

    # 6. Face embedding (run on RGBA for best identity fidelity)
    faceid_embedding = None
    if primary_face and INSIGHTFACE_AVAILABLE:
        faceid_embedding = _extract_face_embedding(cropped_rgba, primary_face["box"])
        if faceid_embedding is not None:
            print(f"  ✓ Face embedding extracted: shape {faceid_embedding.shape}")

    # 7. Parallel edge map generation (body + face)
    def process_body_edges():
        b_edges = _make_edges_enhanced(enhanced, preserve_details=preserve_details)
        if max(b_edges.size) > 1024:
            s = 1024 / max(b_edges.size)
            b_edges = b_edges.resize(
                (int(b_edges.width * s), int(b_edges.height * s)), Image.LANCZOS
            )
        return b_edges

    def process_face_data():
        if not primary_face:
            return None, None, None
        f_img, f_box = _crop_face_with_padding(enhanced, primary_face["box"])
        if min(f_img.size) < 512:
            s = 512 / min(f_img.size)
            f_img = f_img.resize(
                (int(f_img.width * s), int(f_img.height * s)), Image.LANCZOS
            )
        f_edges = _make_edges_enhanced(f_img, preserve_details=preserve_details)
        return f_img, f_edges, f_box

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_body = executor.submit(process_body_edges)
        future_face = executor.submit(process_face_data)
        body_edges = future_body.result()
        face_img, face_edges, face_crop_box = future_face.result()

    return PreprocessedData(
        enhanced=enhanced,
        body_edges=body_edges,
        face_img=face_img,
        face_edges=face_edges,
        faceid_embedding=faceid_embedding,
        gender=gender,
        primary_face=primary_face,
        face_crop_box=face_crop_box,
        original_size=current_size,
        crop_info=crop_info,
    )
