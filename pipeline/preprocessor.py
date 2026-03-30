"""
Portrait Sketch Preprocessing Pipeline - Memory-Only
All processing runs in RAM. No files written to disk.
"""

import logging
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -- TF32 optimisation (no xformers) ------------------------------------------
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -- Optional heavy dependencies -----------------------------------------------
try:
    from controlnet_aux import LineartDetector, HEDdetector
except Exception:                           # ImportError or missing native libs
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

# -- Config import - works whether run from project root or as pipeline package -
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_PATHS, PREPROCESS_DEFAULTS

# -- Module-level model cache --------------------------------------------------
_MODEL_CACHE: Dict = {
    "insightface_app": None,
    "edge_detector":   None,
    "mtcnn":           None,
    "bisenet":         None,
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
        logger.info("Initializing RemBG session...")
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
    corrected = np.power(arr, 1.0 / gamma)
    # Protect highlights: pixels already near-white (> 0.80) get much less
    # gamma lift to prevent light skin/fabric from blowing out to pure white.
    # blend=0 at arr=0.80 (full correction), blend=1 at arr=1.0 (no correction).
    blend = np.clip((arr - 0.80) / 0.20, 0, 1)
    corrected = corrected * (1 - blend) + arr * blend
    return Image.fromarray((corrected * 255).astype(np.uint8))


def _adaptive_brighten(img: Image.Image, target_brightness: float = 0.55) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    # Measure brightness only on non-white pixels (person region).
    # White background pixels (all channels > 0.92) would inflate the mean
    # and cause light skin tones to be blown out.
    if len(arr.shape) == 3:
        non_white_mask = np.any(arr < 0.92, axis=-1)
    else:
        non_white_mask = arr < 0.92
    if non_white_mask.sum() > 0:
        current = float(np.mean(arr[non_white_mask]))
    else:
        current = float(np.mean(arr))
    if current < target_brightness:
        factor = min(target_brightness / (current + 0.01), 1.5)
        # Apply brightening only to non-white pixels to avoid blowing out
        # already-light areas (pale skin, light clothing against white BG).
        if len(arr.shape) == 3:
            arr[non_white_mask] = np.clip(arr[non_white_mask] * factor, 0, 1)
        else:
            arr[non_white_mask] = np.clip(arr[non_white_mask] * factor, 0, 1)
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
    except cv2.error as e:
        logger.warning("Local contrast enhancement failed", exc_info=True)
        return img


def _preprocess_for_sketch(img: Image.Image, gamma: float = 1.3,
                            enhance_contrast: bool = True) -> Image.Image:
    processed = _apply_gamma_correction(img, gamma=gamma)
    processed = _adaptive_brighten(processed, target_brightness=0.55)
    if enhance_contrast:
        processed = _enhance_local_contrast(processed)
    # Final highlight clamp - hard cap at 245/255 so no area ever reaches
    # pure white from cumulative processing. Pure white = invisible in lineart.
    arr = np.array(processed)
    arr = np.clip(arr, 0, 245)
    return Image.fromarray(arr.astype(np.uint8))


def _preprocess_for_edges(img: Image.Image) -> Image.Image:
    """
    Edge-map stream preprocessing — NO brightening.

    Skips gamma correction and adaptive brightening entirely so that
    highlights are never blown out to pure white before the Lineart
    detector runs.  Instead applies an aggressive CLAHE pass in
    grayscale to maximise local edge contrast without clipping.

    Args:
        img: RGB PIL Image (white-background composite).

    Returns:
        RGB PIL Image ready to be fed into _make_edges_enhanced.
    """
    try:
        img_cv = np.array(img)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        rgb_eq = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_eq)
    except cv2.error:
        logger.warning("Edge preprocessing CLAHE failed, returning original", exc_info=True)
        return img


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
    except RuntimeError:
        logger.error("Face detection failed", exc_info=True)
        return []


# ============================================================================
# GENDER DETECTION  (fallback only - JSON override is always preferred)
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
    except Exception:
        logger.warning("Gender detection failed", exc_info=True)
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
    except Exception:
        logger.error("Embedding extraction failed", exc_info=True)
        return None


# ============================================================================
# FACE & EDGE PROCESSING
# ============================================================================

def _enhance_face_region(img: Image.Image, face: Dict,
                         padding: float = 0.2,
                         sharpen_strength: float = 1.5) -> Image.Image:
    """Face enhancement with highlight suppression to reduce sweaty/rough skin texture."""
    x1, y1, x2, y2 = face["box"]
    w, h = x2 - x1, y2 - y1
    px, py = int(w * padding), int(h * padding)
    rx1 = max(0, x1 - px)
    ry1 = max(0, y1 - py)
    rx2 = min(img.width, x2 + px)
    ry2 = min(img.height, y2 + py)
    face_region = img.crop((rx1, ry1, rx2, ry2))

    # Suppress highlights before sharpening - converts blown-out bright skin
    # pixels (sweat, oily skin) to mid-tones so lineart doesn't pick up texture.
    arr = np.array(face_region).astype(np.float32) / 255.0
    # Soft rolloff for pixels above 0.78 brightness - pulls them toward 0.82
    highlight_mask = arr > 0.78
    arr[highlight_mask] = 0.78 + (arr[highlight_mask] - 0.78) * 0.45
    face_region = Image.fromarray((arr * 255).astype(np.uint8))

    # Mild contrast boost (reduced from 1.15 to avoid amplifying skin texture)
    face_region = ImageEnhance.Contrast(face_region).enhance(1.08)

    # Softer sharpening - reduced radius and percent to avoid exaggerating pores
    sharp = face_region.filter(
        ImageFilter.UnsharpMask(radius=0.8, percent=int(90 * sharpen_strength))
    )
    img.paste(sharp, (rx1, ry1))
    return img


def _make_edges_enhanced(img: Image.Image, preserve_details: bool = True) -> Image.Image:
    """Generate edge map. Tries Lineart -> HED -> Canny fallback."""
    if LineartDetector:
        try:
            if _MODEL_CACHE["edge_detector"] is None:
                _MODEL_CACHE["edge_detector"] = LineartDetector.from_pretrained(
                    MODEL_PATHS["annotators_lineart"],
                )
            det = _MODEL_CACHE["edge_detector"]
            result = det(img, coarse=not preserve_details)
            if result is not None:
                result = ImageEnhance.Contrast(result).enhance(1.2)
                return result.convert("RGB")
        except (FileNotFoundError, RuntimeError):
            logger.warning(
                "LineartDetector failed, trying HED fallback", exc_info=True
            )

    if HEDdetector:
        try:
            det = HEDdetector.from_pretrained(
                MODEL_PATHS["annotators_hed"],
            )
            result = det(img)
            if result is not None:
                return result.convert("RGB")
        except (FileNotFoundError, RuntimeError):
            logger.warning(
                "HEDdetector failed, falling back to Canny", exc_info=True
            )

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
# BISENET FACE PARSING — self-contained architecture + inference
#
# Exact port of zllrunning/face-parsing.PyTorch (model.py + resnet.py).
# State-dict keys match the official checkpoint verbatim so load_state_dict
# works with strict=True and zero missing/unexpected keys.
# No external repo or pip package is required.
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ResNet-18 backbone  (mirrors resnet.py from the official repo exactly)
# Key paths in checkpoint: cp.resnet.conv1, cp.resnet.bn1,
#   cp.resnet.layer1-4, cp.resnet.layer{N}.{i}.conv1/bn1/conv2/bn2/downsample
# ---------------------------------------------------------------------------

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class _BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1   = nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2   = nn.BatchNorm2d(out_chan)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        shortcut = self.downsample(x) if self.downsample is not None else x
        return self.relu(shortcut + residual)


def _make_layer(in_chan: int, out_chan: int, bnum: int,
                stride: int = 1) -> nn.Sequential:
    layers = [_BasicBlock(in_chan, out_chan, stride=stride)]
    for _ in range(bnum - 1):
        layers.append(_BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class _Resnet18(nn.Module):
    """
    Custom ResNet-18 matching resnet.py exactly.
    Returns (feat8, feat16, feat32) — 1/8, 1/16, 1/32 of input resolution.
    State-dict keys: conv1, bn1, layer1, layer2, layer3, layer4  (no 'fc').
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = _make_layer(64,  64,  bnum=2, stride=1)
        self.layer2  = _make_layer(64,  128, bnum=2, stride=2)   # → feat8
        self.layer3  = _make_layer(128, 256, bnum=2, stride=2)   # → feat16
        self.layer4  = _make_layer(256, 512, bnum=2, stride=2)   # → feat32

    def forward(self, x: torch.Tensor):
        x       = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x       = self.layer1(x)
        feat8   = self.layer2(x)
        feat16  = self.layer3(feat8)
        feat32  = self.layer4(feat16)
        return feat8, feat16, feat32


# ---------------------------------------------------------------------------
# BiSeNet building blocks  (mirrors model.py from the official repo exactly)
# ---------------------------------------------------------------------------

class _ConvBNReLU(nn.Module):
    def __init__(self, in_chan: int, out_chan: int,
                 ks: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class _BiSeNetOutput(nn.Module):
    """Segmentation head — keys: conv.conv/bn, conv_out."""
    def __init__(self, in_chan: int, mid_chan: int, n_classes: int) -> None:
        super().__init__()
        self.conv     = _ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv(x))


class _AttentionRefinementModule(nn.Module):
    """ARM — keys: conv.conv/bn, conv_atten, bn_atten."""
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.conv        = _ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten  = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten    = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.conv(x)
        atten = self.sigmoid_atten(
            self.bn_atten(self.conv_atten(F.avg_pool2d(feat, feat.size()[2:])))
        )
        return torch.mul(feat, atten)


class _ContextPath(nn.Module):
    """
    Context Path — keys: resnet.*, arm16.*, arm32.*,
                         conv_head32.*, conv_head16.*, conv_avg.*
    Returns (feat8, feat_cp8, feat_cp16)  — all at 1/8 resolution.
    """
    def __init__(self) -> None:
        super().__init__()
        self.resnet      = _Resnet18()
        self.arm16       = _AttentionRefinementModule(256, 128)
        self.arm32       = _AttentionRefinementModule(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = _ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg    = _ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        feat8, feat16, feat32 = self.resnet(x)
        H8,  W8  = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg       = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        avg_up    = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up  = self.conv_head32(
            F.interpolate(feat32_sum, (H16, W16), mode='nearest'))

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up  = self.conv_head16(
            F.interpolate(feat16_sum, (H8, W8), mode='nearest'))

        return feat8, feat16_up, feat32_up   # x8, x8, x16


class _FeatureFusionModule(nn.Module):
    """
    FFM — keys: convblk.conv/bn, conv1, conv2.
    in_chan = 256 (feat_res8 128ch + feat_cp8 128ch), out_chan = 256.
    """
    def __init__(self, in_chan: int, out_chan: int) -> None:
        super().__init__()
        self.convblk = _ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1   = nn.Conv2d(out_chan, out_chan // 4,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2   = nn.Conv2d(out_chan // 4, out_chan,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: torch.Tensor, fcp: torch.Tensor) -> torch.Tensor:
        feat       = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten      = self.sigmoid(self.conv2(self.relu(
            self.conv1(F.avg_pool2d(feat, feat.size()[2:])))))
        return torch.mul(feat, atten) + feat


class _BiSeNet(nn.Module):
    """
    BiSeNet V1 — exact port of zllrunning/face-parsing.PyTorch model.py.

    State-dict top-level keys: cp.*, ffm.*, conv_out.*, conv_out16.*, conv_out32.*
    NOTE: there is NO 'sp' key — the spatial path was replaced by feat_res8
    from the ResNet backbone (see comment in official repo forward()).
    """
    def __init__(self, n_classes: int = 19) -> None:
        super().__init__()
        self.cp        = _ContextPath()
        # sp is intentionally absent — feat_res8 acts as the spatial path
        self.ffm       = _FeatureFusionModule(256, 256)
        self.conv_out  = _BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = _BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = _BiSeNetOutput(128, 64, n_classes)

    def forward(self, x: torch.Tensor):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp   = feat_res8                          # reuse resnet feat8 as SP
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out   = F.interpolate(self.conv_out(feat_fuse),
                                   (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(self.conv_out16(feat_cp8),
                                   (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(self.conv_out32(feat_cp16),
                                   (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32


# -- Class IDs to keep in the binary blend mask --------------------------------
# Facial features: 1-13, neck: 14, hair: 17.
# Strictly excluded: 0 (background), 16 (clothing).
_BISENET_FACE_IDS = frozenset([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17])

_BISENET_MEAN = [0.485, 0.456, 0.406]
_BISENET_STD  = [0.229, 0.224, 0.225]


def _get_bisenet_mask(face_img: Image.Image) -> Optional[Image.Image]:
    """
    Run BiSeNet face-parsing on *face_img* and return a binary PIL mask (mode "L").

    Pixels belonging to facial features (class IDs 1-13), neck (14), or hair
    (17) are set to 255; all other pixels (including background=0 and
    clothing=16) are set to 0.

    Args:
        face_img: RGB PIL Image of the face crop (any size).

    Returns:
        PIL Image (mode "L") at the same size as *face_img*, or None on error.
    """
    try:
        if _MODEL_CACHE["bisenet"] is None:
            logger.info("Loading BiSeNet model from %s", MODEL_PATHS["bisenet"])

            raw = torch.load(
                MODEL_PATHS["bisenet"],
                map_location=_get_torch_device(),
                weights_only=False,
            )

            if isinstance(raw, torch.nn.Module):
                # Checkpoint was saved with torch.save(model, path) — use directly.
                model = raw
                logger.info("BiSeNet loaded as serialised nn.Module")
            else:
                # Checkpoint is a plain state dict (the standard case for
                # zllrunning/face-parsing.PyTorch checkpoints saved via
                # torch.save(model.state_dict(), path)).
                # Instantiate the architecture that lives in this file and
                # load the weights — no external repo or pip package needed.
                model = _BiSeNet(n_classes=19)
                # Strip DataParallel 'module.' prefix when present.
                state = {k.replace("module.", ""): v for k, v in raw.items()}
                missing, unexpected = model.load_state_dict(state, strict=True)
                logger.info("BiSeNet state dict loaded — all keys matched")

            model.to(_get_torch_device())
            model.eval()
            _MODEL_CACHE["bisenet"] = model
            logger.info("BiSeNet model ready and cached on %s", _get_torch_device())
        model = _MODEL_CACHE["bisenet"]

    except Exception:
        logger.error("Failed to load BiSeNet model", exc_info=True)
        return None

    try:
        original_size = face_img.size          # (W, H) — needed for resize-back
        device = _get_torch_device()

        # -- Preprocess: resize → float tensor → normalise → add batch dim ----
        resized = face_img.convert("RGB").resize((512, 512), Image.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0            # H×W×3

        mean = np.array(_BISENET_MEAN, dtype=np.float32)
        std  = np.array(_BISENET_STD,  dtype=np.float32)
        arr  = (arr - mean) / std

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1×3×512×512
        tensor = tensor.to(device)

        # -- Inference: out_main is the primary logit map (index 0) -----------
        with torch.no_grad():
            out_main = model(tensor)[0]                                # 1×19×512×512

        class_ids = out_main.squeeze(0).argmax(dim=0).cpu().numpy()   # 512×512 int64

        # -- Build binary mask: 255 for face/hair/neck, 0 everywhere else -----
        mask_arr = np.zeros(class_ids.shape, dtype=np.uint8)
        for cls_id in _BISENET_FACE_IDS:
            mask_arr[class_ids == cls_id] = 255

        # -- Resize mask back to original face crop dimensions ----------------
        mask_img = Image.fromarray(mask_arr, mode="L")
        mask_img = mask_img.resize(original_size, Image.NEAREST)

        logger.info(
            "BiSeNet mask generated: size=%s, face coverage=%.1f%%",
            mask_img.size,
            float(np.mean(mask_arr > 0)) * 100,
        )
        return mask_img

    except Exception:
        logger.error("BiSeNet inference failed", exc_info=True)
        return None


# ============================================================================
# SEGMENTATION
# ============================================================================

def segment_and_crop(img: Image.Image) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    Background removal, content crop, and white-BG composite.

    Alpha-aware: if the input is already RGBA and more than 5 % of pixels have
    alpha < 250, the rembg pass is skipped entirely to prevent edge erosion on
    subjects like fingers or shoes that touch the frame boundary.

    Returns:
        cropped_rgba  - subject on transparent background (for gender/ID detection)
        img_white_bg  - subject on white background (for sketch/edges)
        crop_info     - crop metadata dict
    """
    if not REMBG_AVAILABLE:
        raise ImportError("rembg not installed: pip install rembg")

    _init_rembg()

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")

    # -- Alpha-aware rembg bypass -------------------------------------------------
    skip_rembg = False
    if img.mode == "RGBA":
        alpha_arr = np.array(img.getchannel("A"))
        transparent_ratio = float(np.mean(alpha_arr < 250))
        if transparent_ratio > 0.05:
            skip_rembg = True
            logger.info(
                "Skipping rembg: %.1f%% of pixels already transparent",
                transparent_ratio * 100,
            )

    if skip_rembg:
        # Image already has a clean alpha channel — use it directly.
        cropped_rgba, crop_info = _crop_to_content(img, padding_pct=0.05)
    else:
        logger.info("Running rembg background removal")
        segmented = remove(img, session=_REMBG_SESSION)       # always returns RGBA
        cropped_rgba, crop_info = _crop_to_content(segmented, padding_pct=0.05)

    white_bg = Image.new("RGB", cropped_rgba.size, (255, 255, 255))
    white_bg.paste(cropped_rgba, mask=cropped_rgba.split()[3])

    return cropped_rgba, white_bg, crop_info


# ============================================================================
# PreprocessedData - in-memory result container
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
        semantic_face_mask: Optional[Image.Image] = None,
    ):
        self.enhanced           = enhanced
        self.body_edges         = body_edges
        self.face_img           = face_img
        self.face_edges         = face_edges
        self.faceid_embedding   = faceid_embedding
        self.gender             = gender
        self.primary_face       = primary_face
        self.face_crop_box      = face_crop_box
        self.original_size      = original_size
        self.crop_info          = crop_info
        self.semantic_face_mask = semantic_face_mask

    @property
    def has_face_embedding(self) -> bool:
        return self.faceid_embedding is not None

    def save_intermediates(self, out_dir: str) -> None:
        """
        Save every visual artifact to *out_dir* for debugging / inspection.

        Files written (only when the artifact is not None):
            enhanced.jpg            — SD-stream full-body image
            body_edges.jpg          — full-body lineart / Canny edge map
            face_img.jpg            — SD-stream face crop
            face_edges.jpg          — face edge map
            semantic_face_mask.png  — BiSeNet binary mask (L-mode, lossless PNG)

        The directory is created automatically if it does not exist.
        """
        import os
        os.makedirs(out_dir, exist_ok=True)

        def _save(img: Optional[Image.Image], filename: str) -> None:
            if img is None:
                return
            path = os.path.join(out_dir, filename)
            # Force RGB for JPEG; keep PNG for mask (lossless, preserves L mode)
            if filename.endswith(".jpg"):
                img.convert("RGB").save(path, quality=95)
            else:
                img.save(path)
            logger.info("Saved intermediate: %s", path)

        _save(self.enhanced,           "enhanced.jpg")
        _save(self.body_edges,         "body_edges.jpg")
        _save(self.face_img,           "face_img.jpg")
        _save(self.face_edges,         "face_edges.jpg")
        _save(self.semantic_face_mask, "semantic_face_mask.png")


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

    # 2. Split into two processing streams from the same white-BG source:
    #
    #    img_for_sd   — gamma + adaptive brightening + local contrast.
    #                   Used for face detection, face enhancement, and the
    #                   `enhanced` image fed to Stable Diffusion.
    #
    #    img_for_edges — grayscale + aggressive CLAHE only (NO brightening).
    #                   Used exclusively to derive body and face edge maps so
    #                   that highlights are never blown out before the Lineart
    #                   detector runs, preventing missing limbs / outlines.
    img_for_sd = _preprocess_for_sketch(img_white_bg, gamma=gamma, enhance_contrast=True)
    img_for_edges = _preprocess_for_edges(img_white_bg)
    logger.info("Dual stream split: img_for_sd and img_for_edges created")

    # 3. Face detection (run on the SD stream — brightness helps MTCNN)
    all_faces = _detect_all_faces(img_for_sd)
    primary_face = all_faces[0] if all_faces else None

    # 4. Gender resolution
    #    Priority: JSON override -> InsightFace detection -> 'unknown'
    override = gender_override.strip().lower() if gender_override else None
    if override in ("male", "female"):
        gender = override
        logger.info("Gender from JSON override: %s", gender)
    elif primary_face:
        gender = _detect_gender(cropped_rgba, primary_face["box"])
        logger.info("Gender from detection: %s", gender)
    else:
        gender = "unknown"
        logger.warning(
            "Gender unknown — no face detected and no JSON override provided"
        )

    # 5. Face enhancement - uniform sharpening, no gender/beard variants
    #    Applied to the SD stream; result becomes the `enhanced` output.
    enhanced = img_for_sd.copy()
    if primary_face and enhance_faces:
        enhanced = _enhance_face_region(enhanced, primary_face)

    # 6. Face embedding (run on RGBA for best identity fidelity)
    faceid_embedding = None
    if primary_face and INSIGHTFACE_AVAILABLE:
        faceid_embedding = _extract_face_embedding(cropped_rgba, primary_face["box"])
        if faceid_embedding is not None:
            logger.info("Face embedding extracted: shape %s", faceid_embedding.shape)

    # 7. Parallel edge map generation (body + face)
    #    Both threads consume img_for_edges (the unbrightened CLAHE stream).
    def process_body_edges():
        b_edges = _make_edges_enhanced(img_for_edges, preserve_details=preserve_details)
        if max(b_edges.size) > 1024:
            s = 1024 / max(b_edges.size)
            b_edges = b_edges.resize(
                (int(b_edges.width * s), int(b_edges.height * s)), Image.LANCZOS
            )
        return b_edges

    def process_face_data():
        if not primary_face:
            return None, None, None, None
        # Crop the face bounding box from img_for_edges (not from enhanced /
        # img_for_sd) so the lineart detector sees unbrightened edge contrast.
        f_img_edges, f_box = _crop_face_with_padding(img_for_edges, primary_face["box"])
        # Return the SD-stream face crop as f_img (used by SD for face reference)
        # but generate edges from the edge stream crop.
        f_img_sd, _ = _crop_face_with_padding(enhanced, primary_face["box"])
        if min(f_img_edges.size) < 512:
            s = 512 / min(f_img_edges.size)
            f_img_edges = f_img_edges.resize(
                (int(f_img_edges.width * s), int(f_img_edges.height * s)), Image.LANCZOS
            )
        if min(f_img_sd.size) < 512:
            s = 512 / min(f_img_sd.size)
            f_img_sd = f_img_sd.resize(
                (int(f_img_sd.width * s), int(f_img_sd.height * s)), Image.LANCZOS
            )
        f_edges = _make_edges_enhanced(f_img_edges, preserve_details=preserve_details)
        f_semantic_mask = _get_bisenet_mask(f_img_sd)
        return f_img_sd, f_edges, f_box, f_semantic_mask

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_body = executor.submit(process_body_edges)
        future_face = executor.submit(process_face_data)
        body_edges = future_body.result()
        face_img, face_edges, face_crop_box, semantic_face_mask = future_face.result()

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
        semantic_face_mask=semantic_face_mask,
    )