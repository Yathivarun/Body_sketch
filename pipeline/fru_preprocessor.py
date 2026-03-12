"""
FRU Face Sketch Preprocessing Pipeline — Memory-Only
All processing runs in RAM. No files written to disk.
Returns FRUPreprocessedData for consumption by fru_generator.py
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple

# ── TF32 optimisation ─────────────────────────────────────────────────────────
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

# ── Config import ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_PATHS, PREPROCESS_DEFAULTS

# ── Module-level model cache ──────────────────────────────────────────────────
_MODEL_CACHE: Dict = {
    "insightface_app": None,
    "edge_detector":   None,
    "mtcnn":           None,
    "bisenet":         None,
}
_REMBG_SESSION = None

# ── BiSeNet label groups ──────────────────────────────────────────────────────
_BISENET_FACE_NO_NECK   = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]
_BISENET_FACE_WITH_HAIR = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17]
_BISENET_FACE_ONLY      = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]


# ============================================================================
# BISENET MODEL DEFINITION
# ============================================================================

def _conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class _BasicBlock(torch.nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2 = torch.nn.BatchNorm2d(out_chan)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = self.downsample(x) if self.downsample else x
        return self.relu(shortcut + residual)


def _make_layer(in_chan, out_chan, bnum, stride=1):
    layers = [_BasicBlock(in_chan, out_chan, stride=stride)]
    for _ in range(bnum - 1):
        layers.append(_BasicBlock(out_chan, out_chan, stride=1))
    return torch.nn.Sequential(*layers)


class _Resnet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _make_layer(64, 64, bnum=2, stride=1)
        self.layer2 = _make_layer(64, 128, bnum=2, stride=2)
        self.layer3 = _make_layer(128, 256, bnum=2, stride=2)
        self.layer4 = _make_layer(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class _ConvBNReLU(torch.nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class _BiSeNetOutput(torch.nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = _ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = torch.nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class _ARM(torch.nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = _ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = torch.nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = torch.nn.BatchNorm2d(out_chan)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid(self.bn_atten(self.conv_atten(atten)))
        return torch.mul(feat, atten)


class _ContextPath(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = _Resnet18()
        self.arm16 = _ARM(256, 128)
        self.arm32 = _ARM(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = _ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = _ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        avg = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        avg_up = F.interpolate(avg, feat32.size()[2:], mode="nearest")
        feat32_up = self.conv_head32(
            F.interpolate(self.arm32(feat32) + avg_up, feat16.size()[2:], mode="nearest")
        )
        feat16_up = self.conv_head16(
            F.interpolate(self.arm16(feat16) + feat32_up, feat8.size()[2:], mode="nearest")
        )
        return feat8, feat16_up, feat32_up


class _FFM(torch.nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = _ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = self.sigmoid(
            self.conv2(self.relu(self.conv1(F.avg_pool2d(feat, feat.size()[2:]))))
        )
        return torch.mul(feat, atten) + feat


class _BiSeNet(torch.nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = _ContextPath()
        self.ffm = _FFM(256, 256)
        self.conv_out = _BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = _BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = _BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        out = F.interpolate(
            self.conv_out(feat_fuse), (H, W), mode="bilinear", align_corners=True
        )
        out16 = F.interpolate(
            self.conv_out16(feat_cp8), (H, W), mode="bilinear", align_corners=True
        )
        out32 = F.interpolate(
            self.conv_out32(feat_cp16), (H, W), mode="bilinear", align_corners=True
        )
        return out, out16, out32


# ============================================================================
# DEVICE UTILITIES
# ============================================================================

def _get_onnx_providers() -> List[str]:
    if (
        onnxruntime
        and torch.cuda.is_available()
        and "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    ):
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
# BISENET LOADER
# ============================================================================

def _init_bisenet() -> Optional[_BiSeNet]:
    """Lazy-load BiSeNet into model cache. Returns None if model file missing."""
    if _MODEL_CACHE["bisenet"] is not None:
        return _MODEL_CACHE["bisenet"]

    model_path = MODEL_PATHS.get("bisenet", "")
    if not model_path or not __import__("pathlib").Path(model_path).exists():
        print(f"  [WARN] BiSeNet model not found at '{model_path}' — falling back to rembg cutout")
        return None
    try:
        device = _get_torch_device()
        net = _BiSeNet(n_classes=19)
        state = torch.load(model_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        net.load_state_dict(state, strict=True)
        net.to(device)
        net.eval()
        _MODEL_CACHE["bisenet"] = net
        print(f"  [INFO] BiSeNet loaded on {device}")
        return net
    except Exception as e:
        print(f"  [WARN] BiSeNet load failed: {e} — falling back to rembg cutout")
        return None


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
# GENDER DETECTION
# ============================================================================

def _detect_gender(img: Image.Image, face_box: Optional[Tuple] = None) -> str:
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
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            _MODEL_CACHE["insightface_app"] = app
        else:
            app = _MODEL_CACHE["insightface_app"]

        if face_box:
            x1, y1, x2, y2 = face_box
            w, h = x2 - x1, y2 - y1
            pad = int(max(w, h) * 0.3)
            cx1 = max(0, x1 - pad)
            cy1 = max(0, y1 - pad)
            cx2 = min(img.width, x2 + pad)
            cy2 = min(img.height, y2 + pad)
            img_crop = img.crop((cx1, cy1, cx2, cy2))
            remapped_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
        else:
            img_crop = img
            remapped_box = None

        img_np = np.array(img_crop)
        img_bgr = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGBA2BGR if img_crop.mode == "RGBA" else cv2.COLOR_RGB2BGR,
        )
        faces = app.get(img_bgr)
        if not faces:
            return "unknown"

        if remapped_box:
            rx1, ry1, rx2, ry2 = remapped_box
            cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
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
        if _MODEL_CACHE["insightface_app"] is None:
            app = FaceAnalysis(
                name="antelopev2",
                root=MODEL_PATHS["insightface"],
                providers=_get_onnx_providers(),
            )
            ctx_id = 0 if torch.cuda.is_available() else -1
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            _MODEL_CACHE["insightface_app"] = app
        else:
            app = _MODEL_CACHE["insightface_app"]

        x1, y1, x2, y2 = face_box
        w, h = x2 - x1, y2 - y1
        pad = int(max(w, h) * 0.3)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(img.width, x2 + pad)
        cy2 = min(img.height, y2 + pad)
        img_crop = img.crop((cx1, cy1, cx2, cy2))
        remapped_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)

        img_np = np.array(img_crop)
        img_bgr = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGBA2BGR if img_crop.mode == "RGBA" else cv2.COLOR_RGB2BGR,
        )
        faces = app.get(img_bgr)
        if not faces:
            return None

        rx1, ry1, rx2, ry2 = remapped_box
        cx, cy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
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
# FACE REGION ENHANCEMENT
# ============================================================================

def _enhance_face_region(img: Image.Image, face: Dict,
                          padding: float = 0.2,
                          sharpen_strength: float = 1.5) -> Image.Image:
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


# ============================================================================
# EDGE MAPS
# ============================================================================

def _make_edges_enhanced(img: Image.Image, preserve_details: bool = True) -> Image.Image:
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


# ============================================================================
# FALLBACK OVAL CROP
# ============================================================================

def _crop_face_oval(img: Image.Image,
                    face_box: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple]:
    x1, y1, x2, y2 = face_box
    w, h = x2 - x1, y2 - y1
    pad_x   = int(w * 0.25)
    pad_top = int(h * 0.40)
    pad_bot = int(h * 0.10)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_top)
    cx2 = min(img.width,  x2 + pad_x)
    cy2 = min(img.height, y2 + pad_bot)
    cropped = img.crop((cx1, cy1, cx2, cy2)).convert("RGBA")
    cw, ch  = cropped.size
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, cw - 1, ch - 1), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(cw, ch) * 0.015))))
    cropped.putalpha(mask)
    return cropped, (cx1, cy1, cx2, cy2)


# ============================================================================
# SEGMENTATION
# ============================================================================

def segment_and_crop(img: Image.Image) -> Tuple[Image.Image, Image.Image, Dict]:
    if not REMBG_AVAILABLE:
        raise ImportError("rembg not installed")
    _init_rembg()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    segmented = remove(img, session=_REMBG_SESSION)
    cropped_rgba, crop_info = _crop_to_content(segmented, padding_pct=0.05)
    white_bg = Image.new("RGB", cropped_rgba.size, (255, 255, 255))
    white_bg.paste(cropped_rgba, mask=cropped_rgba.split()[3])
    return cropped_rgba, white_bg, crop_info


# ============================================================================
# FRUPreprocessedData — in-memory result container
# ============================================================================

class FRUPreprocessedData:
    """
    In-memory container holding all FRU artifacts needed by fru_generator.
    """

    def __init__(
        self,
        face_img: Image.Image,
        face_edges: Image.Image,
        face_mask_crop: Optional[np.ndarray],
        face_box_in_sketch: Optional[Tuple[int, int, int, int]],
        faceid_embedding: Optional[np.ndarray],
        gender: str,
        primary_face: Optional[Dict],
        face_crop_box: Optional[Tuple],
        original_size: Tuple[int, int],
        crop_info: Dict,
    ):
        self.face_img           = face_img
        self.face_edges         = face_edges
        self.face_mask_crop     = face_mask_crop
        self.face_box_in_sketch = face_box_in_sketch
        self.faceid_embedding   = faceid_embedding
        self.gender             = gender
        self.primary_face       = primary_face
        self.face_crop_box      = face_crop_box
        self.original_size      = original_size
        self.crop_info          = crop_info

    @property
    def has_face_embedding(self) -> bool:
        return self.faceid_embedding is not None


# ============================================================================
# MAIN FRU PREPROCESSING FUNCTION
# ============================================================================

def preprocess_fru_image_in_memory(
    img: Image.Image,
    gender_override: Optional[str] = None,
    gamma: float = PREPROCESS_DEFAULTS["gamma"],
    preserve_details: bool = PREPROCESS_DEFAULTS["preserve_details"],
    enhance_faces: bool = PREPROCESS_DEFAULTS["enhance_faces"],
) -> FRUPreprocessedData:
    """
    Full FRU preprocessing pipeline operating entirely in RAM.
    No files written to disk.
    """
    # 1. Background removal + auto-crop
    cropped_rgba, img_white_bg, crop_info = segment_and_crop(img)
    current_size = img_white_bg.size

    # 2. Preprocessing corrections
    img_corrected = _preprocess_for_sketch(img_white_bg, gamma=gamma, enhance_contrast=True)

    # 3. Face detection
    all_faces = _detect_all_faces(img_corrected)
    primary_face = all_faces[0] if all_faces else None

    # 4. Gender resolution: override → detection → unknown
    override = gender_override.strip().lower() if gender_override else None
    if override in ("male", "female"):
        gender = override
        print(f"  [INFO] Gender from override: {gender}")
    elif primary_face:
        gender = _detect_gender(cropped_rgba, primary_face["box"])
        print(f"  [INFO] Gender from detection: {gender}")
    else:
        gender = "unknown"
        print("  [WARN] Gender unknown — no face detected and no override provided")

    # 5. Face enhancement
    enhanced = img_corrected.copy()
    if primary_face and enhance_faces:
        enhanced = _enhance_face_region(enhanced, primary_face)

    # 6. Face embedding
    faceid_embedding = None
    if primary_face and INSIGHTFACE_AVAILABLE:
        faceid_embedding = _extract_face_embedding(cropped_rgba, primary_face["box"])
        if faceid_embedding is not None:
            print(f"  [INFO] Face embedding extracted: shape {faceid_embedding.shape}")
        else:
            print("  [WARN] Face embedding returned None — IP-Adapter will be skipped")

    # 7. BiSeNet face crop
    face_img = None
    face_edges = None
    face_crop_box = None
    face_mask_crop = None
    face_box_in_sketch = None

    bisenet_net = _init_bisenet()

    if bisenet_net is not None:
        try:
            device_bs = _get_torch_device()
            transform_bs = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            bs_img = cropped_rgba.convert("RGB")
            orig_w, orig_h = bs_img.size
            tensor_bs = transform_bs(bs_img).unsqueeze(0).to(device_bs)
            with torch.no_grad():
                bs_out, _, _ = bisenet_net(tensor_bs)
            bs_out_full = F.interpolate(
                bs_out, (orig_h, orig_w), mode="bilinear", align_corners=True
            )
            parsing = bs_out_full.squeeze(0).argmax(0).cpu().numpy()

            FACE_HAIR_LABELS = _BISENET_FACE_WITH_HAIR
            mask = np.zeros(parsing.shape, dtype=np.uint8)
            for lid in FACE_HAIR_LABELS:
                mask[parsing == lid] = 255
            mask = cv2.GaussianBlur(mask, (7, 7), 0)
            mask = (mask > 128).astype(np.uint8) * 255

            face_only_mask = np.zeros(parsing.shape, dtype=np.uint8)
            for lid in _BISENET_FACE_ONLY:
                face_only_mask[parsing == lid] = 255
            face_only_mask = cv2.GaussianBlur(face_only_mask, (7, 7), 0)
            face_only_mask = (face_only_mask > 128).astype(np.uint8) * 255

            rows = np.any(mask > 0, axis=1)
            cols = np.any(mask > 0, axis=0)

            if rows.any():
                pad = 20
                y1_bs = max(0, int(np.where(rows)[0][0]) - pad)
                y2_bs = min(orig_h, int(np.where(rows)[0][-1]) + pad)
                x1_bs = max(0, int(np.where(cols)[0][0]) - pad)
                x2_bs = min(orig_w, int(np.where(cols)[0][-1]) + pad)
                face_crop_box = (x1_bs, y1_bs, x2_bs, y2_bs)

                face_crop_rgba = cropped_rgba.crop(face_crop_box)
                face_mask_crop = mask[y1_bs:y2_bs, x1_bs:x2_bs]
                face_only_mask_crop = face_only_mask[y1_bs:y2_bs, x1_bs:x2_bs]

                fo_rows = np.any(face_only_mask_crop > 0, axis=1)
                fo_cols = np.any(face_only_mask_crop > 0, axis=0)
                bisenet_face_box_in_crop = None
                if fo_rows.any() and fo_cols.any():
                    bisenet_face_box_in_crop = (
                        int(np.where(fo_cols)[0][0]),
                        int(np.where(fo_rows)[0][0]),
                        int(np.where(fo_cols)[0][-1]),
                        int(np.where(fo_rows)[0][-1]),
                    )

                face_crop_rgb = Image.new("RGB", face_crop_rgba.size, (255, 255, 255))
                face_crop_rgb.paste(face_crop_rgba, mask=face_crop_rgba.split()[3])
                face_img_rgb = _preprocess_for_sketch(face_crop_rgb, gamma=gamma, enhance_contrast=True)

                if primary_face and enhance_faces:
                    fx1_m = max(0, primary_face["box"][0] - x1_bs)
                    fy1_m = max(0, primary_face["box"][1] - y1_bs)
                    fx2_m = min(face_crop_rgba.width,  primary_face["box"][2] - x1_bs)
                    fy2_m = min(face_crop_rgba.height, primary_face["box"][3] - y1_bs)
                    remapped = {
                        "box": (fx1_m, fy1_m, fx2_m, fy2_m),
                        "confidence": primary_face["confidence"],
                        "size": (fx2_m - fx1_m, fy2_m - fy1_m),
                        "index": 0,
                    }
                    face_img_rgb = _enhance_face_region(face_img_rgb, remapped)

                upscale_factor = 1.0
                if min(face_img_rgb.size) < 512:
                    s = 512 / min(face_img_rgb.size)
                    upscale_factor = s
                    face_img_rgb = face_img_rgb.resize(
                        (int(face_img_rgb.width * s), int(face_img_rgb.height * s)),
                        Image.LANCZOS,
                    )
                    new_mask_size = (face_img_rgb.width, face_img_rgb.height)
                    face_mask_crop = np.array(
                        Image.fromarray(face_mask_crop).resize(new_mask_size, Image.NEAREST)
                    )

                face_img   = face_img_rgb
                face_edges = _make_edges_enhanced(face_img_rgb, preserve_details=preserve_details)

                if bisenet_face_box_in_crop is not None:
                    bfx1, bfy1, bfx2, bfy2 = bisenet_face_box_in_crop
                    face_box_in_sketch = (
                        int(bfx1 * upscale_factor),
                        int(bfy1 * upscale_factor),
                        int(bfx2 * upscale_factor),
                        int(bfy2 * upscale_factor),
                    )
                    print(f"  [INFO] BiSeNet face-only bbox in sketch: {face_box_in_sketch}")
                elif primary_face:
                    fx1_m = max(0, primary_face["box"][0] - x1_bs)
                    fy1_m = max(0, primary_face["box"][1] - y1_bs)
                    fx2_m = min(face_crop_rgba.width,  primary_face["box"][2] - x1_bs)
                    fy2_m = min(face_crop_rgba.height, primary_face["box"][3] - y1_bs)
                    face_box_in_sketch = (
                        int(fx1_m * upscale_factor),
                        int(fy1_m * upscale_factor),
                        int(fx2_m * upscale_factor),
                        int(fy2_m * upscale_factor),
                    )
                    print(f"  [WARN] BiSeNet face-only mask empty — fell back to MTCNN bbox")
                else:
                    print("  [WARN] No face box available for scene placement")
            else:
                print("  [WARN] BiSeNet found no face regions — will skip generation")

        except Exception as e:
            print(f"  [WARN] BiSeNet crop failed: {e} — falling back to oval crop")
            bisenet_net = None

    # Fallback: oval crop if BiSeNet unavailable or failed
    if bisenet_net is None and primary_face:
        print("  [WARN] Using fallback oval crop")
        face_img_oval, face_crop_box = _crop_face_oval(enhanced, primary_face["box"])
        if min(face_img_oval.size) < 512:
            s = 512 / min(face_img_oval.size)
            face_img_oval = face_img_oval.resize(
                (int(face_img_oval.width * s), int(face_img_oval.height * s)),
                Image.LANCZOS,
            )
        face_img_rgb = Image.new("RGB", face_img_oval.size, (255, 255, 255))
        face_img_rgb.paste(face_img_oval, mask=face_img_oval.split()[3])
        face_img   = face_img_rgb
        face_edges = _make_edges_enhanced(face_img_rgb, preserve_details=preserve_details)
        print("  [INFO] Fallback oval crop generated")

    return FRUPreprocessedData(
        face_img=face_img,
        face_edges=face_edges,
        face_mask_crop=face_mask_crop,
        face_box_in_sketch=face_box_in_sketch,
        faceid_embedding=faceid_embedding,
        gender=gender,
        primary_face=primary_face,
        face_crop_box=face_crop_box,
        original_size=current_size,
        crop_info=crop_info,
    )