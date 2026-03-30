"""
Central configuration for the sketch pipeline.
All paths, model locations, and tunable defaults live here.
Edit this file when deploying to a new environment - nothing else should need changing.

NOTE: All from_pretrained() / from_config() calls in the pipeline modules use
local_files_only=True.  No network access to Hugging Face will ever be attempted
at runtime; every weight must be present under MODELS_ROOT before the server starts.
"""

from pathlib import Path

# ============================================================================
# ROOT DIRECTORIES
# All paths are relative to the project root (where this file lives).
# When volume-mounting models into Docker, mount them at /app/models/
# ============================================================================

_PROJECT_ROOT = Path(__file__).parent

# Model weights - volume-mounted at /app/models/ inside Docker
MODELS_ROOT = _PROJECT_ROOT / "models"

# Scene backgrounds and crop configuration - volume-mounted at /app/inputs/
SCENES_DIR        = _PROJECT_ROOT / "inputs" / "scenes"
CROPS_CONFIG_PATH = _PROJECT_ROOT / "crops.json"

# FRU-specific scene backgrounds and crop configuration
FRU_SCENES_DIR        = _PROJECT_ROOT / "inputs" / "scenes" / "face_bg"
FRU_CROPS_CONFIG_PATH = _PROJECT_ROOT / "crops-face-meta.json"


# ============================================================================
# MODEL PATHS
# ============================================================================

MODEL_PATHS = {
    # Core generative models
    "sd15":       str(MODELS_ROOT / "stable-diffusion-v1-5"),
    "controlnet": str(MODELS_ROOT / "controlnet" / "control_v11p_sd15_lineart"),

    # LoRA weights
    "lora":       str(MODELS_ROOT / "loras" / "Pencil_Sketch_by_vizsumit.safetensors"),
    "lcm_lora":   str(MODELS_ROOT / "loras" / "lcm-lora-sdv1-5"),

    # Tiny VAE for faster LCM decoding
    "taesd":      str(MODELS_ROOT / "taesd"),

    # IP-Adapter FaceID
    "ip_adapter": str(MODELS_ROOT / "ip_adapter" / "ip-adapter-faceid_sd15.bin"),

    # InsightFace (antelopev2 model pack)
    # FaceAnalysis expects root= pointing to the dir that contains models/antelopev2/
    "insightface": str(MODELS_ROOT / "insightface"),

    # ControlNet auxiliary annotators
    "annotators_lineart": str(MODELS_ROOT / "annotators" / "lineart"),
    "annotators_hed":     str(MODELS_ROOT / "annotators" / "hed"),

    # BiSeNet face parsing - used by FRU pipeline only
    "bisenet": str(MODELS_ROOT / "bisenet" / "bisenet_face_parsing.pth"),
}


# ============================================================================
# PREPROCESSING DEFAULTS
# ============================================================================

PREPROCESS_DEFAULTS = {
    "gamma":            1.3,
    "preserve_details": True,
    "enhance_faces":    True,
}


# ============================================================================
# GENERATION DEFAULTS
# ============================================================================

GENERATION_DEFAULTS = {
    "use_lora":        True,
    "lora_scale":      1.0,
    "faceid_strength": 0.85,
    "use_lcm":         True,
}


# ============================================================================
# POST-PROCESSING PARAMETERS
# ============================================================================

POSTPROCESS_PARAMS = {
    "sharpness": 3.5,
    "saturation": 0.0,
    "exposure":   0.9,
}
