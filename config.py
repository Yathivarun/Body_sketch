"""
Central configuration for the sketch pipeline.
All paths, model locations, and tunable defaults live here.
Edit this file when deploying to a new environment — nothing else should need changing.
"""

from pathlib import Path

# ============================================================================
# ROOT DIRECTORIES
# All paths are relative to the project root (where this file lives).
# When volume-mounting models into Docker, mount them at /app/models/
# ============================================================================

_PROJECT_ROOT = Path(__file__).parent

# Model weights — volume-mounted at /app/models/ inside Docker
MODELS_ROOT = _PROJECT_ROOT / "models"

# Scene backgrounds and crop configuration — volume-mounted at /app/inputs/
SCENES_DIR       = _PROJECT_ROOT / "inputs" / "scenes"
CROPS_CONFIG_PATH = _PROJECT_ROOT / "crops.json"


# ============================================================================
# MODEL PATHS
# ============================================================================

MODEL_PATHS = {
    # Core generative models
    "sd15":       str(MODELS_ROOT / "stable-diffusion-v1-5"),
    "controlnet": str(MODELS_ROOT / "controlnet" / "control_v11p_sd15_lineart"),

    # LoRA weights
    "lora":       str(MODELS_ROOT / "loras" / "Pencil_Sketch_by_vizsumit.safetensors"),
    "lcm_lora":   str(MODELS_ROOT / "loras" / "lcm-lora-sdv1-5"),   # local fallback

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
}


# ============================================================================
# PREPROCESSING DEFAULTS
# ============================================================================

PREPROCESS_DEFAULTS = {
    "gamma":            1.3,
    "preserve_details": True,   # fine lineart vs coarse
    "enhance_faces":    True,   # face region sharpening
}


# ============================================================================
# GENERATION DEFAULTS
# ============================================================================

GENERATION_DEFAULTS = {
    "use_lora":        True,
    "lora_scale":      1.0,
    "faceid_strength": 0.85,
    "use_lcm":         True,    # LCM-LoRA fast inference (4 steps)
}


# ============================================================================
# POST-PROCESSING PARAMETERS
# ============================================================================

POSTPROCESS_PARAMS = {
    "sharpness": 3.5,
    "saturation": 0.0,   # 0 = full grayscale
    "exposure":   0.9,
}
