"""Shared model-to-logo-image mapping.

Place provider logo images (square PNGs recommended) in:
    <project_root>/data/images/models/

Naming convention: use the provider prefix from the model string,
e.g. "anthropic.png" for "anthropic/claude-sonnet-4.6".
"""

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

MODELS_IMAGE_DIR = Path(__file__).parent.parent / "data" / "images" / "models"

# Map from model provider prefix to image filename.
# Add entries here as you add new provider logos.
MODEL_IMAGE_MAP: dict[str, str] = {
    "anthropic": "anthropic.png",
    "openai": "openai.png",
    "google": "google.png",
    "deepseek": "deepseek.png",
    "x-ai": "x-ai.png",
    "meta": "meta.png",
    "mistral": "mistral.png",
    "moonshotai": "kimi.png",
}


def load_model_image(model: str) -> np.ndarray | None:
    """Load a provider logo for the given model string (e.g. 'openai/gpt-5.4').

    Returns the image as a numpy array, or None if not found.
    """
    provider = model.split("/")[0]
    filename = MODEL_IMAGE_MAP.get(provider)
    if filename is None:
        return None
    path = MODELS_IMAGE_DIR / filename
    if not path.exists():
        return None
    try:
        return mpimg.imread(str(path))
    except Exception:
        return None
