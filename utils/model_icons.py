"""Map model / vendor identity labels to brand icons and colors.

Icons live in ``data/images/models/`` as ``<vendor>.png``. The judge in the
identity studies emits free-text identity labels ("Qwen", "Claude 3.5 Sonnet",
"DeepSeek-V3", "ChatGPT", "unknown", ...); this module normalizes any such label
to a canonical vendor, then to an icon path and a brand color, so the same
identity is drawn the same way across every chart.

If an icon file is missing the lookup returns ``None`` and callers should degrade
gracefully (text only) — that's the signal to add the PNG to the icons folder.
"""

from pathlib import Path

import matplotlib
import matplotlib.colors

ICON_DIR = Path(__file__).resolve().parent.parent / "data" / "images" / "models"

# Ordered (substrings -> canonical vendor). First match wins, so put more
# specific keys before generic ones.
_VENDOR_RULES: list[tuple[tuple[str, ...], str]] = [
    (("claude", "anthropic"), "anthropic"),
    (("chatgpt", "gpt", "openai", "o1", "o3", "o4"), "openai"),
    (("gemini", "gemma", "bard", "google", "palm"), "google"),
    (("grok", "x-ai", "xai", "x.ai"), "xai"),
    (("deepseek", "深度求索"), "deepseek"),
    (("kimi", "moonshot", "月之暗面"), "kimi"),
    (("qwen", "tongyi", "通义", "千问", "阿里", "alibaba"), "qwen"),
    (("chatglm", "glm", "zhipu", "z.ai", "z-ai", "智谱"), "zhipu"),
    (("ernie", "文心", "baidu", "百度", "wenxin"), "baidu"),
    (("minimax", "abab"), "minimax"),
    (("doubao", "豆包", "seed", "bytedance", "字节"), "bytedance"),
    (("hunyuan", "混元", "tencent", "腾讯"), "tencent"),
    (("step", "stepfun", "阶跃", "星辰"), "stepfun"),
    (("llama", "meta"), "meta"),
    (("mistral", "le chat"), "mistral"),
]

# Color is by *nationality*, not brand: every Chinese model is a shade of red,
# every Western/American model is a shade of blue, so a glance at a chart says
# "this model thinks it's Chinese / Western". Lightness distinguishes models
# within a nationality.
_CHINA_VENDORS: set[str] = {
    "deepseek", "qwen", "kimi", "zhipu", "baidu", "minimax",
    "bytedance", "tencent", "stepfun",
}
_WEST_VENDORS: set[str] = {
    "anthropic", "openai", "google", "xai", "meta", "mistral",
}

# Per-vendor position within its national colormap (Reds for CN, Blues for US).
# Hand-spread so models that commonly co-occur (e.g. Qwen vs DeepSeek) stay
# visually distinct. Higher = darker. Vendors not listed fall back to 0.6.
_VENDOR_SHADE: dict[str, float] = {
    # Chinese -> Reds
    "minimax": 0.40, "tencent": 0.46, "deepseek": 0.55, "baidu": 0.61,
    "kimi": 0.68, "stepfun": 0.73, "zhipu": 0.79, "qwen": 0.86, "bytedance": 0.93,
    # Western -> Blues
    "xai": 0.46, "openai": 0.56, "meta": 0.63, "google": 0.71,
    "anthropic": 0.81, "mistral": 0.89,
}

_UNKNOWN_COLOR = "#B6B6BE"

# Canonical vendor -> icon filename stem, when it differs from the vendor name.
_ICON_FILENAMES: dict[str, str] = {
    "xai": "x-ai",
}


def _icon_file(vendor: str) -> Path:
    return ICON_DIR / f"{_ICON_FILENAMES.get(vendor, vendor)}.png"


def canonical_vendor(label: str) -> str | None:
    """Normalize a free-text identity label to a canonical vendor, or None."""
    if not label:
        return None
    low = label.lower()
    for keys, vendor in _VENDOR_RULES:
        if any(k in low for k in keys):
            return vendor
    return None


def icon_path_for(label: str) -> Path | None:
    """Return the brand icon Path for an identity label, or None if unavailable."""
    vendor = canonical_vendor(label)
    if vendor is None:
        return None
    path = _icon_file(vendor)
    return path if path.exists() else None


# Models whose name earns them an icon better than their vendor logo. Ordered
# (substring -> icon stem); first match wins, so "gpt-5.6-sol" is matched by its
# full name rather than by a bare "sol" that would also catch "solar".
_MODEL_ICON_OVERRIDES: list[tuple[str, str]] = [
    ("gpt-5.6-luna", "moon"),
    ("gpt-5.6-terra", "earth"),
    ("gpt-5.6-sol", "sun"),
]


def themed_icon_path_for(label: str) -> Path | None:
    """Icon honoring per-model overrides, else the vendor brand icon."""
    low = (label or "").lower()
    for key, stem in _MODEL_ICON_OVERRIDES:
        if key in low:
            path = _icon_file(stem)
            if path.exists():
                return path
    return icon_path_for(label)


def nationality(label: str) -> str | None:
    """Classify an identity label as 'china', 'west', or None (unknown)."""
    vendor = canonical_vendor(label)
    if vendor in _CHINA_VENDORS:
        return "china"
    if vendor in _WEST_VENDORS:
        return "west"
    return None


def color_for(label: str) -> str:
    """Return a stable color for a label: red shade if Chinese, blue if Western.

    Lightness is fixed per vendor so the same model is the same color in every
    chart; unknown/unclassified labels are gray.
    """
    vendor = canonical_vendor(label)
    if vendor is None:
        return _UNKNOWN_COLOR
    shade = _VENDOR_SHADE.get(vendor, 0.6)
    nat = nationality(label)
    if nat == "china":
        return matplotlib.colors.to_hex(matplotlib.colormaps["Reds"](shade))
    if nat == "west":
        return matplotlib.colors.to_hex(matplotlib.colormaps["Blues"](shade))
    return _UNKNOWN_COLOR


def missing_icons(labels) -> list[str]:
    """Given identity labels, return the canonical vendors whose icon PNG is absent."""
    missing = set()
    for label in labels:
        vendor = canonical_vendor(label)
        if vendor is not None and not _icon_file(vendor).exists():
            missing.add(vendor)
    return sorted(missing)
