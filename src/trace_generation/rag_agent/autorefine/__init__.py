
from .builder import build_autorefine

SUPPORTED_MODELS = {
    "autorefine": "yrshi/AutoRefine-Qwen2.5-3B-Instruct",
}

__all__ = ["SUPPORTED_MODELS", "build_autorefine"]
