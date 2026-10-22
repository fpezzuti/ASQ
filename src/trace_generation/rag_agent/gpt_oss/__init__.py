from .builder import build_gpt_oss

SUPPORTED_MODELS = {
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",
}

__all__ = ["SUPPORTED_MODELS", "build_gpt_oss"]
