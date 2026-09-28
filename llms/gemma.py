from .base import load_pipeline

MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]


def load(model_id: str, hf_token: str):
    return load_pipeline(model_id, hf_token)
