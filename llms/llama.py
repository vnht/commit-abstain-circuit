from .base import load_pipeline

MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def load(model_id: str, hf_token: str):
    return load_pipeline(model_id, hf_token)
