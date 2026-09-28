import logging
import torch
import transformers
from transformers import AutoConfig

from .base import _apply_generation_config

MODELS = [
    "microsoft/phi-4",
    "microsoft/Phi-4-mini-instruct",
]


class _SuppressRopeWarning(logging.Filter):
    def filter(self, record):
        return "original_max_position_embeddings" not in record.getMessage()


def _patch_rope_factor(config):
    for attr in ("rope_scaling", "rope_parameters"):
        rope = getattr(config, attr, None)
        if (
            isinstance(rope, dict)
            and "original_max_position_embeddings" in rope
            and "factor" not in rope
        ):
            rope["factor"] = config.max_position_embeddings / rope["original_max_position_embeddings"]
            setattr(config, attr, rope)


def load(model_id: str, hf_token: str):
    rope_filter = _SuppressRopeWarning()
    tf_logger = logging.getLogger("transformers")
    for handler in tf_logger.handlers:
        handler.addFilter(rope_filter)
    try:
        config = AutoConfig.from_pretrained(model_id, token=hf_token)
        _patch_rope_factor(config)
        pipe = transformers.pipeline(
            "text-generation",
            model=model_id,
            config=config,
            model_kwargs={"dtype": torch.bfloat16},
            device_map="auto",
            token=hf_token,
        )
        _apply_generation_config(pipe, model_id, hf_token)
    finally:
        for handler in tf_logger.handlers:
            handler.removeFilter(rope_filter)
    return pipe
