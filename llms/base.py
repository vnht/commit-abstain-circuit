import torch
import transformers
from transformers import GenerationConfig


def load_pipeline(model_id: str, hf_token: str) -> transformers.Pipeline:
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"dtype": torch.bfloat16},
        device_map="auto",
        token=hf_token,
    )
    _apply_generation_config(pipe, model_id, hf_token)
    return pipe


def _apply_generation_config(pipe: transformers.Pipeline, model_id: str, hf_token: str):
    try:
        gen_config = GenerationConfig.from_pretrained(model_id, token=hf_token)
    except Exception:
        gen_config = GenerationConfig()

    gen_config.max_new_tokens = 256
    gen_config.max_length = None
    gen_config.do_sample = True
    gen_config.temperature = 0.7

    eos = pipe.tokenizer.eos_token_id
    gen_config.pad_token_id = eos[0] if isinstance(eos, list) else eos

    pipe.model.generation_config = gen_config
