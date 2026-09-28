import sys
import torch
from unittest.mock import MagicMock
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    FineGrainedFP8Config,
    GenerationConfig,
)

# triton is not available on macOS; mock it so FP8 dequantization still works
if sys.platform == "darwin":
    for _mod in ("triton", "triton.language", "triton.runtime", "triton.runtime.jit"):
        sys.modules.setdefault(_mod, MagicMock())

MODELS = [
    "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Ministral-3-14B-Instruct-2512",
]


class MistralPipeline:
    """Thin wrapper around Mistral3 model that matches the transformers pipeline output format."""

    def __init__(self, model, tokenizer, gen_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = device

    def __call__(self, messages):
        tokenized = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                generation_config=self.gen_config,
            )[0]

        new_tokens = output_ids[len(input_ids[0]):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return [{"generated_text": messages + [{"role": "assistant", "content": response}]}]


def load(model_id: str, hf_token: str):
    tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)

    # FP8 dequantization must happen on CPU — MPS does not support Float8_e4m3fn
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        token=hf_token,
        quantization_config=FineGrainedFP8Config(dequantize=True),
    )

    # After dequantization weights are BF16; move to MPS if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)

    try:
        gen_config = GenerationConfig.from_pretrained(model_id, token=hf_token)
    except Exception:
        gen_config = GenerationConfig()

    gen_config.max_new_tokens = 256
    gen_config.max_length = None
    gen_config.do_sample = True
    gen_config.temperature = 0.7
    eos = tokenizer.eos_token_id
    gen_config.pad_token_id = eos[0] if isinstance(eos, list) else eos

    return MistralPipeline(model, tokenizer, gen_config, device)
