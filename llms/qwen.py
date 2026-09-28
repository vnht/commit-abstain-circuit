import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-35B-A3B",
]


class QwenPipeline:
    """Wrapper that disables Qwen3's thinking mode and matches pipeline output format."""

    def __init__(self, model, tokenizer, gen_config, device):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = device

    def __call__(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                generation_config=self.gen_config,
            )[0]

        new_tokens = output_ids[len(inputs["input_ids"][0]):]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return [{"generated_text": messages + [{"role": "assistant", "content": response}]}]


def load(model_id: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    device = next(model.parameters()).device

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

    return QwenPipeline(model, tokenizer, gen_config, device)
