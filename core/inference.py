import sys
import torch
from unittest.mock import MagicMock

if sys.platform == "darwin":
    _triton_mods = [
        "triton", "triton.language", "triton.runtime", "triton.runtime.jit",
        "triton.backends", "triton.backends.compiler", "triton.backends.nvidia",
        "triton.compiler", "triton.compiler.compiler",
    ]
    for _mod in _triton_mods:
        sys.modules.setdefault(_mod, MagicMock())


def load_model(model_id, hf_token):
    """Load model and tokenizer. Returns (model, tokenizer, device, vocab_size)."""
    if "Ministral-3" in model_id:
        return _load_mistral(model_id, hf_token)
    elif "Qwen" in model_id:
        return _load_qwen(model_id, hf_token)
    elif "phi-4" in model_id.lower() or "Phi-4" in model_id:
        return _load_phi(model_id, hf_token)
    else:
        return _load_standard(model_id, hf_token)


def _get_vocab_size(model):
    cfg = model.config
    if hasattr(cfg, "vocab_size"):
        return cfg.vocab_size
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "vocab_size"):
        return cfg.text_config.vocab_size
    return model.get_input_embeddings().weight.shape[0]


def _load_standard(model_id, hf_token):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    device = next(model.parameters()).device
    vocab_size = _get_vocab_size(model)
    return model, tokenizer, device, vocab_size


def _load_qwen(model_id, hf_token):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    device = next(model.parameters()).device
    vocab_size = _get_vocab_size(model)
    return model, tokenizer, device, vocab_size


def _load_phi(model_id, hf_token):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(model_id, token=hf_token)
    for attr in ("rope_scaling", "rope_parameters"):
        rope = getattr(config, attr, None)
        if (isinstance(rope, dict)
                and "original_max_position_embeddings" in rope
                and "factor" not in rope):
            rope["factor"] = config.max_position_embeddings / rope["original_max_position_embeddings"]
            setattr(config, attr, rope)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    device = next(model.parameters()).device
    vocab_size = _get_vocab_size(model)
    return model, tokenizer, device, vocab_size


def _load_mistral(model_id, hf_token):
    from transformers import (
        Mistral3ForConditionalGeneration,
        MistralCommonBackend,
        FineGrainedFP8Config,
    )

    tokenizer = MistralCommonBackend.from_pretrained(model_id, token=hf_token)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="cpu",
        token=hf_token,
        quantization_config=FineGrainedFP8Config(dequantize=True),
    )

    device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device_str)
    model.eval()
    model.generation_config.max_length = None
    device = torch.device(device_str)
    vocab_size = _get_vocab_size(model)
    return model, tokenizer, device, vocab_size


def prepare_inputs(model_id, tokenizer, prompt, device):
    """Tokenize a prompt into model-ready input tensors."""
    messages = [{"role": "user", "content": prompt}]

    if "Ministral-3" in model_id:
        tokenized = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True,
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    if "Qwen" in model_id:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    inputs = tokenizer(text, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def run_inference(model, tokenizer, model_id, prompt, device, max_new_tokens=256):
    """Get first-step logits and greedy-decoded response in a single pass.

    Returns dict with: first_step_logits, first_token_id, response
    """
    inputs = prepare_inputs(model_id, tokenizer, prompt, device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    fwd_kwargs = {"input_ids": input_ids}
    if inputs.get("attention_mask") is not None:
        fwd_kwargs["attention_mask"] = inputs["attention_mask"]

    with torch.no_grad():
        fwd_out = model(**fwd_kwargs)
        first_step_logits = fwd_out.logits[0, -1, :].float().cpu()

    gen_kwargs = {
        **fwd_kwargs,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    eos = tokenizer.eos_token_id
    pad = eos[0] if isinstance(eos, list) else eos
    gen_kwargs["pad_token_id"] = pad

    with torch.no_grad():
        gen_out = model.generate(**gen_kwargs)

    new_ids = gen_out[0][input_len:]
    first_token_id = new_ids[0].item() if len(new_ids) > 0 else None
    response = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "first_step_logits": first_step_logits,
        "first_token_id": first_token_id,
        "response": response,
    }


_JUDGE_PROMPT = (
    'Given the question and the response below, classify the response as '
    '"commit" or "abstain".\n\n'
    'A response is "abstain" if it:\n'
    '- Says it does not know or cannot answer\n'
    '- Explains why the question cannot be answered or is unknowable\n'
    '- Rejects the premise of the question (e.g. "X does not exist", '
    '"there is no record of X")\n'
    '- Says there is no consensus, no evidence, or not enough information\n'
    '- Gives a speculative, hedged, or conditional non-answer\n\n'
    'A response is "commit" only if it provides a direct, specific answer '
    'to the question.\n\n'
    'Question: {question}\n'
    'Response: {response}\n\n'
    'Classification (one word):'
)


def classify_response(model, tokenizer, model_id, question, response, device):
    """Use the LLM to judge whether a response is a commitment or abstention."""
    prompt = _JUDGE_PROMPT.format(
        question=question.strip(),
        response=response.strip(),
    )
    inputs = prepare_inputs(model_id, tokenizer, prompt, device)

    gen_kwargs = {
        **{k: v for k, v in inputs.items() if v is not None},
        "max_new_tokens": 5,
        "do_sample": False,
    }
    eos = tokenizer.eos_token_id
    pad = eos[0] if isinstance(eos, list) else eos
    gen_kwargs["pad_token_id"] = pad

    with torch.no_grad():
        gen_out = model.generate(**gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    new_ids = gen_out[0][input_len:]
    judge_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip().lower()

    if "abstain" in judge_text:
        return "abstain"
    if "commit" in judge_text:
        return "commit"
    return "commit"
