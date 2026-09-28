from . import gemma, llama, mistral, phi, qwen

ALL_MODELS: dict[str, callable] = {
    model_id: module.load
    for module in (gemma, llama, mistral, phi, qwen)
    for model_id in module.MODELS
}
