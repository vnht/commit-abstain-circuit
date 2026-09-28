import string
import torch


def build_token_families(tokenizer, abstention_phrases, vocab_size, device="cpu"):
    """Build abstention and commit token masks for logit scoring.

    Returns:
        abstention_mask: bool tensor (vocab_size,) — True for abstention tokens
        commit_mask: bool tensor (vocab_size,) — True for commit-eligible tokens
        family_info: dict documenting which tokens map to which phrases
    """
    abstention_ids = set()
    family_info = {}

    for phrase in abstention_phrases:
        try:
            ids = tokenizer.encode(phrase, add_special_tokens=False)
            if ids:
                first_id = ids[0]
                abstention_ids.add(first_id)
                decoded = tokenizer.decode([first_id])
                family_info[first_id] = {"phrase": phrase, "first_token": decoded}
        except Exception:
            pass

    exclude_ids = set(abstention_ids)

    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            ids = val if isinstance(val, list) else [val]
            exclude_ids.update(ids)

    all_ids = list(range(vocab_size))
    try:
        all_decoded = tokenizer.batch_decode([[i] for i in all_ids])
    except Exception:
        all_decoded = []
        for i in all_ids:
            try:
                all_decoded.append(tokenizer.decode([i]))
            except Exception:
                all_decoded.append("")

    for idx, decoded in enumerate(all_decoded):
        stripped = decoded.strip()
        if not stripped or all(c in string.punctuation for c in stripped):
            exclude_ids.add(idx)

    abstention_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for idx in abstention_ids:
        if idx < vocab_size:
            abstention_mask[idx] = True

    exclude_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    for idx in exclude_ids:
        if idx < vocab_size:
            exclude_mask[idx] = True

    commit_mask = ~exclude_mask

    return abstention_mask, commit_mask, family_info


def compute_decision_scores(logits, abstention_mask, commit_mask):
    """Compute Δ(x) = s_commit - s_abstain from first-step logits.

    Args:
        logits: tensor (vocab_size,) — raw logits at first generation step
        abstention_mask: bool tensor (vocab_size,)
        commit_mask: bool tensor (vocab_size,)

    Returns:
        dict with s_commit, s_abstain, delta
    """
    if abstention_mask.any():
        s_abstain = logits[abstention_mask].max().item()
    else:
        s_abstain = float("-inf")

    if commit_mask.any():
        s_commit = logits[commit_mask].max().item()
    else:
        s_commit = float("-inf")

    return {
        "s_commit": s_commit,
        "s_abstain": s_abstain,
        "delta": s_commit - s_abstain,
    }
