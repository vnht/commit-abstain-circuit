import re
import string

_ABSTENTION_PHRASES_NORM = [
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "unknown",
    "cannot determine",
    "can't determine",
    "cannot answer",
    "can't answer",
    "not enough information",
    "insufficient information",
    "it is not possible to determine",
    "this cannot be determined",
    "the context does not say",
    "the provided context does not contain enough information",
]


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lstrip(string.punctuation + " ")
    return text


def is_abstention(response):
    norm = normalize_text(response)
    return any(norm.startswith(phrase) for phrase in _ABSTENTION_PHRASES_NORM)


def parse_decision(response):
    if not response or not response.strip():
        return "malformed"
    return "abstain" if is_abstention(response) else "commit"


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_answer_correctness(predicted, gold):
    if gold is None:
        return None

    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    exact_match = pred_norm == gold_norm

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return {"exact_match": exact_match, "f1": 1.0 if exact_match else 0.0}

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return {"exact_match": exact_match, "f1": 0.0}

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"exact_match": exact_match, "f1": f1}
