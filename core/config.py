from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

CLOSED_BOOK_TEMPLATE = """Read the question carefully. Answer in 10 words or less.

If the question is answerable, give the answer directly.
If the question is not answerable from your knowledge, or there is not enough information to answer reliably, say "I don't know".

Question:
{question}"""

OPEN_BOOK_TEMPLATE = """Read the context and question carefully. Answer in 10 words or less.

Answer the question only if it is supported by the provided context.
If the question is not answerable from the provided context, or there is not enough information in the context to answer reliably, say "I don't know".

Context:
{context}

Question:
{question}"""

ABSTENTION_PHRASES = [
    "I don't know",
    "I do not know",
    "I cannot determine",
    "I cannot answer",
    "Unfortunately, I don't have enough information",
    "Unfortunately, I cannot answer",
    "Insufficient information to answer",
    "Insufficient information",
    "Unanswerable based on context",
    "Unanswerable",
    "Unclear from the given context",
    "Unknown; context does not specify",
    "Unknown",
    "Uncertain; lacks a definitive answer",
    "Uncertain",
    "Cannot be determined from context",
    "Cannot determine",
    "Cannot answer",
    "Unable to answer from the passage",
    "Unable to determine",
    "Sorry, I don't have information",
    "Sorry, I cannot answer",
    "Beyond my knowledge",
    "Impossible to determine without context",
    "Impossible to answer",
    "Without more context, I cannot answer",
    "Without sufficient information",
    "Ambiguous; the premise is unclear",
    "Ambiguous",
    "Unsure; evidence is inconclusive",
    "Unsure",
    "Unknowable from the text provided",
    "Unknowable",
    "Indeterminate based on context",
    "Indeterminate",
    "The answer cannot be determined",
    "The context does not provide enough information",
    "The provided context does not contain enough information",
    "This question cannot be answered",
    "This cannot be determined",
    "It is not possible to determine",
    "It cannot be determined",
    "There is insufficient information",
    "There is not enough information",
    "Based on the context, this is unanswerable",
    "Based on the provided context, I cannot determine",
    "Not enough information",
    "Not possible to determine",
    "Given limited context, I cannot answer",
    "Given the information provided, I cannot determine",
]

MODEL_SHORTCUTS = {
    "gemma-4b":      "google/gemma-3-4b-it",
    "gemma-12b":     "google/gemma-3-12b-it",
    "gemma-27b":     "google/gemma-3-27b-it",
    "ministral-3b":  "mistralai/Ministral-3-3B-Instruct-2512",
    "ministral-14b": "mistralai/Ministral-3-14B-Instruct-2512",
    "qwen-4b":       "Qwen/Qwen3.5-4B",
    "qwen-9b":       "Qwen/Qwen3.5-9B",
    "qwen-35b":      "Qwen/Qwen3.5-35B-A3B",
    "phi-4-mini":    "microsoft/Phi-4-mini-instruct",
    "phi-4":         "microsoft/phi-4",
    "llama-3b":      "meta-llama/Llama-3.2-3B-Instruct",
    "llama-8b":      "meta-llama/Llama-3.1-8B-Instruct",
}

DATASET_CONFIGS = {
    "kuq":      {"path": "datasets/kuq.json",      "type": "closed_book"},
    "musique":  {"path": "datasets/musique.json",   "type": "open_book"},
    "squad":    {"path": "datasets/squad.json",     "type": "open_book"},
    "hotpotqa": {"path": "datasets/hotpotqa.json",  "type": "open_book"},
    "selfaware": {"path": "datasets/selfaware.json", "type": "closed_book"},
}

MAX_NEW_TOKENS = 256
