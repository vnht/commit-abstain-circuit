import json
from .config import ROOT_DIR, DATASET_CONFIGS, CLOSED_BOOK_TEMPLATE, OPEN_BOOK_TEMPLATE


def load_dataset(dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    path = ROOT_DIR / cfg["path"]
    with open(path) as f:
        data = json.load(f)

    for item in data:
        if not item["is_answerable"]:
            item["gold_answer"] = "I don't know"
        else:
            item["gold_answer"] = item["expected_output"]

        item["prompt"] = build_prompt(item, cfg["type"])

    return data, cfg["type"]


def build_prompt(item, dataset_type):
    if dataset_type == "closed_book":
        return CLOSED_BOOK_TEMPLATE.format(question=item["question"])
    else:
        return OPEN_BOOK_TEMPLATE.format(
            context=item["context"],
            question=item["question"],
        )
