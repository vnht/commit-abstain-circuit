#!/usr/bin/env python3
"""Delta-Eval: Commit-Abstain Score Validation.

Runs inference on a model × dataset pair, computes the commit-abstain score
Δ(x) = s_commit − s_abstain from first-step logits, and evaluates AUROC and
decision metrics against ground-truth answerability labels.

Usage:
    python delta-eval/run.py --model gemma-4b --dataset squad
    python delta-eval/run.py --model ministral-3b --dataset kuq
    python delta-eval/run.py --model qwen-4b --dataset musique -n 100
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import MODEL_SHORTCUTS, DATASET_CONFIGS, ABSTENTION_PHRASES, MAX_NEW_TOKENS
from core.data import load_dataset
from core.inference import load_model, run_inference
from core.scoring import build_token_families, compute_decision_scores
from core.metrics import compute_confusion, compute_decision_metrics, compute_auroc

import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("fla").setLevel(logging.ERROR)

import transformers
transformers.utils.logging.set_verbosity_error()

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Delta-Eval: Commit-Abstain Score")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_SHORTCUTS.keys()),
        help="Model shortcut",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset name",
    )
    parser.add_argument(
        "-n", "--num-instances",
        type=int,
        default=None,
        help="Max number of instances to process (default: all)",
    )
    return parser.parse_args()


def load_existing_results(jsonl_path):
    """Load already-completed examples from a JSONL file for resumption."""
    done = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["question_id"]] = r
    return done


def process_examples(data, model, tokenizer, model_id, device,
                     abstention_mask, commit_mask, jsonl_path):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    done = load_existing_results(jsonl_path)
    if done:
        print(f"  Resuming: {len(done)}/{len(data)} already completed")

    results = []
    n = len(data)

    with open(jsonl_path, "a") as fout:
        for i, item in enumerate(data):
            qid = item["question_id"]

            if qid in done:
                results.append(done[qid])
                continue

            t0 = time.perf_counter()
            inf = run_inference(
                model, tokenizer, model_id, item["prompt"], device, MAX_NEW_TOKENS,
            )
            scores = compute_decision_scores(
                inf["first_step_logits"], abstention_mask, commit_mask,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            result = {
                "question_id": qid,
                "is_answerable": item["is_answerable"],
                "response": inf["response"],
                "first_token_id": inf["first_token_id"],
                "s_commit": scores["s_commit"],
                "s_abstain": scores["s_abstain"],
                "delta": scores["delta"],
                "score_decision": 1 if scores["delta"] > 0 else 0,
                "elapsed_ms": round(elapsed_ms),
            }

            fout.write(json.dumps(result, default=str) + "\n")
            fout.flush()
            results.append(result)

            correct = result["score_decision"] == (1 if item["is_answerable"] else 0)
            print(
                f"  {i+1}/{n}  {'✓' if correct else '✗'}  "
                f"Δ={scores['delta']:+.2f}  "
                f"ans={'Y' if item['is_answerable'] else 'N'}  "
                f"{elapsed_ms:.0f}ms",
            )

    return results


def evaluate_results(results):
    labels = [1 if r["is_answerable"] else 0 for r in results]
    deltas = [r["delta"] for r in results]
    score_decs = [r["score_decision"] for r in results]

    tp, tn, fp, fn = compute_confusion(labels, score_decs)
    metrics = compute_decision_metrics(tp, tn, fp, fn)
    auroc = compute_auroc(deltas, labels)

    ans_deltas   = [d for d, l in zip(deltas, labels) if l == 1]
    unans_deltas = [d for d, l in zip(deltas, labels) if l == 0]

    return {
        "n": len(results),
        "n_answerable": sum(labels),
        "n_unanswerable": len(labels) - sum(labels),
        "auroc": auroc,
        "overall_accuracy":  metrics["overall_accuracy"],
        "true_commitment":   metrics["true_commitment"],
        "false_commitment":  metrics["false_commitment"],
        "true_abstention":   metrics["true_abstention"],
        "false_abstention":  metrics["false_abstention"],
        "mean_delta_answerable":   sum(ans_deltas)   / len(ans_deltas)   if ans_deltas   else None,
        "mean_delta_unanswerable": sum(unans_deltas) / len(unans_deltas) if unans_deltas else None,
    }



def main():
    args = parse_args()
    model_id = MODEL_SHORTCUTS[args.model]
    dataset_name = args.dataset

    print(f"\n{'='*70}")
    print(f"Delta-Eval: {args.model} × {dataset_name}")
    print(f"Model: {model_id}")
    print(f"{'='*70}\n")

    hf_token = os.getenv("HF_TOKEN")

    print("Loading dataset...")
    data, dataset_type = load_dataset(dataset_name)
    if args.num_instances is not None:
        data = data[:args.num_instances]
        print(f"  Limited to first {args.num_instances} instances")
    n_a = sum(1 for x in data if x["is_answerable"])
    n_u = len(data) - n_a
    print(f"  {len(data)} examples ({n_a} answerable, {n_u} unanswerable)")

    print(f"\nLoading model: {model_id}...")
    model, tokenizer, device, vocab_size = load_model(model_id, hf_token)
    print(f"  Device: {device}")

    print("\nBuilding token families...")
    abstention_mask, commit_mask, _ = build_token_families(
        tokenizer, ABSTENTION_PHRASES, vocab_size, device="cpu",
    )

    jsonl_path = RESULTS_DIR / f"{args.model}_{dataset_name}.jsonl"

    print(f"\n--- Processing {len(data)} examples ---")
    results = process_examples(
        data, model, tokenizer, model_id, device,
        abstention_mask, commit_mask, jsonl_path,
    )

    evaluation = evaluate_results(results)

    e = evaluation
    print(f"\n{'='*70}")
    print(f"RESULTS: {args.model} × {dataset_name}")
    print(f"{'='*70}")
    print(f"  AUROC:              {e['auroc']:.3f}")
    print(f"  Overall Accuracy:   {e['overall_accuracy']:.3f}")
    print(f"  True Commitment:    {e['true_commitment']:.3f}")
    print(f"  False Commitment:   {e['false_commitment']:.3f}")
    print(f"  True Abstention:    {e['true_abstention']:.3f}")
    print(f"  False Abstention:   {e['false_abstention']:.3f}")
    if e["mean_delta_answerable"] is not None:
        print(f"  Δ (answerable):     {e['mean_delta_answerable']:+.3f}")
    if e["mean_delta_unanswerable"] is not None:
        print(f"  Δ (unanswerable):   {e['mean_delta_unanswerable']:+.3f}")

    out_path = RESULTS_DIR / f"{args.model}_{dataset_name}.json"
    output = {
        "model_id": model_id,
        "model_shortcut": args.model,
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "evaluation": evaluation,
        "per_example": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Per-example log: {jsonl_path}")


if __name__ == "__main__":
    main()
