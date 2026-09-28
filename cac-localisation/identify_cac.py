#!/usr/bin/env python3
"""Identify CAC components from multi-seed gate results.

Loads all joint seed results for a model, classifies each component as
abstention-promoting (A), commitment-promoting (C), or decision-irrelevant (I)
using a configurable score threshold and seed-consistency criterion.

Classification (score threshold τ_s, seed fraction τ_f):
  - abstention-promoting:  abstention score > τ_s in >= τ_f fraction of seeds
  - commitment-promoting:  commitment score > τ_s in >= τ_f fraction of seeds
  - decision-irrelevant:   neither of the above

Output is saved as {model}_joint_summary.json in results/, and is required
by functional-analysis scripts and the policy module.

Usage:
    python cac-localisation/identify_cac.py --model phi-4
    python cac-localisation/identify_cac.py --model phi-4 --seed-fraction 1.0
    python cac-localisation/identify_cac.py --all
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.config import MODEL_SHORTCUTS

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    ap = argparse.ArgumentParser(description="Identify CAC components from gate results")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODEL_SHORTCUTS.keys()))
    group.add_argument("--all", action="store_true",
                       help="Summarise all models with joint results")
    ap.add_argument("--score-threshold", type=float, default=0.5,
                    help="Component score threshold (default: 0.5)")
    ap.add_argument("--seed-fraction", type=float, default=0.8,
                    help="Fraction of seeds required (default: 0.8 = 80%%)")
    return ap.parse_args()


def identify_model(model_short, score_thresh=0.5, seed_frac=0.8):
    """Classify components for one model. Returns dict or None if no data."""
    files = sorted(RESULTS_DIR.glob(f"{model_short}_joint_seed*.json"))
    if not files:
        return None

    seeds = []
    for f in files:
        with open(f) as fh:
            seeds.append(json.load(fh))

    s0 = seeds[0]
    active_comps = s0.get("active_components", [])
    n_heads = s0["n_heads"]
    n_layers = s0["n_layers"]
    n_seeds = len(seeds)
    min_seeds = max(1, int(np.ceil(n_seeds * seed_frac)))

    agg_keys = ["head_commitment", "head_abstention", "head_irrelevant",
                "mlp_commitment", "mlp_abstention", "mlp_irrelevant"]
    agg = {}
    for k in agg_keys:
        agg[k] = np.mean([np.array(s[k]) for s in seeds], axis=0).tolist()

    abstention_promoting = []
    commitment_promoting = []
    irrelevant = []

    # When no active_components list is stored, classify all heads and MLPs
    if not active_comps:
        active_comps = []
        for l in range(n_layers):
            for h in range(n_heads):
                active_comps.append({"layer": l, "type": "head", "head": h})
            active_comps.append({"layer": l, "type": "mlp"})

    for comp in active_comps:
        l = comp["layer"]
        is_head = comp["type"] == "head"

        a_count = 0
        c_count = 0
        a_scores = []
        c_scores = []
        for s in seeds:
            if is_head:
                h = comp["head"]
                a_val = s["head_abstention"][l][h]
                c_val = s["head_commitment"][l][h]
            else:
                a_val = s["mlp_abstention"][l]
                c_val = s["mlp_commitment"][l]
            a_scores.append(a_val)
            c_scores.append(c_val)
            if a_val > score_thresh:
                a_count += 1
            if c_val > score_thresh:
                c_count += 1

        entry = {
            **comp,
            "mean_abstention": float(np.mean(a_scores)),
            "mean_commitment": float(np.mean(c_scores)),
            "seed_abstention_count": a_count,
            "seed_commitment_count": c_count,
        }

        is_a = a_count >= min_seeds
        is_c = c_count >= min_seeds

        if is_a and not is_c:
            entry["role"] = "abstention"
            abstention_promoting.append(entry)
        elif is_c and not is_a:
            entry["role"] = "commitment"
            commitment_promoting.append(entry)
        elif is_a and is_c:
            if np.mean(a_scores) > np.mean(c_scores):
                entry["role"] = "abstention"
                abstention_promoting.append(entry)
            else:
                entry["role"] = "commitment"
                commitment_promoting.append(entry)
        else:
            entry["role"] = "irrelevant"
            irrelevant.append(entry)

    abstention_promoting.sort(key=lambda x: x["mean_abstention"], reverse=True)
    commitment_promoting.sort(key=lambda x: x["mean_commitment"], reverse=True)

    a_layers = [c["layer"] for c in abstention_promoting]
    top_half_a = sum(1 for l in a_layers if l >= n_layers // 2) if a_layers else 0

    summary = {
        "model_shortcut": model_short,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_seeds": n_seeds,
        "score_threshold": score_thresh,
        "seed_fraction": seed_frac,
        "min_seeds_required": min_seeds,
        "n_active": len(active_comps),
        "n_abstention": len(abstention_promoting),
        "n_commitment": len(commitment_promoting),
        "n_irrelevant": len(irrelevant),
        "abstention_top_half_pct": (top_half_a / len(a_layers) * 100) if a_layers else 0,
        "active_components": active_comps,
        "abstention_promoting": abstention_promoting,
        "commitment_promoting": commitment_promoting,
        "irrelevant": irrelevant,
        **agg,
    }
    return summary


def main():
    args = parse_args()

    if args.all:
        models = list(MODEL_SHORTCUTS.keys())
    else:
        models = [args.model]

    for model_short in models:
        summary = identify_model(
            model_short,
            score_thresh=args.score_threshold,
            seed_frac=args.seed_fraction,
        )
        if summary is None:
            print(f"  {model_short}: no joint results found, skipping")
            continue

        out_path = RESULTS_DIR / f"{model_short}_joint_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        n_a = summary["n_abstention"]
        n_c = summary["n_commitment"]
        n_i = summary["n_irrelevant"]
        n_total = summary["n_active"]
        top_pct = summary["abstention_top_half_pct"]
        print(f"  {model_short:<16}  A={n_a}  C={n_c}  I={n_i}  "
              f"(of {n_total})  A_top={top_pct:.0f}%  "
              f"[{summary['n_seeds']} seeds, >={summary['min_seeds_required']} required]"
              f"  -> {out_path.name}")


if __name__ == "__main__":
    main()
