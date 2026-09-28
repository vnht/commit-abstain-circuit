#!/usr/bin/env python3
"""Functional Analysis — Ablation: Zero-ablation of A and C components.

Runs full zero-ablation of abstention-promoting (A) and commitment-promoting (C)
components from a CAC localisation summary. Records AUROC, directional sign flips,
and mean Δ(x) shifts on held-out validation data.

Requires cac-localisation results: {model}_joint_summary.json must exist in
cac-localisation/results/ before running.

Usage:
    python functional-analysis/ablation.py --model phi-4
    python functional-analysis/ablation.py --model phi-4 --dataset kuq
    python functional-analysis/ablation.py --all
"""

import os
import sys
import json
import argparse
import warnings
import logging
import random
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("fla").setLevel(logging.ERROR)

import transformers
transformers.utils.logging.set_verbosity_error()

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import torch

from core.config import MODEL_SHORTCUTS, ABSTENTION_PHRASES
from core.data import load_dataset
from core.inference import load_model
from core.scoring import build_token_families, compute_decision_scores
from core.gating import CausalComponentGating

RESULTS_DIR = Path(__file__).parent / "results"
CAC_RESULTS_DIR = Path(__file__).parent.parent / "cac-localisation" / "results"
JOINT_DATASETS = ["kuq", "musique", "squad"]


def parse_args():
    ap = argparse.ArgumentParser(description="Functional Analysis — Ablation")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODEL_SHORTCUTS.keys()))
    group.add_argument("--all", action="store_true", help="Run all models")
    ap.add_argument("--dataset", default=None, choices=JOINT_DATASETS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("-n", "--num-instances", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def load_cac_summary(model_short):
    src = CAC_RESULTS_DIR / f"{model_short}_joint_summary.json"
    if not src.exists():
        raise FileNotFoundError(
            f"No CAC summary for {model_short}.\n"
            f"Run: python cac-localisation/identify_cac.py --model {model_short}"
        )
    with open(src) as f:
        return json.load(f)


def load_val_data(dataset_name):
    """Load held-out validation split (last 250 ans + 250 unans per dataset)."""
    data_all, _ = load_dataset(dataset_name)
    ans = sorted([x for x in data_all if x["is_answerable"]],
                 key=lambda x: x["question_id"])
    unans = sorted([x for x in data_all if not x["is_answerable"]],
                   key=lambda x: x["question_id"])
    half = 250
    selected = ans[half:half * 2] + unans[half:half * 2]
    random.seed(42)
    random.shuffle(selected)
    return selected



def _tokenize_batch(prompts, model_id, tokenizer, device):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_ids = []
    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        kwargs = dict(return_tensors="pt", return_dict=True)
        if "Qwen" in model_id:
            kwargs.update(add_generation_prompt=True, enable_thinking=False)
        elif "Ministral" not in model_id:
            kwargs["add_generation_prompt"] = True
        toks = tokenizer.apply_chat_template(msgs, **kwargs)
        all_ids.append(toks["input_ids"][0])

    max_len = max(len(ids) for ids in all_ids)
    padded = torch.full((len(all_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attn = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    seq_lengths = torch.zeros(len(all_ids), dtype=torch.long)
    for i, ids in enumerate(all_ids):
        L = len(ids)
        padded[i, :L] = ids
        attn[i, :L] = 1
        seq_lengths[i] = L

    return {"input_ids": padded.to(device), "attention_mask": attn.to(device),
            "seq_lengths": seq_lengths}


def collect_deltas_batched(gating, data, model, tokenizer, model_id, device,
                           commit_mask, abstention_mask, batch_size=16):
    gating.attach()
    deltas = []
    for i in range(0, len(data), batch_size):
        chunk = data[i:i + batch_size]
        prompts = [ex["prompt"] for ex in chunk]
        batch = _tokenize_batch(prompts, model_id, tokenizer, device)
        seq_lengths = batch.pop("seq_lengths")
        with torch.no_grad():
            logits = model(**batch).logits.float().cpu()
        last_positions = seq_lengths - 1
        for j in range(len(chunk)):
            last_logits = logits[j, last_positions[j], :]
            sc = compute_decision_scores(last_logits, abstention_mask, commit_mask)
            deltas.append(sc["delta"])
    gating.detach()
    return deltas


def _stats(vals):
    if not vals:
        return {"mean": None, "median": None, "min": None, "max": None, "n": 0}
    s = sorted(vals)
    return {
        "mean": round(mean(s), 6),
        "median": round(s[len(s) // 2], 6),
        "min": round(s[0], 6),
        "max": round(s[-1], 6),
        "n": len(s),
    }


def compute_sign_flips(orig_deltas, new_deltas):
    """Compute directional sign flips with before/after Δ(x) for flipped inputs."""
    c_to_a = 0
    a_to_c = 0
    ca_before, ca_after = [], []
    ac_before, ac_after = [], []
    noflip_shifts = []
    all_before = []
    all_after = []

    for d_orig, d_new in zip(orig_deltas, new_deltas):
        all_before.append(d_orig)
        all_after.append(d_new)
        if d_orig > 0 and d_new <= 0:
            c_to_a += 1
            ca_before.append(d_orig)
            ca_after.append(d_new)
        elif d_orig <= 0 and d_new > 0:
            a_to_c += 1
            ac_before.append(d_orig)
            ac_after.append(d_new)
        else:
            noflip_shifts.append(d_new - d_orig)

    total = c_to_a + a_to_c
    n = len(orig_deltas)

    return {
        "sign_flips": total,
        "sign_flip_rate": round(total / n, 4) if n else 0,
        "commit_to_abstain": c_to_a,
        "abstain_to_commit": a_to_c,
        "commit_to_abstain_rate": round(c_to_a / n, 4) if n else 0,
        "abstain_to_commit_rate": round(a_to_c / n, 4) if n else 0,
        "all_delta_before": _stats(all_before),
        "all_delta_after": _stats(all_after),
        "all_shift": _stats([b - a for a, b in zip(all_before, all_after)]),
        "ca_delta_before": _stats(ca_before),
        "ca_delta_after": _stats(ca_after),
        "ca_shift": _stats([a - b for b, a in zip(ca_before, ca_after)]),
        "ac_delta_before": _stats(ac_before),
        "ac_delta_after": _stats(ac_after),
        "ac_shift": _stats([a - b for b, a in zip(ac_before, ac_after)]),
        "noflip_shift": _stats(noflip_shifts),
    }


def ablate_and_measure(gating, components, data, labels, orig_deltas,
                       model, tokenizer, model_id, device,
                       commit_mask, abstention_mask, batch_size):
    gating.set_all_open()
    for c in components:
        gating.suppress_component(
            c["layer"], c["type"],
            c.get("head") if c["type"] == "head" else None)

    deltas = collect_deltas_batched(
        gating, data, model, tokenizer, model_id, device,
        commit_mask, abstention_mask, batch_size=batch_size,
    )
    flips = compute_sign_flips(orig_deltas, deltas)
    md = sum(deltas) / len(deltas)

    ca_b = flips['ca_delta_before']['mean']
    ca_a = flips['ca_delta_after']['mean']
    ac_b = flips['ac_delta_before']['mean']
    ac_a = flips['ac_delta_after']['mean']
    print(f"    mean_Δ={md:+.4f}  "
          f"flips={flips['sign_flips']} (C→A:{flips['commit_to_abstain']} A→C:{flips['abstain_to_commit']})  "
          f"C→A: {ca_b}→{ca_a}  A→C: {ac_b}→{ac_a}",
          flush=True)

    return {
        "n_ablated": len(components),
        "mean_delta": round(md, 6),
        **flips,
    }


def aggregate_joint(model_short, force=False):
    out_path = RESULTS_DIR / f"{model_short}_all.json"
    if out_path.exists() and not force:
        print(f"  Joint file exists -> {out_path.name} (use --force)", flush=True)
        return out_path

    per_ds = {}
    for ds in JOINT_DATASETS:
        p = RESULTS_DIR / f"{model_short}_{ds}.json"
        if not p.exists():
            print(f"  Cannot aggregate: missing {p.name}", flush=True)
            return None
        with open(p) as f:
            per_ds[ds] = json.load(f)

    ref = per_ds[JOINT_DATASETS[0]]
    SCALAR_KEYS = [
        "mean_delta",
        "sign_flips", "sign_flip_rate",
        "commit_to_abstain", "abstain_to_commit",
        "commit_to_abstain_rate", "abstain_to_commit_rate",
    ]
    NESTED_KEYS = [
        "all_delta_before", "all_delta_after", "all_shift",
        "ca_delta_before", "ca_delta_after", "ca_shift",
        "ac_delta_before", "ac_delta_after", "ac_shift",
        "noflip_shift",
    ]

    joint = {
        "model_shortcut": model_short,
        "datasets": JOINT_DATASETS,
        "n_examples_per_dataset": ref["n_examples"],
        "n_examples_total": ref["n_examples"] * len(JOINT_DATASETS),
        "baseline_mean_delta": round(mean(
            per_ds[ds]["baseline_mean_delta"] for ds in JOINT_DATASETS), 6),
        "n_abstention_components": ref["n_abstention_components"],
        "n_commitment_components": ref["n_commitment_components"],
    }

    for test in ["ablate_A", "ablate_C"]:
        t = {"n_ablated": ref[test]["n_ablated"]}
        for key in SCALAR_KEYS:
            vals = [per_ds[ds][test][key] for ds in JOINT_DATASETS
                    if per_ds[ds][test][key] is not None]
            if vals:
                t[f"{key}_mean"] = round(mean(vals), 6)
                t[f"{key}_per_dataset"] = {ds: per_ds[ds][test][key] for ds in JOINT_DATASETS}
            else:
                t[f"{key}_mean"] = None
                t[f"{key}_per_dataset"] = {ds: None for ds in JOINT_DATASETS}
        for key in NESTED_KEYS:
            t[key] = {"per_dataset": {ds: per_ds[ds][test][key] for ds in JOINT_DATASETS}}
            for stat in ["mean", "median", "min", "max", "n"]:
                vals = [per_ds[ds][test][key][stat] for ds in JOINT_DATASETS
                        if per_ds[ds][test][key][stat] is not None]
                t[key][f"{stat}_across_ds"] = round(mean(vals), 6) if vals else None
        joint[test] = t

    elapsed = sum(per_ds[ds]["elapsed_seconds"] for ds in JOINT_DATASETS)
    joint["elapsed_seconds_total"] = round(elapsed, 2)

    with open(out_path, "w") as f:
        json.dump(joint, f, indent=2)

    print(f"\n  Joint: baseline_Δ={joint['baseline_mean_delta']:+.4f}  "
          f"ablate_A_Δ={joint['ablate_A']['mean_delta_mean']:+.4f}  "
          f"ablate_C_Δ={joint['ablate_C']['mean_delta_mean']:+.4f}  "
          f"-> {out_path.name}", flush=True)
    return out_path


def run_model(model_short, datasets, batch_size, num_instances, force):
    model_id = MODEL_SHORTCUTS[model_short]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Functional Analysis — Ablation: {model_short}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"{'='*70}\n")

    summary = load_cac_summary(model_short)
    n_a = summary["n_abstention"]
    n_c = summary["n_commitment"]
    print(f"  Abstention: {n_a}  Commitment: {n_c}  Irrelevant: {summary['n_irrelevant']}")

    hf_token = os.getenv("HF_TOKEN")
    print("\nLoading model ...", flush=True)
    model, tokenizer, device, vocab_size = load_model(model_id, hf_token)
    for p in model.parameters():
        p.requires_grad_(False)

    abstention_mask, commit_mask, _ = build_token_families(
        tokenizer, ABSTENTION_PHRASES, vocab_size, device="cpu",
    )

    gating = CausalComponentGating(model, model_id).to(device)
    if summary.get("active_components"):
        gating.set_active_components(summary["active_components"])
    print(f"  Active gates: {gating.n_active} / {gating.n_components}")

    a_comps = summary["abstention_promoting"]
    c_comps = summary["commitment_promoting"]

    for ds_name in datasets:
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'─'*60}", flush=True)

        out_path = RESULTS_DIR / f"{model_short}_{ds_name}.json"
        if out_path.exists() and not force:
            print(f"  Already done -> {out_path.name}, skipping (use --force)")
            continue

        data = load_val_data(ds_name)
        if num_instances:
            data = data[:num_instances]
        labels = [1 if x["is_answerable"] else 0 for x in data]
        print(f"  {len(data)} examples  ({sum(labels)} answerable)")

        print("\n  Baseline ...", flush=True)
        gating.set_all_open()
        baseline_deltas = collect_deltas_batched(
            gating, data, model, tokenizer, model_id, device,
            commit_mask, abstention_mask, batch_size=batch_size,
        )
        baseline_md = sum(baseline_deltas) / len(baseline_deltas)
        n_commit = sum(1 for d in baseline_deltas if d > 0)
        n_abstain = len(baseline_deltas) - n_commit
        print(f"    mean_Δ={baseline_md:+.4f}  commit={n_commit}  abstain={n_abstain}")

        output = {
            "model_shortcut": model_short,
            "dataset_name": ds_name,
            "n_examples": len(data),
            "baseline_mean_delta": round(baseline_md, 6),
            "baseline_commit_count": n_commit,
            "baseline_abstain_count": n_abstain,
            "n_abstention_components": n_a,
            "n_commitment_components": n_c,
        }

        print(f"\n  --- Ablate A ({n_a} components) ---", flush=True)
        output["ablate_A"] = ablate_and_measure(
            gating, a_comps, data, labels, baseline_deltas,
            model, tokenizer, model_id, device,
            commit_mask, abstention_mask, batch_size)

        print(f"\n  --- Ablate C ({n_c} components) ---", flush=True)
        output["ablate_C"] = ablate_and_measure(
            gating, c_comps, data, labels, baseline_deltas,
            model, tokenizer, model_id, device,
            commit_mask, abstention_mask, batch_size)

        elapsed = time.time() - t0
        output["elapsed_seconds"] = round(elapsed, 2)

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Done {ds_name} ({elapsed:.0f}s) -> {out_path.name}", flush=True)

    if len(datasets) == len(JOINT_DATASETS):
        aggregate_joint(model_short, force=force)

    del model, tokenizer, gating
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    datasets = [args.dataset] if args.dataset else JOINT_DATASETS

    if args.all:
        for model_short in MODEL_SHORTCUTS:
            try:
                run_model(model_short, datasets, args.batch_size,
                          args.num_instances, args.force)
            except Exception as e:
                print(f"\n  ERROR on {model_short}: {e}", flush=True)
                continue
    else:
        run_model(args.model, datasets, args.batch_size,
                  args.num_instances, args.force)

    print(f"\n{'='*70}")
    print("All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
