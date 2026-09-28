#!/usr/bin/env python3
"""Functional Analysis — Amplification: Causal amplification of A and C components.

Two primary experiments:

  FC on A — Scales A-component outputs by α on unanswerable false-commitment (FC)
             instances and measures conversion to abstention.

  FA on C — Scales C-component outputs by α on answerable false-abstention (FA)
             instances and measures conversion to commitment.

Both sweeps use α ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 10, 100, 1000}.
α=1.0 is a mathematical no-op and serves as the intra-run baseline.

Requires:
  - delta-eval/results/{model}_{dataset}.jsonl  (baseline Δ values)
  - cac-localisation/results/{model}_joint_summary.json  (A/C components)

Usage:
    python functional-analysis/amplification.py --model llama-3b
    python functional-analysis/amplification.py --all
"""

import os
import sys
import json
import time
import argparse
import random
import warnings
import logging
from pathlib import Path

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
from core.inference import load_model, prepare_inputs
from core.scoring import build_token_families
from core.gating import _get_layers, _get_attn_module, _get_o_proj, _get_head_info

RESULTS_DIR = Path(__file__).parent / "results"
DELTA_EVAL_DIR = Path(__file__).parent.parent / "delta-eval" / "results"
CAC_RESULTS_DIR = Path(__file__).parent.parent / "cac-localisation" / "results"
JOINT_DATASETS = ["kuq", "musique", "squad"]

TARGET_MODELS = [
    "llama-3b", "qwen-4b", "gemma-4b", "ministral-3b", "phi-4-mini",
    "llama-8b", "qwen-9b", "gemma-12b", "ministral-14b", "phi-4",
]
ALPHA_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 10.0, 100.0, 1000.0]
CONTROL_ALPHAS = [1.0, 1.5, 2.0, 2.5, 3.0, 10.0, 100.0, 1000.0]
VERIFY_TOLERANCE = 0.05
N_VERIFY = 20


# ── data loading ──────────────────────────────────────────────────────

def load_val_split():
    """Load held-out val split (last 250 ans + 250 unans per dataset)."""
    all_data = []
    for ds in JOINT_DATASETS:
        data_ds, _ = load_dataset(ds)
        ans = sorted([x for x in data_ds if x["is_answerable"]],
                     key=lambda x: x["question_id"])
        unans = sorted([x for x in data_ds if not x["is_answerable"]],
                       key=lambda x: x["question_id"])
        half = 250
        selected = ans[half:half * 2] + unans[half:half * 2]
        for x in selected:
            x["source_dataset"] = ds
        all_data.extend(selected)
    random.seed(42)
    random.shuffle(all_data)
    return all_data


def load_baseline_deltas(model_short, val_ids):
    """Load baseline Δ(x) from delta-eval JSONL results."""
    lookup = {}
    for ds in JOINT_DATASETS:
        p = DELTA_EVAL_DIR / f"{model_short}_{ds}.jsonl"
        if not p.exists():
            print(f"  WARNING: {p.name} not found — will compute fresh baselines")
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                qid = r["question_id"]
                if qid in val_ids and qid not in lookup:
                    lookup[qid] = r["delta"]
    return lookup


def load_cac_summary(model_short):
    """Load CAC classification from cac-localisation results."""
    p = CAC_RESULTS_DIR / f"{model_short}_joint_summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"CAC summary not found: {p}\n"
            f"Run: python cac-localisation/identify_cac.py --model {model_short}"
        )
    with open(p) as f:
        return json.load(f)


# ── hook-based component scaling ──────────────────────────────────────

class ComponentScaler:
    """Install forward hooks to scale specific attention heads and MLP sublayers.

    The hook for attention head (ℓ, h) multiplies the head's slice of the
    concatenated pre-o_proj tensor by α before W^O is applied.
    The hook for MLP ℓ multiplies the sublayer output by α before residual add.
    At α=1.0 the intervention is a mathematical no-op.
    """

    def __init__(self, model, model_id):
        self._layers = _get_layers(model)
        self.n_layers = len(self._layers)
        self.n_heads, self.head_dim = _get_head_info(self._layers, model)
        self._hooks: list = []
        self._head_targets: dict[int, dict[int, float]] = {}
        self._mlp_targets: dict[int, float] = {}

    def configure(self, components: list[dict], alpha: float):
        """Prepare target sets for a given component list and scale factor."""
        self._head_targets = {}
        self._mlp_targets = {}
        for comp in components:
            l = comp["layer"]
            if comp["type"] == "head":
                self._head_targets.setdefault(l, {})[comp["head"]] = alpha
            else:
                self._mlp_targets[l] = alpha

    def attach(self):
        """Install scaling hooks."""
        self.detach()
        for layer_idx, head_scales in self._head_targets.items():
            layer = self._layers[layer_idx]
            attn = _get_attn_module(layer)
            o_proj = _get_o_proj(attn)
            self._hooks.append(
                o_proj.register_forward_pre_hook(
                    self._make_head_hook(head_scales)
                )
            )
        for layer_idx, scale in self._mlp_targets.items():
            layer = self._layers[layer_idx]
            self._hooks.append(
                layer.mlp.register_forward_hook(
                    self._make_mlp_hook(scale)
                )
            )

    def detach(self):
        """Remove all installed hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _make_head_hook(self, head_scales: dict[int, float]):
        hd = self.head_dim

        def hook(_module, args):
            x = args[0].clone()
            for h_idx, scale in head_scales.items():
                s = h_idx * hd
                x[..., s:s + hd] = x[..., s:s + hd] * scale
            return (x,)

        return hook

    @staticmethod
    def _make_mlp_hook(scale: float):
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                return (out[0] * scale,) + out[1:]
            return out * scale

        return hook


# ── forward pass ──────────────────────────────────────────────────────

@torch.no_grad()
def run_forward(model, tokenizer, model_id, device,
                commit_mask, abstention_mask, ex) -> float:
    """Single forward pass; returns Δ(x) = s_commit − s_abstain."""
    inp = prepare_inputs(model_id, tokenizer, ex["prompt"], device)
    fwd = {"input_ids": inp["input_ids"]}
    if inp.get("attention_mask") is not None:
        fwd["attention_mask"] = inp["attention_mask"]
    logits = model(**fwd).logits[0, -1, :].float().cpu()
    cl = logits.clone()
    cl[~commit_mask] = float("-inf")
    al = logits.clone()
    al[~abstention_mask] = float("-inf")
    return (cl.max() - al.max()).item()


def run_sweep(model, tokenizer, model_id, device,
              commit_mask, abstention_mask,
              scaler, components, alphas, examples,
              n_denom: int = 0,
              desc="",
              track_abstain: bool = False) -> dict[str, list[dict]]:
    """Run forward passes for each α and return per-example deltas."""
    results = {}
    n_total = len(examples)
    denom = n_denom if n_denom > 0 else n_total
    LOG_EVERY = max(1, n_total // 10)
    metric_label = "FA" if track_abstain else "FC"

    for alpha in alphas:
        tag = f"{alpha:.1f}"
        label = f"α={alpha:.1f}{' ' + desc if desc else ''}"
        scaler.configure(components, alpha)
        scaler.attach()
        per_ex = []
        n_commit = 0
        t0 = time.time()
        try:
            for i, ex in enumerate(examples):
                delta = run_forward(
                    model, tokenizer, model_id, device,
                    commit_mask, abstention_mask, ex
                )
                commits = delta > 0.0
                if commits:
                    n_commit += 1
                per_ex.append({
                    "question_id": ex["question_id"],
                    "delta_alpha": delta,
                    "commits": commits,
                })
                done = i + 1
                if done % LOG_EVERY == 0 or done == n_total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_total - done) / rate if rate > 0 else 0
                    tracked = (n_total - n_commit) / denom if track_abstain else n_commit / denom
                    print(f"    {label:<22}  {done:>4}/{n_total}  "
                          f"{metric_label}={tracked:.3f}  "
                          f"Δ={delta:+.3f}  "
                          f"{elapsed:.0f}s  ETA {eta:.0f}s",
                          flush=True)
        finally:
            scaler.detach()
        elapsed = time.time() - t0
        n_abstain = n_total - n_commit
        tracked_final = n_abstain / denom if track_abstain else n_commit / denom
        tracked_n = n_abstain if track_abstain else n_commit
        print(f"    {label:<22}  done  {metric_label} rate={tracked_final:.3f}  "
              f"({tracked_n}/{denom})  total {elapsed:.0f}s",
              flush=True)
        results[tag] = per_ex
    return results


# ── metrics ───────────────────────────────────────────────────────────

def compute_fc_metrics(per_ex: list[dict], baseline_deltas: dict[str, float],
                       n_unans: int) -> dict:
    """Compute FC rate with n_unans as denominator."""
    n_fc_instances = len(per_ex)
    if n_fc_instances == 0 or n_unans == 0:
        return {"n": 0, "fc_rate": None, "conversion": None, "mean_delta": None}
    n_still_commit = sum(1 for x in per_ex if x["commits"])
    mean_delta = sum(x["delta_alpha"] for x in per_ex) / n_fc_instances
    baseline_vals = [baseline_deltas[x["question_id"]]
                     for x in per_ex if x["question_id"] in baseline_deltas]
    fc_rate = n_still_commit / n_unans
    baseline_fc_rate = n_fc_instances / n_unans
    conversion = baseline_fc_rate - fc_rate
    return {
        "n_fc_instances": n_fc_instances,
        "n_unans": n_unans,
        "n_still_commit": n_still_commit,
        "fc_rate": fc_rate,
        "baseline_fc_rate": baseline_fc_rate,
        "conversion": conversion,
        "mean_delta": mean_delta,
        "mean_baseline_delta": sum(baseline_vals) / len(baseline_vals) if baseline_vals else None,
    }


def compute_fa_metrics(per_ex: list[dict], n_ans: int) -> dict:
    """Compute FA rate with n_ans as denominator."""
    n_fa_instances = len(per_ex)
    if n_fa_instances == 0 or n_ans == 0:
        return {"n": 0, "fa_rate": None, "conversion": None, "mean_delta": None}
    n_still_abstain = sum(1 for x in per_ex if not x["commits"])
    mean_delta = sum(x["delta_alpha"] for x in per_ex) / n_fa_instances
    fa_rate = n_still_abstain / n_ans
    baseline_fa_rate = n_fa_instances / n_ans
    conversion = baseline_fa_rate - fa_rate
    return {
        "n_fa_instances": n_fa_instances,
        "n_ans": n_ans,
        "n_still_abstain": n_still_abstain,
        "fa_rate": fa_rate,
        "baseline_fa_rate": baseline_fa_rate,
        "conversion": conversion,
        "mean_delta": mean_delta,
    }


# ── verification ──────────────────────────────────────────────────────

def verify_alpha_one(baseline_deltas, sweep_results, examples, tol=VERIFY_TOLERANCE):
    """Spot-check that α=1.0 reproduces baseline Δ values."""
    tag = "1.0"
    if tag not in sweep_results:
        return None, 0, False
    per_ex_1 = sweep_results[tag]
    diffs = []
    for row in per_ex_1:
        qid = row["question_id"]
        if qid in baseline_deltas:
            diffs.append(abs(row["delta_alpha"] - baseline_deltas[qid]))
    if not diffs:
        return None, 0, False
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    passed = mean_diff < tol
    print(f"\n  α=1.0 verification vs delta-eval baseline:")
    print(f"    Checked {len(diffs)} examples")
    print(f"    Max |Δ diff|  = {max_diff:.4f}")
    print(f"    Mean |Δ diff| = {mean_diff:.4f}  (tolerance={tol})")
    print(f"    {'✓ PASS' if passed else '✗ FAIL (proceeding anyway)'}")
    return max_diff, len(diffs), passed


# ── main per-model routine ─────────────────────────────────────────────

def run_model(model_short, verify_only=False):
    out_path = RESULTS_DIR / f"{model_short}_amplification.json"
    if out_path.exists():
        print(f"\n  {model_short}: already done ({out_path.name}), skipping")
        return

    model_id = MODEL_SHORTCUTS[model_short]
    print(f"\n{'='*70}")
    print(f"Functional Analysis — Amplification: {model_short}  ({model_id})")
    print(f"{'='*70}\n")

    print("Loading CAC summary ...")
    cac = load_cac_summary(model_short)
    a_comps = cac["abstention_promoting"]
    c_comps = cac["commitment_promoting"]
    print(f"  A-components: {len(a_comps)}  C-components: {len(c_comps)}")

    if len(a_comps) == 0:
        print(f"  WARNING: no A-components found for {model_short}, skipping")
        return

    print("\nLoading val split ...")
    val_data = load_val_split()
    val_ids = {x["question_id"] for x in val_data}
    n_ans_total = sum(1 for x in val_data if x["is_answerable"])
    n_unans_total = sum(1 for x in val_data if not x["is_answerable"])
    print(f"  Val split: {len(val_data)} examples "
          f"({n_ans_total} ans, {n_unans_total} unans)")

    print("\nLoading baseline Δ from delta-eval ...")
    baseline_deltas = load_baseline_deltas(model_short, val_ids)
    print(f"  Loaded {len(baseline_deltas)}/{len(val_ids)} baselines")

    # FC instances: unanswerable, model commits at baseline (Δ > 0)
    fc_examples = []
    missing_baseline = 0
    for ex in val_data:
        qid = ex["question_id"]
        if qid not in baseline_deltas:
            missing_baseline += 1
            continue
        if not ex["is_answerable"] and baseline_deltas[qid] > 0.0:
            fc_examples.append(ex)

    n_unans = sum(1 for x in val_data
                  if not x["is_answerable"] and x["question_id"] in baseline_deltas)
    n_ans = sum(1 for x in val_data
                if x["is_answerable"] and x["question_id"] in baseline_deltas)

    # FA instances: answerable, model abstains at baseline (Δ ≤ 0)
    fa_examples = [
        ex for ex in val_data
        if ex["is_answerable"]
        and ex["question_id"] in baseline_deltas
        and baseline_deltas[ex["question_id"]] <= 0.0
    ]

    if missing_baseline:
        print(f"  ({missing_baseline} instances skipped: no baseline)")
    print(f"\n  Unanswerable (Y=0):        {n_unans}")
    print(f"  FC instances  (Y=0, Δ>0): {len(fc_examples)}  "
          f"(baseline FC rate = {len(fc_examples)/n_unans:.3f})")
    print(f"  Answerable (Y=1):          {n_ans}")
    print(f"  FA instances  (Y=1, Δ≤0): {len(fa_examples)}  "
          f"(baseline FA rate = {len(fa_examples)/n_ans:.3f})")

    if len(fc_examples) == 0:
        print(f"  ERROR: no FC instances found for {model_short}, skipping")
        return

    if verify_only:
        print("\n--verify-only: loading model to check α=1.0 reproduction ...")

    print(f"\nLoading model ...")
    hf_token = os.getenv("HF_TOKEN")
    try:
        model, tokenizer, device, vocab_size = load_model(model_id, hf_token)
    except (OSError, EnvironmentError) as e:
        print(f"  SKIP: could not load {model_id}\n  Reason: {e}")
        return None
    for p in model.parameters():
        p.requires_grad_(False)

    abstention_mask, commit_mask, _ = build_token_families(
        tokenizer, ABSTENTION_PHRASES, vocab_size, device="cpu"
    )

    scaler = ComponentScaler(model, model_id)
    print(f"  {scaler.n_layers} layers × {scaler.n_heads} heads  "
          f"(head_dim={scaler.head_dim})")

    # ── Experiment 1: FC on A — A-component amplification on FC instances ──
    print(f"\n[FC on A] A-component sweep on {len(fc_examples)} FC instances ...")
    fc_a_sweep = run_sweep(
        model, tokenizer, model_id, device,
        commit_mask, abstention_mask,
        scaler, a_comps, ALPHA_VALUES, fc_examples,
        n_denom=n_unans,
        desc="(A-comp, FC)",
    )

    # Verify α=1.0 no-op
    print(f"\nVerifying α=1.0 no-op on {N_VERIFY} instances ...")
    verify_examples = fc_examples[:N_VERIFY]
    verify_sweep = run_sweep(
        model, tokenizer, model_id, device,
        commit_mask, abstention_mask,
        scaler, a_comps, [1.0], verify_examples,
        n_denom=n_unans,
        desc="(verify)",
    )
    max_diff, n_checked, passed = verify_alpha_one(
        baseline_deltas, verify_sweep, verify_examples
    )

    if verify_only:
        print("\n--verify-only: done.")
        return

    # ── Experiment 2: FA on C — C-component amplification on FA instances ──
    fa_c_sweep = None
    if len(fa_examples) > 0 and len(c_comps) > 0:
        print(f"\n[FA on C] C-component sweep on {len(fa_examples)} FA instances ...")
        fa_c_sweep = run_sweep(
            model, tokenizer, model_id, device,
            commit_mask, abstention_mask,
            scaler, c_comps, ALPHA_VALUES, fa_examples,
            n_denom=n_ans,
            desc="(C-comp, FA)",
            track_abstain=True,
        )

    # ── Assemble result ──────────────────────────────────────────────────
    baseline_fc_rate = len(fc_examples) / n_unans
    baseline_fa_rate = len(fa_examples) / n_ans if n_ans else 0.0
    baseline_mean_delta_fc = (
        sum(baseline_deltas[x["question_id"]] for x in fc_examples
            if x["question_id"] in baseline_deltas)
        / len(fc_examples)
    )

    fc_a_metrics = {
        "1.0": {
            "n_fc_instances": len(fc_examples),
            "n_unans": n_unans,
            "n_still_commit": len(fc_examples),
            "fc_rate": baseline_fc_rate,
            "baseline_fc_rate": baseline_fc_rate,
            "conversion": 0.0,
            "mean_delta": baseline_mean_delta_fc,
            "source": "baseline (no forward passes)",
        }
    }
    for alpha in ALPHA_VALUES:
        tag = f"{alpha:.1f}"
        fc_a_metrics[tag] = compute_fc_metrics(fc_a_sweep[tag], baseline_deltas, n_unans)

    result = {
        "model_shortcut": model_short,
        "model_id": model_id,
        "n_unans": n_unans,
        "n_ans": n_ans,
        "n_fc_instances": len(fc_examples),
        "n_fa_instances": len(fa_examples),
        "baseline_fc_rate": baseline_fc_rate,
        "baseline_fa_rate": baseline_fa_rate,
        "n_a_components": len(a_comps),
        "n_c_components": len(c_comps),
        "a_components": a_comps,
        "c_components": c_comps,
        "verification": {
            "max_abs_diff_vs_baseline": max_diff,
            "n_checked": n_checked,
            "passed": passed,
            "tolerance": VERIFY_TOLERANCE,
        },
        "fc_on_a": {
            tag: {
                "metrics": fc_a_metrics[tag],
                "per_example": fc_a_sweep[tag] if tag != "1.0" else [],
            }
            for tag in ["1.0"] + [f"{a:.1f}" for a in ALPHA_VALUES]
        },
    }

    if fa_c_sweep is not None:
        baseline_mean_delta_fa = (
            sum(baseline_deltas[x["question_id"]] for x in fa_examples
                if x["question_id"] in baseline_deltas)
            / len(fa_examples)
        ) if fa_examples else 0.0
        fa_c_metrics = {
            "1.0": {
                "n_fa_instances": len(fa_examples),
                "n_ans": n_ans,
                "n_still_abstain": len(fa_examples),
                "fa_rate": baseline_fa_rate,
                "baseline_fa_rate": baseline_fa_rate,
                "conversion": 0.0,
                "mean_delta": baseline_mean_delta_fa,
                "source": "baseline (no forward passes)",
            }
        }
        for alpha in ALPHA_VALUES:
            tag = f"{alpha:.1f}"
            fa_c_metrics[tag] = compute_fa_metrics(fa_c_sweep[tag], n_ans)
        result["fa_on_c"] = {
            tag: {
                "metrics": fa_c_metrics[tag],
                "per_example": fa_c_sweep[tag] if tag != "1.0" else [],
            }
            for tag in ["1.0"] + [f"{a:.1f}" for a in ALPHA_VALUES]
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Saved → {out_path}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Functional Analysis — Amplification")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=TARGET_MODELS,
                       help="Run a single model")
    group.add_argument("--all", action="store_true",
                       help="Run all target models sequentially")
    ap.add_argument("--verify-only", action="store_true",
                    help="Load model and verify α=1.0 reproduction only")
    return ap.parse_args()


def main():
    args = parse_args()
    models = TARGET_MODELS if args.all else [args.model]

    t_total = time.time()
    for m in models:
        run_model(m, verify_only=args.verify_only)

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"All done.  Total wall time: {elapsed/60:.1f} min")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
