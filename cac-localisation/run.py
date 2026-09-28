#!/usr/bin/env python3
"""CAC Localisation: Causal Component Gating for the commit-abstain circuit.

Fits differentiable soft gates over attention heads and MLP sublayers to
identify which components causally support the commit-abstain decision.
Three-phase protocol following CHG (Nam et al., 2025):
    Phase 1  (λ = 0) : shared initialisation
    Phase 2a (λ > 0) : retention bias  → G⁺
    Phase 2b (λ < 0) : removal bias    → G⁻

From G⁺ and G⁻, a causal taxonomy is derived:
    Commitment-promoting  = G⁻           (resists removal → supports correct Δ)
    Abstention-promoting  = 1 − G⁺       (suppressed despite retention pressure)
    Decision-irrelevant   = G⁺ × (1−G⁻)  (follows regularisation)

After running this script, use identify_cac.py to classify components from
multi-seed results into A/C/I categories.

Optional pre-screening (--top-k):
    Run prescreen.py first to rank components by |mean Δ contribution diff|.
    Then pass --top-k N to restrict gate fitting to the N highest-ranked
    components. This reduces noise and cuts default training steps from 500
    to 60. Summaries are read from cac-localisation/results/ by default
    (override with --exp2-dir).

Usage:
    python cac-localisation/run.py --model gemma-4b --dataset joint
    python cac-localisation/run.py --model gemma-4b --dataset joint --seeds 3 --top-k 20
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

from core.config import MODEL_SHORTCUTS, DATASET_CONFIGS, ABSTENTION_PHRASES
from core.data import load_dataset
from core.inference import load_model, prepare_inputs
from core.scoring import build_token_families
from core.gating import CausalComponentGating

RESULTS_DIR = Path(__file__).parent / "results"
JOINT_DATASETS = ["kuq", "musique", "squad"]

def parse_args():
    ap = argparse.ArgumentParser(description="CAC Localisation: Causal Component Gating")
    ap.add_argument("--model", required=True, choices=list(MODEL_SHORTCUTS.keys()))
    ap.add_argument("--dataset", required=True,
                    choices=list(DATASET_CONFIGS.keys()) + ["joint"])
    ap.add_argument("--top-k", type=int, default=50,
                    help="Pre-screen: gate only the top-K components ranked by "
                         "prescreen.py attribution (default: 50)")
    ap.add_argument("--exp2-dir", type=Path,
                    default=Path(__file__).parent / "results",
                    help="Directory containing {model}_{dataset}_summary.json files "
                         "produced by prescreen.py (default: cac-localisation/results/)")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Number of random seeds (default: 10)")
    ap.add_argument("--steps", type=int, default=None,
                    help="Gradient steps per phase "
                         "(default: 60 with --top-k, 500 without)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lam", type=float, default=None,
                    help="Regularisation magnitude (default: adaptive)")
    ap.add_argument("-n", "--num-instances", type=int, default=None)
    args = ap.parse_args()
    if args.steps is None:
        args.steps = 60 if args.top_k else 500
    return args


# ── optional pre-screening ─────────────────────────────────────────────

def load_exp2_components(exp2_dir, model_short, dataset_name, top_k):
    """Load top-K ranked components from an experiment-2 attribution summary."""
    p = exp2_dir / f"{model_short}_{dataset_name}_summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Experiment-2 summary not found: {p}\n"
            f"Run experiment-2 first, or drop --top-k to gate all components."
        )
    with open(p) as f:
        ranked = json.load(f).get("ranked_components", [])
    if len(ranked) < top_k:
        print(f"  WARNING: exp-2 has only {len(ranked)} ranked components, "
              f"using all of them")
        top_k = len(ranked)
    return ranked[:top_k]


def load_exp2_components_joint(exp2_dir, model_short, top_k):
    """Load union of top-K components across all joint datasets."""
    seen = set()
    merged = []
    for ds in JOINT_DATASETS:
        p = exp2_dir / f"{model_short}_{ds}_summary.json"
        if not p.exists():
            raise FileNotFoundError(
                f"Experiment-2 summary not found: {p}\n"
                f"Run experiment-2 for {ds} first, or drop --top-k."
            )
        with open(p) as f:
            ranked = json.load(f).get("ranked_components", [])
        for c in ranked[:top_k]:
            key = (c["layer"], c["type"], c.get("head"))
            if key not in seen:
                seen.add(key)
                merged.append(c)
    merged.sort(key=lambda c: (c["layer"], c["type"], c.get("head", -1)))
    print(f"  Joint active set: {len(merged)} unique components "
          f"(union of top-{top_k} from {', '.join(JOINT_DATASETS)})")
    return merged


def load_joint_data(split="train", per_dataset=None):
    """Load pooled data from all joint datasets with deterministic split.

    Sort each dataset by question_id, then:
      train = first 250 ans + first 250 unans per dataset (1500 total)
      val   = last 250 ans + last 250 unans per dataset  (1500 total)

    per_dataset: if set, cap to this many examples per dataset (balanced).
    """
    all_data = []
    for ds in JOINT_DATASETS:
        data_ds, _ = load_dataset(ds)
        ans = sorted([x for x in data_ds if x["is_answerable"]],
                     key=lambda x: x["question_id"])
        unans = sorted([x for x in data_ds if not x["is_answerable"]],
                       key=lambda x: x["question_id"])
        half = min(250, per_dataset // 2) if per_dataset else 250
        if split == "train":
            selected = ans[:half] + unans[:half]
        else:
            selected = ans[half:half * 2] + unans[half:half * 2]
        for x in selected:
            x["source_dataset"] = ds
        all_data.extend(selected)
    random.seed(42)
    random.shuffle(all_data)
    return all_data


def precompute_clean(data, model, tokenizer, model_id, device,
                     commit_mask, abstention_mask):
    """Forward-pass every example WITHOUT gates to get Δ(x), t*(x), b*(x)."""
    items = []
    for i, ex in enumerate(data):
        inp = prepare_inputs(model_id, tokenizer, ex["prompt"], device)
        fwd = {"input_ids": inp["input_ids"]}
        if inp.get("attention_mask") is not None:
            fwd["attention_mask"] = inp["attention_mask"]

        with torch.no_grad():
            logits = model(**fwd).logits[0, -1, :].float().cpu()

        cl = logits.clone(); cl[~commit_mask] = float("-inf")
        al = logits.clone(); al[~abstention_mask] = float("-inf")

        items.append({
            "question_id": ex["question_id"],
            "prompt": ex["prompt"],
            "is_answerable": ex["is_answerable"],
            "delta": (cl.max() - al.max()).item(),
            "t_star": cl.argmax().item(),
            "b_star": al.argmax().item(),
        })
        if (i + 1) % 50 == 0 or i + 1 == len(data):
            print(f"  Pre-computed {i + 1}/{len(data)}", flush=True)
    return items


def fit_gates(gating, items, model, tokenizer, model_id, device, *,
              lam, n_steps, batch_size, lr_init, lr_final):
    """Run gradient-based gate fitting and return a training log.

    Loss = −(1/B) Σ (2y−1)·Δ(x|G)  +  R(G, λ)
    """
    optimizer = torch.optim.Adam(
        [gating.head_logits, gating.mlp_logits], lr=lr_init,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0,
        end_factor=lr_final / lr_init, total_iters=n_steps,
    )

    gating.attach()
    n_data = len(items)
    log = []

    for step in range(n_steps):
        optimizer.zero_grad()
        batch = [items[random.randint(0, n_data - 1)] for _ in range(batch_size)]

        sum_delta_ans, sum_delta_unans = 0.0, 0.0
        n_a, n_u = 0, 0

        for ex in batch:
            inp = prepare_inputs(model_id, tokenizer, ex["prompt"], device)
            fwd = {"input_ids": inp["input_ids"]}
            if inp.get("attention_mask") is not None:
                fwd["attention_mask"] = inp["attention_mask"]

            out = model(**fwd)
            logits_last = out.logits[0, -1, :].float()
            delta = logits_last[ex["t_star"]] - logits_last[ex["b_star"]]

            sign = 1.0 if ex["is_answerable"] else -1.0
            (-(sign * delta) / batch_size).backward()

            with torch.no_grad():
                dv = delta.item()
            if ex["is_answerable"]:
                sum_delta_ans += dv; n_a += 1
            else:
                sum_delta_unans += dv; n_u += 1

            del out, logits_last, delta

        reg = gating.regularization(lam)
        reg.backward()

        optimizer.step()
        scheduler.step()

        is_last = (step + 1) == n_steps
        if (step + 1) % 10 == 0 or step == 0 or is_last:
            hg = gating.head_gates.detach()
            mg = gating.mlp_gates.detach()
            entry = {
                "step": step + 1,
                "mean_delta_ans": sum_delta_ans / n_a if n_a else 0,
                "mean_delta_unans": sum_delta_unans / n_u if n_u else 0,
                "mean_head_gate": hg.mean().item(),
                "mean_mlp_gate": mg.mean().item(),
                "min_head_gate": hg.min().item(),
                "max_head_gate": hg.max().item(),
            }
            log.append(entry)
            print(f"    step {step+1:4d}  Δ_a={entry['mean_delta_ans']:+.3f}  "
                  f"Δ_u={entry['mean_delta_unans']:+.3f}  "
                  f"<Gh>={entry['mean_head_gate']:.3f}  "
                  f"<Gm>={entry['mean_mlp_gate']:.3f}",
                  flush=True)

    gating.detach()
    return log


def compute_taxonomy(gp_h, gp_m, gm_h, gm_m,
                     active_head_mask=None, active_mlp_mask=None):
    """Derive causal taxonomy. Inactive components are zeroed out."""
    hc = gm_h.clone()
    ha = 1.0 - gp_h
    hi = gp_h * (1.0 - gm_h)
    mc = gm_m.clone()
    ma = 1.0 - gp_m
    mi = gp_m * (1.0 - gm_m)

    if active_head_mask is not None:
        inactive_h = ~active_head_mask
        hc[inactive_h] = 0.0
        ha[inactive_h] = 0.0
        hi[inactive_h] = 0.0
    if active_mlp_mask is not None:
        inactive_m = ~active_mlp_mask
        mc[inactive_m] = 0.0
        ma[inactive_m] = 0.0
        mi[inactive_m] = 0.0

    return {
        "head_commitment": hc.tolist(),
        "head_abstention": ha.tolist(),
        "head_irrelevant": hi.tolist(),
        "mlp_commitment": mc.tolist(),
        "mlp_abstention": ma.tolist(),
        "mlp_irrelevant": mi.tolist(),
    }


def main():
    args = parse_args()
    model_id = MODEL_SHORTCUTS[args.model]
    is_joint = args.dataset == "joint"

    # Optional pre-screening: restrict gating to top-k components from exp-2
    active_components = None
    if args.top_k:
        if is_joint:
            active_components = load_exp2_components_joint(
                args.exp2_dir, args.model, args.top_k)
        else:
            active_components = load_exp2_components(
                args.exp2_dir, args.model, args.dataset, args.top_k)

    mode = f"top-{args.top_k} pre-screened" if args.top_k else "all components"
    if is_joint:
        mode += f" (joint: {', '.join(JOINT_DATASETS)})"

    print(f"\n{'='*70}")
    print(f"CAC Localisation — Causal Component Gating")
    print(f"  Model  : {args.model}  ({model_id})")
    print(f"  Dataset: {args.dataset}")
    print(f"  Mode   : {mode}")
    print(f"  Seeds  : {args.seeds}   Steps: {args.steps}   "
          f"Batch: {args.batch_size}   λ: {args.lam or 'adaptive'}")
    print(f"{'='*70}\n")

    hf_token = os.getenv("HF_TOKEN")

    print("Loading dataset ...")
    if is_joint:
        data = load_joint_data(split="train", per_dataset=args.num_instances)
    else:
        data_all, _ = load_dataset(args.dataset)
        if args.num_instances:
            half = args.num_instances // 2
            ans = [x for x in data_all if x["is_answerable"]][:half]
            unans = [x for x in data_all if not x["is_answerable"]][:half]
            data = ans + unans
            random.seed(42)
            random.shuffle(data)
        else:
            data = data_all
    n_a = sum(1 for x in data if x["is_answerable"])
    print(f"  {len(data)} examples  ({n_a} answerable, {len(data)-n_a} unanswerable)")
    if is_joint:
        from collections import Counter
        ds_counts = Counter(x.get("source_dataset", "?") for x in data)
        print(f"  Per dataset: {dict(ds_counts)}")

    print(f"\nLoading model ...")
    model, tokenizer, device, vocab_size = load_model(model_id, hf_token)
    for p in model.parameters():
        p.requires_grad_(False)

    abstention_mask, commit_mask, _ = build_token_families(
        tokenizer, ABSTENTION_PHRASES, vocab_size, device="cpu",
    )

    print("\nPre-computing clean Δ(x) ...")
    items = precompute_clean(
        data, model, tokenizer, model_id, device, commit_mask, abstention_mask,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gating = CausalComponentGating(model, model_id).to(device)
    if active_components is not None:
        gating.set_active_components(active_components)
    print(f"  {gating.n_layers} layers × {gating.n_heads} heads  "
          f"({gating.n_active} active / {gating.n_components} total gates)")

    if args.lam is None:
        mean_abs_delta = sum(abs(ex["delta"]) for ex in items) / len(items)
        lam = mean_abs_delta / (args.batch_size * gating.n_active)
        print(f"\n  Adaptive λ: mean|Δ|={mean_abs_delta:.3f}, B={args.batch_size}, "
              f"K={gating.n_active}  →  λ = {lam:.6f}")
    else:
        lam = args.lam
        print(f"\n  Fixed λ = {lam}")

    for seed in range(args.seeds):
        out_path = RESULTS_DIR / f"{args.model}_{args.dataset}_seed{seed}.json"
        if out_path.exists():
            print(f"\n--- Seed {seed} — already done, skipping ---")
            continue

        print(f"\n{'─'*60}", flush=True)
        print(f"  Seed {seed}", flush=True)
        print(f"{'─'*60}", flush=True)
        random.seed(seed)
        torch.manual_seed(seed)

        result = {
            "model_id": model_id,
            "model_shortcut": args.model,
            "dataset_name": args.dataset,
            "seed": seed,
            "n_layers": gating.n_layers,
            "n_heads": gating.n_heads,
            "head_dim": gating.head_dim,
            "n_active": gating.n_active,
            "hyperparameters": {
                "lam": lam, "lam_mode": "adaptive" if args.lam is None else "fixed",
                "steps": args.steps, "batch_size": args.batch_size,
            },
            "status": "running",
            "training_log": {},
        }

        def _save():
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        # Phase 1: λ = 0  (shared initialisation)
        print("  Phase 1  (λ = 0)", flush=True)
        gating.randomize()
        log_init = fit_gates(
            gating, items, model, tokenizer, model_id, device,
            lam=0.0, n_steps=args.steps, batch_size=args.batch_size,
            lr_init=0.1, lr_final=0.01,
        )
        snap_init = gating.snapshot()
        result["training_log"]["init"] = log_init
        _save()
        print("    ✓ Phase 1 saved", flush=True)

        # Phase 2a: λ > 0  (G⁺, retention)
        print(f"  Phase 2a (λ = +{lam:.6f})", flush=True)
        gating.copy_from(snap_init)
        log_plus = fit_gates(
            gating, items, model, tokenizer, model_id, device,
            lam=lam, n_steps=args.steps, batch_size=args.batch_size,
            lr_init=0.5, lr_final=0.1,
        )
        gp_h = gating.head_gates.detach().cpu()
        gp_m = gating.mlp_gates.detach().cpu()
        result["training_log"]["plus"] = log_plus
        result["g_plus_head"] = gp_h.tolist()
        result["g_plus_mlp"] = gp_m.tolist()
        _save()
        print("    ✓ Phase 2a saved", flush=True)

        # Phase 2b: λ < 0  (G⁻, removal)
        print(f"  Phase 2b (λ = -{lam:.6f})", flush=True)
        gating.copy_from(snap_init)
        log_minus = fit_gates(
            gating, items, model, tokenizer, model_id, device,
            lam=-lam, n_steps=args.steps, batch_size=args.batch_size,
            lr_init=0.5, lr_final=0.1,
        )
        gm_h = gating.head_gates.detach().cpu()
        gm_m = gating.mlp_gates.detach().cpu()
        result["training_log"]["minus"] = log_minus
        result["g_minus_head"] = gm_h.tolist()
        result["g_minus_mlp"] = gm_m.tolist()

        taxonomy = compute_taxonomy(
            gp_h, gp_m, gm_h, gm_m,
            gating._active_head_mask, gating._active_mlp_mask,
        )
        result.update(taxonomy)
        result["status"] = "complete"
        _save()
        print(f"  ✓ Seed {seed} complete → {out_path}", flush=True)

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\n{'='*70}")
    print("All seeds complete.")
    print(f"  Next step: python cac-localisation/identify_cac.py --model {args.model}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
