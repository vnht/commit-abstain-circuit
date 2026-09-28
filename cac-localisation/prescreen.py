#!/usr/bin/env python3
"""CAC Localisation — Pre-screening: Per-head Δ(x) decomposition.

Runs a single forward pass per example with hooks to decompose Δ(x) into
per-head and per-MLP contributions via the output-projection linearity trick.
Components are ranked by |mean diff (answerable − unanswerable)| and the
top-50 are saved as a summary JSON consumed by run.py --top-k.

Run this before run.py if you want to restrict gate fitting to the most
attribution-relevant components (recommended for large models).

Output: results/{model}_{dataset}_summary.json

Usage:
    python cac-localisation/prescreen.py --model gemma-4b --dataset joint
    python cac-localisation/prescreen.py --model gemma-4b --dataset kuq
"""

import os
import sys
import json
import time
import argparse
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

RESULTS_DIR = Path(__file__).parent / "results"
JOINT_DATASETS = ["kuq", "musique", "squad"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Pre-screening: per-head Δ(x) decomposition and component ranking"
    )
    ap.add_argument("--model", required=True, choices=list(MODEL_SHORTCUTS.keys()))
    ap.add_argument("--dataset", required=True,
                    choices=list(DATASET_CONFIGS.keys()) + ["joint"])
    ap.add_argument("-n", "--num-instances", type=int, default=None,
                    help="Limit examples per dataset (default: all)")
    ap.add_argument("--top-k-save", type=int, default=50,
                    help="Number of top components to save in summary (default: 50)")
    return ap.parse_args()


# ── activation collector ──────────────────────────────────────────────

class ActivationCollector:
    """Register hooks on all layers; decompose Δ(x) per head and per MLP.

    The per-head decomposition is exact when the output projection has no bias:
    each head's contribution to the residual stream is isolated by slicing the
    concatenated pre-o_proj activations and projecting through the corresponding
    columns of W_O.
    """

    def __init__(self, model, model_id):
        self.model = model
        self.model_id = model_id
        self._hooks = []
        self.raw = {}

        self.layers = self._get_layers()
        self.lm_head = self._get_lm_head()
        self.final_norm = self._get_final_norm()
        self.n_layers = len(self.layers)
        self.n_heads, self.head_dim = self._get_head_info()

    def _get_layers(self):
        m = self.model
        for path in [
            lambda: m.model.language_model.layers,
            lambda: m.model.layers,
        ]:
            try:
                layers = path()
                if hasattr(layers, "__len__") and len(layers) > 0:
                    return layers
            except AttributeError:
                continue
        raise RuntimeError(f"Cannot find decoder layers for {self.model_id}")

    def _get_lm_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        raise RuntimeError(f"Cannot find lm_head for {self.model_id}")

    def _get_final_norm(self):
        m = self.model
        for path in [
            lambda: m.model.language_model.norm,
            lambda: m.model.norm,
        ]:
            try:
                norm = path()
                if hasattr(norm, "weight"):
                    return norm
            except AttributeError:
                continue
        raise RuntimeError(f"Cannot find final norm for {self.model_id}")

    @staticmethod
    def _get_attn_module(layer):
        for name in ("self_attn", "linear_attn", "attention"):
            if hasattr(layer, name):
                return getattr(layer, name)
        raise RuntimeError(
            f"Cannot find attention module in {type(layer).__name__}"
        )

    @staticmethod
    def _get_o_proj(attn):
        for name in ("o_proj", "out_proj"):
            if hasattr(attn, name):
                return getattr(attn, name)
        raise RuntimeError(
            f"Cannot find output projection in {type(attn).__name__}"
        )

    def _get_head_info(self):
        attn = self._get_attn_module(self.layers[0])
        o_proj = self._get_o_proj(attn)
        inner_dim = o_proj.weight.shape[1]

        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            cfg = self.model.config
            head_dim = getattr(cfg, "head_dim", None)
            if head_dim is None and hasattr(cfg, "text_config"):
                head_dim = getattr(cfg.text_config, "head_dim", None)
        if head_dim is None and hasattr(attn, "in_proj_a"):
            n = attn.in_proj_a.weight.shape[0]
            head_dim = inner_dim // n
        if head_dim is None:
            raise RuntimeError(f"Cannot determine head_dim for {self.model_id}")

        return inner_dim // head_dim, head_dim

    def attach(self):
        self.raw = {}
        self._hooks = []
        for i, layer in enumerate(self.layers):
            attn_mod = self._get_attn_module(layer)
            o_proj = self._get_o_proj(attn_mod)
            self._hooks.append(
                o_proj.register_forward_hook(self._oproj_hook(i))
            )
            self._hooks.append(
                layer.mlp.register_forward_hook(self._fwd_hook(f"mlp_{i}"))
            )
        self._hooks.append(
            self.layers[-1].register_forward_hook(self._fwd_hook("final_res"))
        )

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.raw = {}

    def _fwd_hook(self, key):
        storage = self.raw
        def hook(_module, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            storage[key] = x[:, -1, :].detach().float().cpu()
        return hook

    def _oproj_hook(self, layer_idx):
        storage = self.raw
        def hook(_module, inp, _out):
            storage[f"oproj_in_{layer_idx}"] = (
                inp[0][:, -1, :].detach().float().cpu()
            )
        return hook

    def decompose(self, logits, commit_mask, abstain_mask):
        """Project each head/MLP contribution onto d(x) = W_lm[t*] − W_lm[b*]."""
        logits_f = logits.float()
        cl = logits_f.clone(); cl[~commit_mask] = float("-inf")
        t_star = cl.argmax().item()
        al = logits_f.clone(); al[~abstain_mask] = float("-inf")
        b_star = al.argmax().item()
        delta = logits_f[t_star].item() - logits_f[b_star].item()

        W = self.lm_head.weight.detach().float().cpu()
        d_raw = W[t_star] - W[b_star]
        gamma = self.final_norm.weight.detach().float().cpu()
        d_scaled = d_raw * gamma

        final_res = self.raw["final_res"].squeeze(0)
        eps = getattr(self.final_norm, "variance_epsilon",
                      getattr(self.final_norm, "eps", 1e-6))
        rms = (final_res.pow(2).mean() + eps).sqrt()
        d_norm = d_scaled / rms

        device = self.lm_head.weight.device
        d_norm_dev = d_norm.to(device)

        head_v, attn_v, mlp_v = [], [], []
        for i in range(self.n_layers):
            attn_mod = self._get_attn_module(self.layers[i])
            o_proj = self._get_o_proj(attn_mod)
            d_proj = (o_proj.weight.float().T @ d_norm_dev).cpu()
            oproj_input = self.raw[f"oproj_in_{i}"].squeeze(0)
            d_heads = d_proj.reshape(self.n_heads, self.head_dim)
            o_heads = oproj_input.reshape(self.n_heads, self.head_dim)
            hv = (d_heads * o_heads).sum(dim=1)
            head_v.append(hv.tolist())
            bias_v = 0.0
            if o_proj.bias is not None:
                bias_v = (d_norm @ o_proj.bias.detach().float().cpu()).item()
            attn_v.append(hv.sum().item() + bias_v)
            mv = (d_norm @ self.raw[f"mlp_{i}"].squeeze(0)).item()
            mlp_v.append(mv)

        layer_sum = sum(attn_v) + sum(mlp_v)
        embedding_v = delta - layer_sum
        recon_error = abs(delta - (embedding_v + layer_sum)) / (abs(delta) + 1e-8)

        return {
            "delta": delta, "t_star": t_star, "b_star": b_star,
            "head_v": head_v, "attn_v": attn_v, "mlp_v": mlp_v,
            "embedding_v": embedding_v, "reconstruction_error": recon_error,
            "n_layers": self.n_layers, "n_heads": self.n_heads,
        }


# ── data loading ──────────────────────────────────────────────────────

def load_data(dataset_arg, num_instances=None):
    if dataset_arg == "joint":
        all_data = []
        for ds in JOINT_DATASETS:
            data_ds, _ = load_dataset(ds)
            ans = sorted([x for x in data_ds if x["is_answerable"]],
                         key=lambda x: x["question_id"])
            unans = sorted([x for x in data_ds if not x["is_answerable"]],
                           key=lambda x: x["question_id"])
            half = min(250, num_instances // 2) if num_instances else 250
            selected = ans[:half] + unans[:half]
            for x in selected:
                x["source_dataset"] = ds
            all_data.extend(selected)
        return all_data, "joint"
    else:
        data, dataset_type = load_dataset(dataset_arg)
        if num_instances:
            data = data[:num_instances]
        return data, dataset_type


# ── contribution collection ───────────────────────────────────────────

def collect_contributions(data, model, tokenizer, model_id, device,
                           commit_mask, abstain_mask, jsonl_path):
    """Forward-pass each example with hooks; append results to JSONL."""
    done = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if "head_v" in r:
                        done[r["question_id"]] = r
        if done:
            print(f"  Resuming: {len(done)}/{len(data)} already done")

    collector = ActivationCollector(model, model_id)
    print(f"  {collector.n_layers} layers × {collector.n_heads} heads/layer")

    results = []
    n = len(data)
    t0 = time.time()

    with open(jsonl_path, "a") as fout:
        for i, item in enumerate(data):
            qid = item["question_id"]
            if qid in done:
                results.append(done[qid])
                continue

            inputs = prepare_inputs(model_id, tokenizer, item["prompt"], device)
            fwd_kwargs = {"input_ids": inputs["input_ids"]}
            if inputs.get("attention_mask") is not None:
                fwd_kwargs["attention_mask"] = inputs["attention_mask"]

            collector.attach()
            with torch.no_grad():
                out = model(**fwd_kwargs)
            logits = out.logits[0, -1, :].float().cpu()
            decomp = collector.decompose(logits, commit_mask, abstain_mask)
            collector.detach()

            result = {
                "question_id": qid,
                "is_answerable": item["is_answerable"],
                "delta": decomp["delta"],
                "t_star": decomp["t_star"],
                "b_star": decomp["b_star"],
                "n_layers": decomp["n_layers"],
                "n_heads": decomp["n_heads"],
                "head_v": decomp["head_v"],
                "attn_v": decomp["attn_v"],
                "mlp_v": decomp["mlp_v"],
                "reconstruction_error": decomp["reconstruction_error"],
            }
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            results.append(result)

            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            tag = "Y" if item["is_answerable"] else "N"
            print(f"  {i+1}/{n}  Δ={decomp['delta']:+.3f}  ans={tag}  "
                  f"recon={decomp['reconstruction_error']:.4f}  "
                  f"ETA {eta:.0f}s", flush=True)

    return results


# ── ranking ───────────────────────────────────────────────────────────

def rank_components(results, top_k_save):
    """Rank heads and MLPs by |mean contribution diff (ans − unans)|."""
    n_layers = results[0]["n_layers"]
    n_heads = results[0]["n_heads"]

    ans_head = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    unans_head = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
    ans_mlp = [[] for _ in range(n_layers)]
    unans_mlp = [[] for _ in range(n_layers)]

    for r in results:
        for l in range(n_layers):
            if r["is_answerable"]:
                ans_mlp[l].append(r["mlp_v"][l])
                for h in range(n_heads):
                    ans_head[l][h].append(r["head_v"][l][h])
            else:
                unans_mlp[l].append(r["mlp_v"][l])
                for h in range(n_heads):
                    unans_head[l][h].append(r["head_v"][l][h])

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    components = []
    for l in range(n_layers):
        for h in range(n_heads):
            diff = mean(ans_head[l][h]) - mean(unans_head[l][h])
            components.append({
                "layer": l, "type": "head", "head": h,
                "mean_diff": diff, "abs_diff": abs(diff),
            })
        mlp_diff = mean(ans_mlp[l]) - mean(unans_mlp[l])
        components.append({
            "layer": l, "type": "mlp",
            "mean_diff": mlp_diff, "abs_diff": abs(mlp_diff),
        })

    components.sort(key=lambda c: c["abs_diff"], reverse=True)
    for rank, c in enumerate(components):
        c["rank"] = rank + 1

    return components[:top_k_save]


# ── main ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model_id = MODEL_SHORTCUTS[args.model]

    print(f"\n{'='*70}")
    print(f"CAC Pre-screening — Per-head Δ(x) decomposition")
    print(f"  Model  : {args.model}  ({model_id})")
    print(f"  Dataset: {args.dataset}")
    print(f"{'='*70}\n")

    hf_token = os.getenv("HF_TOKEN")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = JOINT_DATASETS if args.dataset == "joint" else [args.dataset]

    print("Loading model ...")
    model, tokenizer, device, vocab_size = load_model(model_id, hf_token)
    for p in model.parameters():
        p.requires_grad_(False)

    abstention_mask, commit_mask, _ = build_token_families(
        tokenizer, ABSTENTION_PHRASES, vocab_size, device="cpu",
    )

    for ds_name in datasets:
        print(f"\n{'─'*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'─'*60}")

        summary_path = RESULTS_DIR / f"{args.model}_{ds_name}_summary.json"
        if summary_path.exists():
            print(f"  Already done → {summary_path.name}, skipping")
            continue

        data, _ = load_dataset(ds_name)
        if args.num_instances:
            data = data[:args.num_instances]
        n_a = sum(1 for x in data if x["is_answerable"])
        print(f"  {len(data)} examples ({n_a} answerable, {len(data)-n_a} unanswerable)")

        jsonl_path = RESULTS_DIR / f"{args.model}_{ds_name}_contributions.jsonl"
        results = collect_contributions(
            data, model, tokenizer, model_id, device,
            commit_mask, abstain_mask=abstention_mask, jsonl_path=jsonl_path,
        )

        avg_recon = sum(r["reconstruction_error"] for r in results) / len(results)
        print(f"\n  Mean reconstruction error: {avg_recon:.6f}")

        print(f"  Ranking components (top {args.top_k_save}) ...")
        ranked = rank_components(results, args.top_k_save)
        print(f"  Top 10:")
        for c in ranked[:10]:
            label = (f"L{c['layer']:2d} H{c['head']}" if c["type"] == "head"
                     else f"L{c['layer']:2d} MLP")
            print(f"    #{c['rank']:3d}  {label:<12}  diff={c['mean_diff']:+.4f}")

        summary = {
            "model_id": model_id,
            "model_shortcut": args.model,
            "dataset_name": ds_name,
            "n_examples": len(results),
            "n_layers": results[0]["n_layers"],
            "n_heads": results[0]["n_heads"],
            "mean_reconstruction_error": avg_recon,
            "ranked_components": ranked,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved → {summary_path.name}")

    print(f"\n{'='*70}")
    print(f"Pre-screening complete.")
    print(f"  Next: python cac-localisation/run.py --model {args.model} "
          f"--dataset {args.dataset} --top-k 50")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
