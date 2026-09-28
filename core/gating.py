"""Causal Component Gating for the commit-abstain circuit.

Differentiable soft gates over attention heads and MLP sublayers, adapted
from Causal Head Gating (Nam et al., 2025).  Extended to gate both heads
and MLPs — justified by the finding that MLPs often carry larger causal
effects than individual heads.

Gate parameterisation
    G_{l,h} = σ(g_{l,h})   where g ∈ ℝ is a learnable logit.
    The gate multiplies the head's pre-o_proj output (heads) or the MLP
    sublayer output (MLPs), scaling their contribution to the residual.

Causal taxonomy
    Commitment-promoting  = G⁻           (resists removal → supports correct Δ)
    Abstention-promoting  = 1 − G⁺       (suppressed under retention pressure)
    Decision-irrelevant   = G⁺ × (1−G⁻)  (follows regularisation either way)

Regularisation (L1 on logit-space gates, clipped at ±CLIP)
    R(G, λ) = −λ Σ clip(g, −C, C)
    λ > 0 → push gates toward 1 (retention, produces G⁺)
    λ < 0 → push gates toward 0 (removal, produces G⁻)
"""

import torch
import torch.nn as nn

CLIP = 4.0


# ── architecture helpers ──────────────────────────────────────────────

def _get_layers(model):
    for accessor in [
        lambda: model.model.language_model.layers,
        lambda: model.model.layers,
    ]:
        try:
            layers = accessor()
            if hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise RuntimeError("Cannot locate decoder layers")


def _get_attn_module(layer):
    for name in ("self_attn", "linear_attn", "attention"):
        if hasattr(layer, name):
            return getattr(layer, name)
    raise RuntimeError(f"Cannot locate attention in {type(layer).__name__}")


def _get_o_proj(attn):
    for name in ("o_proj", "out_proj"):
        if hasattr(attn, name):
            return getattr(attn, name)
    raise RuntimeError(f"Cannot locate output projection in {type(attn).__name__}")


def _get_head_info(layers, model):
    attn = _get_attn_module(layers[0])
    o_proj = _get_o_proj(attn)
    inner_dim = o_proj.weight.shape[1]

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None:
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None and hasattr(cfg, "text_config"):
            head_dim = getattr(cfg.text_config, "head_dim", None)
    if head_dim is None and hasattr(attn, "in_proj_a"):
        n = attn.in_proj_a.weight.shape[0]
        head_dim = inner_dim // n
    if head_dim is None:
        raise RuntimeError("Cannot determine head_dim")

    return inner_dim // head_dim, head_dim


# ── main module ───────────────────────────────────────────────────────

class CausalComponentGating(nn.Module):
    """Learnable soft gates for every attention head and MLP sublayer."""

    def __init__(self, model, model_id):
        super().__init__()
        self._model = model
        self.model_id = model_id
        self._hooks: list = []

        self.layers = _get_layers(model)
        self.n_layers = len(self.layers)
        self.n_heads, self.head_dim = _get_head_info(self.layers, model)

        self.head_logits = nn.Parameter(
            torch.zeros(self.n_layers, self.n_heads)
        )
        self.mlp_logits = nn.Parameter(torch.zeros(self.n_layers))

        self.register_buffer("_active_head_mask", None)
        self.register_buffer("_active_mlp_mask", None)

        self._head_scales = {}  # (layer, head) -> float
        self._mlp_scales = {}   # layer -> float

    def set_active_components(self, components):
        """Restrict optimisation to *components*; all others passthrough."""
        hm = torch.zeros(self.n_layers, self.n_heads, dtype=torch.bool)
        mm = torch.zeros(self.n_layers, dtype=torch.bool)
        for c in components:
            if c["type"] == "head":
                hm[c["layer"], c["head"]] = True
            else:
                mm[c["layer"]] = True
        dev = self.head_logits.device
        self._active_head_mask = hm.to(dev)
        self._active_mlp_mask = mm.to(dev)
        self.head_logits.data.fill_(10.0)
        self.mlp_logits.data.fill_(10.0)

    @property
    def head_gates(self):
        return torch.sigmoid(self.head_logits)

    @property
    def mlp_gates(self):
        return torch.sigmoid(self.mlp_logits)

    @property
    def n_components(self):
        return self.n_layers * self.n_heads + self.n_layers

    @property
    def n_active(self):
        if self._active_head_mask is None:
            return self.n_components
        return int(self._active_head_mask.sum() + self._active_mlp_mask.sum())

    def randomize(self):
        """Uniform-in-[0,1] gate initialisation (logit-space)."""
        if self._active_head_mask is None:
            u = torch.rand_like(self.head_logits).clamp(0.02, 0.98)
            self.head_logits.data.copy_(torch.log(u / (1 - u)))
            u = torch.rand_like(self.mlp_logits).clamp(0.02, 0.98)
            self.mlp_logits.data.copy_(torch.log(u / (1 - u)))
        else:
            dev = self.head_logits.device
            self.head_logits.data.fill_(10.0)
            self.mlp_logits.data.fill_(10.0)
            nh = self._active_head_mask.sum().item()
            nm = self._active_mlp_mask.sum().item()
            if nh:
                u = torch.rand(nh, device=dev).clamp(0.02, 0.98)
                self.head_logits.data[self._active_head_mask] = torch.log(u / (1 - u))
            if nm:
                u = torch.rand(nm, device=dev).clamp(0.02, 0.98)
                self.mlp_logits.data[self._active_mlp_mask] = torch.log(u / (1 - u))

    def copy_from(self, src):
        """Copy gate logits from *src* (another instance or a state dict)."""
        if isinstance(src, CausalComponentGating):
            self.head_logits.data.copy_(src.head_logits.data)
            self.mlp_logits.data.copy_(src.mlp_logits.data)
        else:
            self.head_logits.data.copy_(src["head_logits"])
            self.mlp_logits.data.copy_(src["mlp_logits"])

    def snapshot(self):
        """Return a detached CPU copy of current gate logits."""
        return {
            "head_logits": self.head_logits.data.detach().cpu().clone(),
            "mlp_logits": self.mlp_logits.data.detach().cpu().clone(),
        }

    def attach(self):
        """Install scaling hooks."""
        self.detach()
        for i, layer in enumerate(self.layers):
            has_head = (self._active_head_mask is None
                        or self._active_head_mask[i].any().item())
            has_mlp = (self._active_mlp_mask is None
                       or self._active_mlp_mask[i].item())
            if not has_head and not has_mlp:
                continue
            if has_head:
                attn = _get_attn_module(layer)
                o_proj = _get_o_proj(attn)
                self._hooks.append(
                    o_proj.register_forward_pre_hook(self._head_prehook(i))
                )
            if has_mlp:
                self._hooks.append(
                    layer.mlp.register_forward_hook(self._mlp_hook(i))
                )

    def detach(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _head_prehook(self, layer_idx):
        hd = self.head_dim
        mask = self._active_head_mask

        def hook(_module, args):
            x = args[0]
            g = torch.sigmoid(
                self.head_logits[layer_idx].to(device=x.device)
            )
            if mask is not None:
                m = mask[layer_idx].to(device=x.device)
                g = torch.where(m, g, torch.ones_like(g))
            for h_idx in range(g.shape[0]):
                key = (layer_idx, h_idx)
                if key in self._head_scales:
                    g[h_idx] = self._head_scales[key]
            return (x * g.to(x.dtype).repeat_interleave(hd),)

        return hook

    def _mlp_hook(self, layer_idx):
        active = (self._active_mlp_mask is None
                  or self._active_mlp_mask[layer_idx].item())

        def hook(_module, _inp, out):
            if not active:
                return out
            if layer_idx in self._mlp_scales:
                g = self._mlp_scales[layer_idx]
            else:
                g = torch.sigmoid(self.mlp_logits[layer_idx])
            if isinstance(out, tuple):
                o = out[0]
                return (o * g,) + out[1:]
            return out * g

        return hook

    def regularization(self, lam):
        """−λ Σ clip(g, −C, C) over active gates only."""
        if self._active_head_mask is None:
            all_g = torch.cat([self.head_logits.reshape(-1), self.mlp_logits])
        else:
            parts = []
            if self._active_head_mask.any():
                parts.append(self.head_logits[self._active_head_mask])
            if self._active_mlp_mask.any():
                parts.append(self.mlp_logits[self._active_mlp_mask])
            if not parts:
                return torch.tensor(0.0, requires_grad=True)
            all_g = torch.cat(parts)
        return -lam * all_g.clamp(-CLIP, CLIP).sum()

    def set_gates_from_values(self, head_vals, mlp_vals):
        """Set logits so that σ(logit) ≈ *vals* (clamped away from 0/1)."""
        hv = torch.as_tensor(head_vals, dtype=torch.float32).clamp(0.001, 0.999)
        mv = torch.as_tensor(mlp_vals, dtype=torch.float32).clamp(0.001, 0.999)
        self.head_logits.data.copy_(torch.log(hv / (1 - hv)))
        self.mlp_logits.data.copy_(torch.log(mv / (1 - mv)))

    def set_all_open(self):
        """All gates ≈ 1 (passthrough), clear scale overrides."""
        self.head_logits.data.fill_(10.0)
        self.mlp_logits.data.fill_(10.0)
        self._head_scales.clear()
        self._mlp_scales.clear()

    def suppress_component(self, layer, comp_type, head=None):
        """Set one component's gate to ≈ 0."""
        if comp_type == "head":
            self.head_logits.data[layer, head] = -10.0
        else:
            self.mlp_logits.data[layer] = -10.0

    def scale_component(self, layer, comp_type, head=None, scale=1.0):
        """Set one component's effective multiplier (can be >1)."""
        if comp_type == "head":
            self._head_scales[(layer, head)] = scale
        else:
            self._mlp_scales[layer] = scale

    def clear_scales(self):
        """Remove all direct scale overrides."""
        self._head_scales.clear()
        self._mlp_scales.clear()
