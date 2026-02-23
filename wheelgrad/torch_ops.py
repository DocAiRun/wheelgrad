"""
wheelgrad/torch_ops.py
──────────────────────
Real PyTorch integration for WheelGrad.

Drop-in replacements for torch.nn.functional and torch.nn layers.
Uses custom autograd.Function to:
  1. Run Wheel arithmetic in forward pass (numpy bridge)
  2. Zero gradients at ⊥ positions in backward pass
  3. Preserve standard gradient flow for finite outputs

Usage:
    from wheelgrad.torch_ops import (
        WheelSoftmax, WheelLayerNorm, WheelMultiheadAttention,
        wheel_softmax_fn, replace_layers
    )
    
    # Option A: direct replacement
    model.attn.softmax = WheelSoftmax(dim=-1)
    
    # Option B: auto-replace all softmax/layernorm in a model
    model = replace_layers(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from .ops import wheel_softmax, wheel_layernorm
from .core import WheelTensor


# ── Core autograd Functions ──────────────────────────────────────────────────

class WheelSoftmaxFunction(torch.autograd.Function):
    """
    Custom autograd Function for Wheel softmax.
    
    Forward:  Wheel arithmetic (total division, ⊥ instead of NaN)
    Backward: Standard softmax Jacobian, zeroed at ⊥ positions
    
    The zero-gradient rule at ⊥ is the algebraically correct choice:
    ⊥ is an absorbing element — no information should flow back through it.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Run Wheel softmax via numpy bridge
        x_np = logits.detach().cpu().numpy().astype(np.float64)
        wheel_out = wheel_softmax(x_np, axis=dim)

        # Store for backward
        probs_np = wheel_out.to_numpy_safe(bot_fill=0.0, inf_fill=0.0)
        probs_t  = torch.from_numpy(probs_np.astype(np.float32)).to(logits.device)
        bot_mask = torch.from_numpy(wheel_out.bot_mask).to(logits.device)

        ctx.save_for_backward(probs_t)
        ctx.bot_mask = bot_mask
        ctx.dim = dim

        return probs_t

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        probs, = ctx.saved_tensors
        bot_mask = ctx.bot_mask
        dim = ctx.dim

        # Standard softmax Jacobian-vector product
        # ∂L/∂x_i = p_i * (∂L/∂p_i - Σ_j p_j * ∂L/∂p_j)
        dot = (grad_output * probs).sum(dim=dim, keepdim=True)
        grad = probs * (grad_output - dot)

        # KEY: zero gradient at ⊥ positions
        # Absorbing element → no gradient signal through undefined outputs
        grad[bot_mask] = 0.0

        return grad, None


class WheelLayerNormFunction(torch.autograd.Function):
    """
    Custom autograd Function for Wheel LayerNorm.
    
    Key difference from standard: when std=0 (all features identical),
    standard LayerNorm returns tiny biased values (due to eps).
    WheelGrad returns ⊥ — and zeroes the gradient there.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x_np = x.detach().cpu().numpy().astype(np.float64)
        g_np = weight.detach().cpu().numpy() if weight is not None else None
        b_np = bias.detach().cpu().numpy()   if bias   is not None else None

        wheel_out = wheel_layernorm(x_np, g_np, b_np)

        out_np = wheel_out.to_numpy_safe(bot_fill=0.0)
        out_t  = torch.from_numpy(out_np.astype(np.float32)).to(x.device)
        bot_t  = torch.from_numpy(wheel_out.bot_mask).to(x.device)

        # For backward: store normalized x (without gamma/beta)
        norm_np = wheel_layernorm(x_np).to_numpy_safe(bot_fill=0.0)
        norm_t  = torch.from_numpy(norm_np.astype(np.float32)).to(x.device)

        ctx.save_for_backward(norm_t, weight, bias)
        ctx.bot_mask = bot_t
        ctx.has_weight = weight is not None
        ctx.has_bias   = bias is not None

        return out_t

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_norm, weight, bias = ctx.saved_tensors
        bot_mask = ctx.bot_mask

        # Zero grad at ⊥ positions
        grad_output = grad_output.clone()
        grad_output[bot_mask] = 0.0

        N = x_norm.shape[-1]

        # Grad w.r.t. input (standard LayerNorm backward)
        if weight is not None:
            dy = grad_output * weight
        else:
            dy = grad_output

        mean_dy       = dy.mean(dim=-1, keepdim=True)
        mean_dy_xnorm = (dy * x_norm).mean(dim=-1, keepdim=True)
        grad_x = (dy - mean_dy - x_norm * mean_dy_xnorm) / N

        # Grad w.r.t. gamma and beta
        grad_weight = (grad_output * x_norm).sum(dim=0) if ctx.has_weight else None
        grad_bias   = grad_output.sum(dim=0)             if ctx.has_bias   else None

        return grad_x, grad_weight, grad_bias


# ── nn.Module wrappers ───────────────────────────────────────────────────────

class WheelSoftmax(nn.Module):
    """
    Drop-in replacement for nn.Softmax / F.softmax.
    
    Usage:
        # Replace
        self.softmax = WheelSoftmax(dim=-1)
        
        # In forward
        out = self.softmax(logits)  # identical API
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return WheelSoftmaxFunction.apply(x, self.dim)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, arithmetic=wheel'


class WheelLayerNorm(nn.Module):
    """
    Drop-in replacement for nn.LayerNorm.
    
    Identical API — just swap the class name.
    Handles std=0 algebraically (⊥) instead of with epsilon bias.
    
    Usage:
        # Before:
        self.norm = nn.LayerNorm(d_model)
        # After:
        self.norm = WheelLayerNorm(d_model)
    """
    def __init__(
        self,
        normalized_shape,
        elementwise_affine: bool = True,
        device=None,
        dtype=None
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
            self.bias   = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype))
        else:
            self.weight = self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return WheelLayerNormFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, elementwise_affine={self.elementwise_affine}, arithmetic=wheel'


class WheelMultiheadAttention(nn.Module):
    """
    Multi-head attention with Wheel softmax.
    
    Replaces the standard softmax in scaled dot-product attention.
    All other operations (QKV projection, output projection) remain standard.
    Only the softmax — where 0/0 occurs on masked tokens — uses Wheel.
    
    Usage:
        # Before:
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        # After:
        self.attn = WheelMultiheadAttention(d_model, n_heads)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_heads  = num_heads
        self.head_dim   = embed_dim // num_heads
        self.scale      = self.head_dim ** -0.5
        self.batch_first = batch_first

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.wheel_softmax = WheelSoftmax(dim=-1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = query.shape

        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Reshape to (B, heads, T, head_dim)
        def reshape(x):
            return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = reshape(Q), reshape(K), reshape(V)

        # Scaled dot-product scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply masks BEFORE Wheel softmax
        if attn_mask is not None:
            # Standard additive mask (e.g. causal): -inf for masked positions
            # Wheel will convert -inf → ∞ → and handle competing ∞ as ⊥
            scores = scores + attn_mask

        if key_padding_mask is not None:
            # key_padding_mask: (B, T) True = ignore
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        # WHEEL SOFTMAX — the key replacement
        B_, H, T1, T2 = scores.shape
        scores_flat = scores.reshape(B_ * H, T1, T2)
        weights_flat = torch.stack([
            self.wheel_softmax(scores_flat[i]) for i in range(B_ * H)
        ])
        weights = weights_flat.reshape(B_, H, T1, T2)

        weights = self.dropout(weights)

        # Attend to values
        out = torch.matmul(weights, V)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, weights


# ── Model surgery utility ────────────────────────────────────────────────────

def replace_layers(
    model: nn.Module,
    replace_softmax: bool = True,
    replace_layernorm: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Automatically replace nn.Softmax and nn.LayerNorm with Wheel versions.
    
    Walks the model recursively and replaces in-place.
    Works on any PyTorch model — transformers, CNNs, custom architectures.
    
    Parameters
    ----------
    model : nn.Module
        Any PyTorch model
    replace_softmax : bool
        Replace nn.Softmax layers
    replace_layernorm : bool
        Replace nn.LayerNorm layers
    verbose : bool
        Print what was replaced
    
    Returns
    -------
    nn.Module
        Same model with Wheel layers (modified in-place + returned)
    
    Example
    -------
    >>> model = TransformerModel(...)
    >>> model = replace_layers(model, verbose=True)
    Replaced: encoder.layer.0.attention.softmax  →  WheelSoftmax(dim=-1)
    Replaced: encoder.layer.0.norm1              →  WheelLayerNorm((768,))
    ...
    """
    replaced = []

    def _replace(parent: nn.Module, prefix: str = ''):
        for name, child in list(parent.named_children()):
            full_name = f'{prefix}.{name}' if prefix else name

            if replace_softmax and isinstance(child, nn.Softmax):
                new_layer = WheelSoftmax(dim=child.dim)
                setattr(parent, name, new_layer)
                replaced.append((full_name, 'WheelSoftmax'))

            elif replace_layernorm and isinstance(child, nn.LayerNorm):
                new_layer = WheelLayerNorm(
                    child.normalized_shape,
                    elementwise_affine=child.elementwise_affine,
                )
                if child.elementwise_affine:
                    new_layer.weight = nn.Parameter(child.weight.data.clone())
                    new_layer.bias   = nn.Parameter(child.bias.data.clone())
                setattr(parent, name, new_layer)
                replaced.append((full_name, 'WheelLayerNorm'))

            else:
                _replace(child, full_name)

    _replace(model)

    if verbose:
        if replaced:
            print(f"WheelGrad: replaced {len(replaced)} layer(s):")
            for name, kind in replaced:
                print(f"  {name}  →  {kind}")
        else:
            print("WheelGrad: no nn.Softmax or nn.LayerNorm found to replace.")
            print("  For MultiheadAttention, use WheelMultiheadAttention directly.")

    return model


def wheel_loss_report(model: nn.Module) -> dict:
    """
    Scan all parameters for NaN/Inf after a training step.
    Returns a dict of {param_name: status}.
    """
    report = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            if has_nan or has_inf:
                report[name] = {
                    'nan_count': torch.isnan(param.grad).sum().item(),
                    'inf_count': torch.isinf(param.grad).sum().item(),
                    'status': 'CORRUPTED'
                }
            else:
                report[name] = {'status': 'OK'}
    return report
