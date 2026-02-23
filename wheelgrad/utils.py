"""
wheelgrad/utils.py
──────────────────
Utilities, PyTorch bridge, and diagnostic tools.
"""

import numpy as np
from .core import WheelTensor


def encode_wheel(values: np.ndarray, bot_mask: np.ndarray = None):
    """
    Encode a numpy array as a Wheel tensor pair ready for GPU.
    
    PyTorch usage (when available):
        import torch
        vals_t = torch.from_numpy(values).float()
        mask_t = torch.from_numpy(bot_mask).bool()
        # These two tensors encode the full Wheel state
        # Pass both through the computation graph
    """
    return WheelTensor(values, bot_mask)


def decode_wheel(wt: WheelTensor, nan_for_bot: bool = True) -> np.ndarray:
    """Convert WheelTensor back to numpy. ⊥ → NaN or configurable fill."""
    if nan_for_bot:
        return wt.to_numpy()
    return wt.to_numpy_safe()


def wheel_status_report(wt: WheelTensor) -> str:
    """Generate a diagnostic report for a WheelTensor."""
    n = wt.values.size
    nb = wt.bot_count()
    ni = wt.inf_count()
    nf = n - nb - ni

    report = [
        "═" * 50,
        "  WheelGrad Status Report",
        "═" * 50,
        f"  Shape        : {wt.shape}",
        f"  Total elems  : {n}",
        f"  Finite (✓)   : {nf} ({100*nf/n:.1f}%)",
        f"  Infinity (∞) : {ni} ({100*ni/n:.1f}%)",
        f"  Bottom (⊥)   : {nb} ({100*nb/n:.1f}%)",
        "─" * 50,
    ]
    if nb > 0:
        report.append("  ⚠  ⊥ detected — algebraically defined indeterminate")
        report.append("     (would have been silent NaN in standard arithmetic)")
    if ni > 0:
        report.append("  ⚠  ∞ detected — projective infinity, check input scale")
    if nb == 0 and ni == 0:
        report.append("  ✓  Tensor is clean — no special elements")
    report.append("═" * 50)
    return "\n".join(report)


# ── PyTorch bridge (graceful degradation) ────────────────────────────────────

def try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


class WheelFunction:
    """
    PyTorch autograd.Function wrapper for Wheel operations.
    
    When PyTorch is available:
        - Registers custom forward/backward for Wheel softmax
        - Gradient of ⊥ outputs = 0 (absorbing → no gradient)
        - Gradient of ∞ outputs = 0 (projective point → stop)
    
    When PyTorch is not available:
        - Falls back to numpy implementation
    
    Usage (PyTorch):
        class WheelSoftmaxFn(WheelFunction):
            @staticmethod
            def forward(ctx, x):
                result = wheel_softmax(x.numpy())
                ctx.save_for_backward(torch.from_numpy(result.values))
                ctx.bot_mask = torch.from_numpy(result.bot_mask)
                return torch.from_numpy(result.to_numpy_safe())
            
            @staticmethod
            def backward(ctx, grad_output):
                probs, = ctx.saved_tensors
                bot_mask = ctx.bot_mask
                # Standard softmax gradient
                s = probs
                grad = s * (grad_output - (grad_output * s).sum(dim=-1, keepdim=True))
                # Zero out gradients for ⊥ positions (absorbing element)
                grad[bot_mask] = 0.0
                return grad
    """
    pass


TORCH_INTEGRATION_TEMPLATE = '''
"""
wheelgrad/torch_ops.py  (requires PyTorch)
──────────────────────────────────────────
Drop-in replacements for torch.nn.functional operations.

Install: pip install torch
Usage:
    from wheelgrad.torch_ops import WheelSoftmax, WheelLayerNorm
    
    # Replace nn.functional.softmax with:
    model.attention.softmax = WheelSoftmax()
"""

import torch
import torch.nn as nn
import numpy as np
from wheelgrad.ops import wheel_softmax, wheel_layernorm


class WheelSoftmaxFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        x_np = x.detach().numpy()
        result = wheel_softmax(x_np, axis=dim)
        ctx.save_for_backward(torch.from_numpy(result.values.astype(np.float32)))
        ctx.bot_mask = torch.from_numpy(result.bot_mask)
        ctx.dim = dim
        return torch.from_numpy(result.to_numpy_safe().astype(np.float32))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        probs, = ctx.saved_tensors
        bot_mask = ctx.bot_mask
        dim = ctx.dim
        # Standard softmax Jacobian-vector product
        s = probs
        grad = s * (grad_output - (grad_output * s).sum(dim=dim, keepdim=True))
        # KEY: zero gradient at ⊥ positions — absorbing element stops gradient
        grad[bot_mask] = 0.0
        return grad, None


class WheelSoftmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return WheelSoftmaxFn.apply(x, self.dim)


class WheelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.weight.detach().numpy() if self.weight is not None else None
        beta  = self.bias.detach().numpy()   if self.bias  is not None else None
        result = wheel_layernorm(x.detach().numpy(), gamma, beta)
        # Convert back, preserving ⊥ as 0 with gradient zeroed
        out = torch.from_numpy(result.to_numpy_safe().astype(np.float32))
        out.requires_grad_(x.requires_grad)
        return out
'''
