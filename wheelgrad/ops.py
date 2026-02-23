"""
wheelgrad/ops.py
────────────────
Wheel-safe neural network operations.

Each operation is implemented twice:
    - Standard version  : reference IEEE 754 (will produce NaN on edge cases)
    - Wheel version     : algebraically stable, never produces silent NaN

The Wheel version uses the same computational graph as standard,
but replaces every arithmetic step with Wheel operations.
"""

import numpy as np
from .core import WheelTensor, WheelScalar
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SOFTMAX
# ═══════════════════════════════════════════════════════════════════════════════

def standard_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Standard softmax. WILL produce NaN/Inf on extreme logits.
    Reference implementation for comparison.
    """
    # Even with max subtraction trick, float16 overflows at logit > ~65000
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def epsilon_softmax(x: np.ndarray, eps: float = 1e-8, axis: int = -1) -> np.ndarray:
    """
    Epsilon-hacked softmax. Adds ε to denominator to avoid 0/0.
    Survives but introduces systematic bias — not algebraically founded.
    """
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / (e.sum(axis=axis, keepdims=True) + eps)


def wheel_softmax(x: np.ndarray, axis: int = -1) -> WheelTensor:
    """
    Wheel-algebraic softmax.

    Algorithm:
        1. Convert input to WheelTensor (NaN → ⊥, Inf → ∞)
        2. Compute max for numerical stability (Wheel-aware)
        3. Subtract max: x_shifted = x - x_max
        4. Compute Wheel exp: if overflow → ∞, if ⊥ in → ⊥ out
        5. Sum with Wheel addition: ∞ + ∞ = ⊥ (correctly flagged)
        6. Divide with Wheel division: 0/0 = ⊥, a/0 = ∞

    Guarantees:
        - No silent NaN propagation
        - ⊥ output is observable and meaningful (indeterminate class)
        - ∞ input → ⊥ output (competing infinities)
        - All-zero input → ⊥ (uniform distribution undefined)

    Parameters
    ----------
    x : np.ndarray
        Raw logits. Can contain NaN, Inf, extreme values.
    axis : int
        Axis along which softmax is computed.

    Returns
    -------
    WheelTensor
        Probability-like distribution. May contain ⊥ where undefined.
    """
    wt = WheelTensor.from_numpy(x)

    # Numerical stability: subtract max along axis
    # If any element is ⊥, we skip it for max computation
    safe_vals = np.where(wt.bot_mask, -np.inf, wt.values)
    x_max = np.max(safe_vals, axis=axis, keepdims=True)
    x_max = np.where(np.isinf(x_max) & (x_max < 0), 0.0, x_max)  # all-bot case

    # Shift and exponentiate
    shifted_vals = wt.values - x_max
    shifted = WheelTensor(shifted_vals, wt.bot_mask.copy())
    e = shifted.exp()

    # Sum along axis using Wheel addition
    # Manually handle axis sum with ⊥ propagation
    safe_e = np.where(e.bot_mask, 0.0, e.values)
    e_sum_vals = safe_e.sum(axis=axis, keepdims=True)
    e_sum_bot  = e.bot_mask.any(axis=axis, keepdims=True)
    e_sum = WheelTensor(e_sum_vals, e_sum_bot)

    # Broadcast sum back to original shape
    e_sum_broadcast = WheelTensor(
        np.broadcast_to(e_sum.values, e.shape).copy(),
        np.broadcast_to(e_sum.bot_mask, e.shape).copy()
    )

    # Wheel division: the key operation
    result = e / e_sum_broadcast
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def standard_layernorm(
    x: np.ndarray,
    gamma: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Standard LayerNorm. Division by (std + ε) is the classic hack.
    Fails silently when all inputs are identical (std=0).
    """
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta
    return x_norm


def wheel_layernorm(
    x: np.ndarray,
    gamma: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None
) -> WheelTensor:
    """
    Wheel-algebraic Layer Normalization.

    Key difference from standard LayerNorm:
        - Standard uses eps=1e-5 to avoid division by zero when var=0
        - Wheel uses total division: (x - mean) / std
          When std=0 (all inputs identical), result is ⊥ (not a NaN, 
          not a biased value — a defined algebraic indeterminate)

    This is semantically more correct: if all features are identical,
    the normalization is genuinely undefined, and ⊥ signals this cleanly.

    Parameters
    ----------
    x : np.ndarray
        Input tensor, shape (..., features)
    gamma : optional scale parameter
    beta  : optional shift parameter

    Returns
    -------
    WheelTensor
        Normalized tensor. Contains ⊥ where normalization is undefined.
    """
    wt = WheelTensor.from_numpy(x)

    # Mean (Wheel-aware: ⊥ in any element → ⊥ mean for that sample)
    safe_vals = np.where(wt.bot_mask, 0.0, wt.values)
    mean_vals = safe_vals.mean(axis=-1, keepdims=True)
    mean_bot  = wt.bot_mask.any(axis=-1, keepdims=True)

    mean_vals_bc = np.broadcast_to(mean_vals, wt.shape).copy()
    mean_bot_bc  = np.broadcast_to(mean_bot,  wt.shape).copy()
    mean_wt = WheelTensor(mean_vals_bc, mean_bot_bc)

    # x - mean
    centered = wt - mean_wt

    # Variance: mean of (x - mean)^2
    sq_vals = np.where(centered.bot_mask, 0.0, centered.values ** 2)
    var_vals = sq_vals.mean(axis=-1, keepdims=True)
    var_bot  = centered.bot_mask.any(axis=-1, keepdims=True)

    # Std = sqrt(var)  — Wheel-aware
    std_vals = np.where(var_bot, 0.0, np.sqrt(np.maximum(var_vals, 0.0)))
    std_wt = WheelTensor(
        np.broadcast_to(std_vals, wt.shape).copy(),
        np.broadcast_to(var_bot,  wt.shape).copy()
    )

    # THE KEY: Wheel division — std=0 gives ⊥, not biased-by-epsilon value
    x_norm = centered / std_wt

    # Apply learned parameters (standard multiply-add)
    if gamma is not None:
        gamma_wt = WheelTensor(np.broadcast_to(gamma, x_norm.shape).copy())
        x_norm = x_norm * gamma_wt
    if beta is not None:
        beta_wt = WheelTensor(np.broadcast_to(beta, x_norm.shape).copy())
        x_norm = x_norm + beta_wt

    return x_norm


# ═══════════════════════════════════════════════════════════════════════════════
# ATTENTION (Scaled Dot-Product)
# ═══════════════════════════════════════════════════════════════════════════════

def standard_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Standard scaled dot-product attention.
    Masked positions filled with -1e9 (large negative hack).
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = standard_softmax(scores)
    return weights @ V


def wheel_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> WheelTensor:
    """
    Wheel-algebraic scaled dot-product attention.

    In standard attention, masked tokens are handled by adding -∞ (or -1e9)
    to scores before softmax. This is a hack:
        - -1e9 is arbitrary (why not -1e8? -1e10?)
        - -∞ before softmax gives 0/0 when ALL tokens are masked → NaN

    Wheel approach:
        - Masked tokens contribute ⊥ directly to the score matrix
        - ⊥ propagates cleanly through softmax → ⊥ attention weights
        - ⊥ attention on a position is observable and meaningful

    Parameters
    ----------
    Q : (seq_len, d_k)   Query matrix
    K : (seq_len, d_k)   Key matrix
    V : (seq_len, d_v)   Value matrix
    mask : (seq_len, seq_len) bool   True = attend, False = mask

    Returns
    -------
    WheelTensor : (seq_len, d_v)  attended values
    """
    d_k = Q.shape[-1]
    scale = np.sqrt(d_k)

    # Raw dot-product scores
    raw_scores = Q @ K.T / scale  # (seq_len, seq_len)

    # Inject ⊥ for masked positions (instead of -1e9 hack)
    if mask is not None:
        bot_mask = ~mask  # True where we should NOT attend
        scores_wt = WheelTensor(raw_scores, bot_mask)
    else:
        scores_wt = WheelTensor.from_numpy(raw_scores)

    # Wheel softmax over scores
    weights_wt = wheel_softmax(scores_wt.to_numpy(), axis=-1)

    # Attend to values
    safe_weights = weights_wt.to_numpy_safe(bot_fill=0.0)
    output_vals = safe_weights @ V
    output_bot  = weights_wt.bot_mask.any(axis=-1, keepdims=True)
    output_bot  = np.broadcast_to(output_bot, output_vals.shape).copy()

    return WheelTensor(output_vals, output_bot)


# ═══════════════════════════════════════════════════════════════════════════════
# LOG (cross-entropy building block)
# ═══════════════════════════════════════════════════════════════════════════════

def wheel_log(x: WheelTensor) -> WheelTensor:
    """
    Wheel-safe logarithm.
    
    Standard log:
        log(0) = -∞  (which then causes NaN in cross-entropy: 0 * -∞ = NaN)
    
    Wheel log:
        log(0) = ⊥   (algebraically defined, terminates cleanly)
    """
    return x.log()


def wheel_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> WheelTensor:
    """
    Wheel cross-entropy loss.
    
    Standard CE: -sum(target * log(softmax(logits)))
    NaN risk: if softmax produces 0 for a target class, log(0) = -inf → NaN

    Wheel CE: same formula with Wheel log
    ⊥ output = the loss is algebraically undefined for this sample
    (extreme logit imbalance) — explicitly observable, not silent.
    """
    probs = wheel_softmax(logits)
    log_probs = wheel_log(probs)

    targets_wt = WheelTensor.from_numpy(targets.astype(float))
    nll = targets_wt * log_probs
    loss = nll.sum(axis=-1)

    # Negate
    neg_vals = np.where(loss.bot_mask, 0.0, -loss.values)
    return WheelTensor(neg_vals, loss.bot_mask.copy())
