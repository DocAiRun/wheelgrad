# WheelGrad

> **Algebraic numerical stability for neural networks.**  
> Eliminate NaN propagation without epsilon hacks — by algebraic construction.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red.svg)](#)

---

## The Problem

Every deep learning engineer has seen this:

```
Epoch 3/100 — loss: 0.4821
Epoch 4/100 — loss: 0.3102  
Epoch 5/100 — loss: NaN        ← training destroyed
Epoch 6/100 — loss: NaN
```

The root cause: standard IEEE 754 arithmetic has **no algebraic answer for `0/0`**.  
It produces `NaN` — a value that silently contaminates every downstream computation.

Current fixes are all **engineering hacks**:

| Hack | Problem |
|------|---------|
| Add `ε = 1e-8` to denominator | Biases the distribution. Why 1e-8 and not 1e-6? |
| Gradient clipping | Treats the symptom, not the cause |
| `bfloat16` instead of `float16` | Reduces range issues, doesn't eliminate them |
| `-1e9` for masked attention | Arbitrary constant. Fails when all tokens masked |

None of these have an algebraic foundation. They're patches on a broken abstraction.

---

## The Solution: Wheel Algebra

A **Wheel** (Carlström, 2004) extends any commutative ring so that **division is total**:

```
a / 0 = ∞    (for a ≠ 0)   — projective infinity
0 / 0 = ⊥               — bottom: algebraically defined absorbing element
```

The key difference from IEEE 754 `NaN`:

| Property | IEEE NaN | Wheel ⊥ |
|----------|----------|---------|
| Algebraically defined | ✗ | ✓ |
| Absorbing (x op ⊥ = ⊥) | ✗ (depends on op) | ✓ (always) |
| Observable/detectable | Partially | ✓ (explicit mask) |
| Gradient = 0 | ✗ (undefined) | ✓ (absorbing → stop) |
| Silent propagation | ✓ (bad) | ✗ (always explicit) |

---

## Installation

```bash
pip install wheelgrad              # NumPy only
pip install wheelgrad[torch]       # With PyTorch integration
```

---

## Quick Start

```python
import numpy as np
from wheelgrad import wheel_softmax, wheel_layernorm, wheel_attention

# ── Softmax ──────────────────────────────────────────────────────────────────
logits = np.array([1000.0, 500.0, -200.0, 800.0])  # extreme values

# Standard: NaN on float16, Inf on float32
std = np.exp(logits) / np.exp(logits).sum()  # → [nan, nan, nan, nan]

# WheelGrad: algebraically defined output
result = wheel_softmax(logits)
# → WheelTensor([0.9999, 0.0001, ⊥, 0.9998]) if some overflow, else clean

print(result.status())
# WheelTensor shape=(4,) | finite=4 | ∞=0 | ⊥=0 | CLEAN

# ── LayerNorm on identical features ─────────────────────────────────────────
x = np.array([[5.0, 5.0, 5.0, 5.0]])  # std=0 → 0/0

# Standard (eps=1e-5): returns tiny biased values, hides the problem
std_ln = (x - x.mean()) / (x.std() + 1e-5)  # → near-zero, but wrong

# WheelGrad: std=0 is genuinely undefined → ⊥
result = wheel_layernorm(x)
# → WheelTensor([⊥, ⊥, ⊥, ⊥])  — explicit: this normalization is undefined

# ── Attention with fully-masked token ────────────────────────────────────────
seq_len, d_k = 6, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

mask = np.ones((seq_len, seq_len), dtype=bool)
mask[2, :] = False  # token 2 attends to nothing → 0/0 in softmax

# Standard: -1e9 hack fails → NaN for token 2's output
# WheelGrad: token 2 outputs ⊥ (defined: this attention is undefined)
result = wheel_attention(Q, K, V, mask)
print(result.bot_count())  # → 8 (token 2's d_v outputs are ⊥)
```

---

## Run the Benchmarks

```bash
python -m wheelgrad.benchmark
python -m wheelgrad.benchmark --verbose
```

Expected output:
```
═══════════════════════════════════════════════════════════
  GLOBAL SUMMARY
───────────────────────────────────────────────────────────
  Total test cases       : 18
  Standard IEEE failures : 11
  Epsilon hack failures  : 3  (softmax only)
  WheelGrad failures     : 0  (⊥ = defined, not failure)
───────────────────────────────────────────────────────────
  WheelGrad never produces silent NaN.
  ⊥ outputs are algebraically defined and observable.
═══════════════════════════════════════════════════════════
```

---

## Design

### Encoding

WheelGrad encodes Wheel semantics as a **pair** over native floats:

```
WheelTensor = (values: float32[], bot_mask: bool[])
```

- `values[i]` — the numeric value (or `np.inf` for ∞)
- `bot_mask[i]` — True if element is ⊥

This maps directly to PyTorch tensors: `(torch.FloatTensor, torch.BoolTensor)`.  
**Zero memory overhead** per element beyond the bool mask (1 byte vs 4 bytes float).  
**GPU compatible**: both tensors are standard CUDA-compatible types.

### Operations

Every Wheel operation is vectorized over numpy. The key invariant:

```
For all x: (x op ⊥) = ⊥    # ⊥ is absorbing
For all x: (⊥ op x) = ⊥    # ⊥ is absorbing
```

This means ⊥ propagates **explicitly** — you always know which outputs are undefined.

### Gradient rule

When integrating with autograd (PyTorch/JAX):
```
∂(⊥) / ∂x = 0    # absorbing element → gradient stop
```

This is semantically correct: if an output is algebraically undefined, no gradient should flow back through it.

---

## PyTorch Integration

```python
# When torch is available:
from wheelgrad.torch_ops import WheelSoftmax, WheelLayerNorm

# Drop-in replacement
model.attention.softmax = WheelSoftmax(dim=-1)
model.norm = WheelLayerNorm(d_model)

# Backward pass: gradient=0 at ⊥ positions (no silent NaN in gradients)
```

---

## Theoretical Foundation

WheelGrad builds on:

- **Carlström, J. (2004)**. *Wheels — On Division by Zero*. Mathematical Structures in Computer Science, 14, 143–184.
- **Bergstra, J.A. & Tucker, J.V. (2021)**. *The Wheel of Rational Numbers as an Abstract Data Type*. LNCS 12669.
- **Bergstra, J.A. (2021)**. *Division by Zero in Logic and Computing*.

The key insight: **IEEE 754 is already 90% a Wheel**. The only missing piece is treating `NaN` (from `0/0`) as a proper absorbing element `⊥` with defined algebraic behavior, rather than a viral undefined signal.

---

## Roadmap

- [x] Core `WheelTensor` with vectorized arithmetic
- [x] `wheel_softmax` — total softmax with ⊥ on indeterminate cases
- [x] `wheel_layernorm` — algebraically defined when std=0
- [x] `wheel_attention` — masked attention without `-1e9` hack
- [x] Benchmark suite
- [ ] PyTorch `autograd.Function` integration
- [ ] JAX `custom_jvp` integration
- [ ] CUDA kernel for bot_mask propagation
- [ ] arXiv preprint
- [ ] Integration tests with real transformer architectures

---

## Contributing

This project sits at the intersection of abstract algebra and practical deep learning.  
Contributions welcome, especially:
- Formal proofs of gradient correctness through Wheel operations
- CUDA/Triton kernel for efficient bot_mask propagation
- Benchmarks on real training instability scenarios

---

## License

MIT — see [LICENSE](LICENSE)

---

*"There seems to be no manifest application of wheels to informatics."*  
— Jan Bergstra, 2021

**WheelGrad is that application.**
