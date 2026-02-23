"""
wheelgrad/core.py
─────────────────
Core Wheel arithmetic implementation.

A Wheel extends any commutative ring so that division is total:
    a / 0 = ∞    (for a ≠ 0)
    0 / 0 = ⊥    (bottom — absorbing element)

Key property: ⊥ is NOT NaN. It is a DEFINED algebraic element.
    ⊥ + x = ⊥   (absorbing under addition)
    ⊥ · x = ⊥   (absorbing under multiplication)
    ⊥ / x = ⊥   (absorbing under division)

Implementation strategy:
    We encode Wheel semantics as a pair (values, mask) over float32:
        values : numpy array of float32 — numeric values
        bot_mask : bool array          — True where element is ⊥
    
    This encodes three states per element:
        - Finite value    : bot_mask=False, values=x
        - Infinity (∞)    : bot_mask=False, values=np.inf
        - Bottom (⊥)      : bot_mask=True  (values ignored)
    
    GPU compatibility: float32 array + bool mask → native CUDA tensors.
    Zero overhead compared to standard float32 computation.
"""

import numpy as np
from typing import Union, Tuple


# ── Scalar Wheel element ─────────────────────────────────────────────────────

class WheelScalar:
    """
    A single element of the Wheel ℝ_w = ℝ ∪ {∞, ⊥}.
    
    Examples
    --------
    >>> WheelScalar(5.0) / WheelScalar(0.0)
    WheelScalar(∞)
    >>> WheelScalar(0.0) / WheelScalar(0.0)
    WheelScalar(⊥)
    >>> WheelScalar(3.0) + WheelScalar.bottom()
    WheelScalar(⊥)
    """
    __slots__ = ('_value', '_is_bot')

    def __init__(self, value: float, _is_bot: bool = False):
        self._is_bot = _is_bot
        if _is_bot:
            self._value = 0.0
        elif np.isinf(value):
            self._value = float('inf')
        else:
            self._value = float(value)

    # ── Constructors ──────────────────────────────────────────────────────
    @classmethod
    def bottom(cls) -> 'WheelScalar':
        """Return the absorbing element ⊥ (0/0)."""
        return cls(0.0, _is_bot=True)

    @classmethod
    def infinity(cls) -> 'WheelScalar':
        """Return the projective infinity ∞ (1/0)."""
        return cls(float('inf'))

    @classmethod
    def from_float(cls, x: float) -> 'WheelScalar':
        if np.isnan(x):
            # IEEE NaN → Wheel ⊥ (the algebraically defined equivalent)
            return cls.bottom()
        return cls(x)

    # ── Properties ───────────────────────────────────────────────────────
    @property
    def is_bot(self) -> bool:
        return self._is_bot

    @property
    def is_inf(self) -> bool:
        return not self._is_bot and np.isinf(self._value)

    @property
    def is_finite(self) -> bool:
        return not self._is_bot and not np.isinf(self._value)

    # ── Wheel operations ─────────────────────────────────────────────────
    def __add__(self, other: 'WheelScalar') -> 'WheelScalar':
        if self._is_bot or other._is_bot:
            return WheelScalar.bottom()
        if self.is_inf or other.is_inf:
            # ∞ + ∞ = ⊥  (indeterminate in projective sense)
            if self.is_inf and other.is_inf:
                return WheelScalar.bottom()
            return WheelScalar.infinity()
        return WheelScalar(self._value + other._value)

    def __mul__(self, other: 'WheelScalar') -> 'WheelScalar':
        if self._is_bot or other._is_bot:
            return WheelScalar.bottom()
        if self.is_inf or other.is_inf:
            # 0 · ∞ = ⊥
            if self._value == 0.0 or other._value == 0.0:
                return WheelScalar.bottom()
            return WheelScalar.infinity()
        v = self._value * other._value
        return WheelScalar.bottom() if np.isnan(v) else WheelScalar(v)

    def __truediv__(self, other: 'WheelScalar') -> 'WheelScalar':
        """Core Wheel axiom: total division."""
        if self._is_bot or other._is_bot:
            return WheelScalar.bottom()
        if other.is_inf:
            return WheelScalar(0.0)
        if other._value == 0.0:
            # THE WHEEL AXIOM
            return WheelScalar.bottom() if self._value == 0.0 else WheelScalar.infinity()
        if self.is_inf:
            return WheelScalar.infinity()
        v = self._value / other._value
        return WheelScalar.bottom() if np.isnan(v) else WheelScalar(v)

    def __neg__(self) -> 'WheelScalar':
        if self._is_bot or self.is_inf:
            return self
        return WheelScalar(-self._value)

    def exp(self) -> 'WheelScalar':
        if self._is_bot:
            return WheelScalar.bottom()
        if self.is_inf:
            return WheelScalar.infinity()
        v = np.exp(self._value)
        return WheelScalar.infinity() if np.isinf(v) else WheelScalar(v)

    def log(self) -> 'WheelScalar':
        if self._is_bot:
            return WheelScalar.bottom()
        if self.is_inf:
            return WheelScalar.infinity()
        if self._value <= 0.0:
            return WheelScalar.bottom() if self._value == 0.0 else WheelScalar(float('-inf'))
        return WheelScalar(np.log(self._value))

    # ── Display ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        if self._is_bot:
            return 'WheelScalar(⊥)'
        if self.is_inf:
            return 'WheelScalar(∞)'
        return f'WheelScalar({self._value})'

    def __str__(self) -> str:
        if self._is_bot: return '⊥'
        if self.is_inf:  return '∞'
        return str(self._value)

    def to_float(self) -> float:
        """Convert to float. ⊥ → NaN, ∞ → inf."""
        if self._is_bot: return float('nan')
        return self._value


# ── Tensor Wheel (vectorized over numpy) ─────────────────────────────────────

class WheelTensor:
    """
    Vectorized Wheel arithmetic over numpy arrays.
    
    Internally stores:
        values   : np.ndarray[float64] — numeric values (∞ for infinity)
        bot_mask : np.ndarray[bool]    — True where element is ⊥
    
    This design maps directly to PyTorch tensors:
        torch.Tensor (float32) + torch.BoolTensor
    with zero extra memory overhead per element (1 bool = 1 byte vs 4 bytes float).
    
    Parameters
    ----------
    values : array-like
        Numeric values. Use np.inf for ∞ elements.
    bot_mask : array-like of bool, optional
        True where elements are ⊥. Defaults to all False.
    
    Examples
    --------
    >>> t = WheelTensor([1.0, 0.0, 2.0])
    >>> t / WheelTensor([2.0, 0.0, 0.0])
    WheelTensor([0.5, ⊥, ∞])
    """

    def __init__(
        self,
        values: Union[np.ndarray, list],
        bot_mask: Union[np.ndarray, list, None] = None
    ):
        self.values = np.asarray(values, dtype=np.float64)
        if bot_mask is None:
            self.bot_mask = np.zeros(self.values.shape, dtype=bool)
        else:
            self.bot_mask = np.asarray(bot_mask, dtype=bool)

        # Auto-detect NaN in input → convert to ⊥
        nan_mask = np.isnan(self.values)
        self.bot_mask |= nan_mask
        self.values[nan_mask] = 0.0

    # ── Constructors ──────────────────────────────────────────────────────
    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> 'WheelTensor':
        """Import numpy array; NaN → ⊥, inf stays ∞."""
        return cls(np.nan_to_num(arr, nan=0.0), bot_mask=np.isnan(arr))

    @classmethod
    def zeros(cls, *shape) -> 'WheelTensor':
        return cls(np.zeros(shape))

    @classmethod
    def ones(cls, *shape) -> 'WheelTensor':
        return cls(np.ones(shape))

    # ── Shape ─────────────────────────────────────────────────────────────
    @property
    def shape(self) -> Tuple:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    def __len__(self) -> int:
        return len(self.values)

    # ── Status queries ────────────────────────────────────────────────────
    @property
    def has_bot(self) -> bool:
        return bool(self.bot_mask.any())

    @property
    def has_inf(self) -> bool:
        return bool(np.isinf(self.values[~self.bot_mask]).any())

    @property
    def is_clean(self) -> bool:
        """True if no ⊥ or ∞ elements."""
        return not self.has_bot and not self.has_inf

    def bot_count(self) -> int:
        return int(self.bot_mask.sum())

    def inf_count(self) -> int:
        return int(np.isinf(self.values[~self.bot_mask]).sum())

    # ── Wheel arithmetic (vectorized) ─────────────────────────────────────
    def __add__(self, other: 'WheelTensor') -> 'WheelTensor':
        # ⊥ absorbs
        new_bot = self.bot_mask | other.bot_mask
        # ∞ + ∞ = ⊥
        both_inf = np.isinf(self.values) & np.isinf(other.values) & ~new_bot
        new_bot |= both_inf

        new_vals = self.values + other.values
        # Clean up NaN from inf arithmetic
        new_vals[new_bot] = 0.0
        return WheelTensor(new_vals, new_bot)

    def __sub__(self, other: 'WheelTensor') -> 'WheelTensor':
        return self.__add__(-other)

    def __neg__(self) -> 'WheelTensor':
        new_vals = np.where(self.bot_mask, 0.0, -self.values)
        return WheelTensor(new_vals, self.bot_mask.copy())

    def __mul__(self, other: Union['WheelTensor', float, int]) -> 'WheelTensor':
        if isinstance(other, (int, float)):
            other = WheelTensor(np.full(self.shape, other))
        new_bot = self.bot_mask | other.bot_mask
        # 0 · ∞ = ⊥
        zero_inf = (
            ((self.values == 0.0) & np.isinf(other.values)) |
            ((other.values == 0.0) & np.isinf(self.values))
        ) & ~new_bot
        new_bot |= zero_inf
        new_vals = self.values * other.values
        new_vals[np.isnan(new_vals)] = 0.0
        new_vals[new_bot] = 0.0
        return WheelTensor(new_vals, new_bot)

    def __truediv__(self, other: Union['WheelTensor', float, int]) -> 'WheelTensor':
        """
        Total division — the core Wheel operation.
        
        Rules (all vectorized):
            ⊥ / x = ⊥       (⊥ absorbs)
            x / ⊥ = ⊥       (⊥ absorbs)
            0 / 0 = ⊥       (indeterminate)
            a / 0 = ∞        (a ≠ 0)
            x / ∞ = 0
            ∞ / x = ∞        (x finite, x ≠ 0)
        """
        if isinstance(other, (int, float)):
            other = WheelTensor(np.full(self.shape, float(other)))

        new_bot  = self.bot_mask | other.bot_mask
        div_zero = (~new_bot) & (other.values == 0.0)

        # 0/0 = ⊥
        bot_from_00 = div_zero & (self.values == 0.0)
        new_bot |= bot_from_00

        # a/0 = ∞  (a ≠ 0)
        inf_from_a0 = div_zero & ~bot_from_00

        # Safe division (avoid actual /0 in numpy)
        safe_denom = np.where(other.values == 0.0, 1.0, other.values)
        safe_denom = np.where(other.bot_mask, 1.0, safe_denom)
        new_vals = self.values / safe_denom

        # Apply infinity from a/0
        new_vals = np.where(inf_from_a0, np.inf, new_vals)
        # x/∞ = 0
        new_vals = np.where(np.isinf(other.values) & ~other.bot_mask, 0.0, new_vals)

        new_vals[new_bot] = 0.0
        return WheelTensor(new_vals, new_bot)

    def __rtruediv__(self, other: float) -> 'WheelTensor':
        return WheelTensor(np.full(self.shape, float(other))).__truediv__(self)

    def exp(self) -> 'WheelTensor':
        new_vals = np.where(self.bot_mask, 0.0, np.exp(np.clip(self.values, -500, 500)))
        # Original overflow → ∞ (not ⊥)
        overflow = (~self.bot_mask) & (self.values > 500)
        new_vals = np.where(overflow, np.inf, new_vals)
        return WheelTensor(new_vals, self.bot_mask.copy())

    def log(self) -> 'WheelTensor':
        new_bot = self.bot_mask.copy()
        # log(0) = ⊥  (not -∞, which would be a NaN risk)
        new_bot |= ((self.values == 0.0) & ~self.bot_mask)
        new_vals = np.where(new_bot, 0.0,
                   np.where(np.isinf(self.values), np.inf,
                   np.log(np.maximum(self.values, 1e-300))))
        return WheelTensor(new_vals, new_bot)

    def sum(self, axis=None) -> 'WheelTensor':
        if axis is None:
            has_bot = self.bot_mask.any()
            val = float(self.values[~self.bot_mask].sum()) if not has_bot else 0.0
            return WheelTensor([val], [bool(has_bot)])
        # Sum along axis: if any ⊥ in a slice, result is ⊥
        bot_out = self.bot_mask.any(axis=axis)
        # Temporarily zero out ⊥ positions before summing
        safe_vals = np.where(self.bot_mask, 0.0, self.values)
        val_out = safe_vals.sum(axis=axis)
        val_out[bot_out] = 0.0
        return WheelTensor(val_out, bot_out)

    def max(self, axis=None):
        safe_vals = np.where(self.bot_mask, -np.inf, self.values)
        if axis is None:
            return WheelScalar.from_float(float(safe_vals.max()))
        return WheelTensor(safe_vals.max(axis=axis),
                           self.bot_mask.all(axis=axis))

    def __sub__(self, other: Union['WheelTensor', float]) -> 'WheelTensor':
        if isinstance(other, (int, float)):
            other = WheelTensor(np.full(self.shape, float(other)))
        return self.__add__(-other)

    def __rsub__(self, other: float) -> 'WheelTensor':
        return WheelTensor(np.full(self.shape, float(other))).__sub__(self)

    def __radd__(self, other: float) -> 'WheelTensor':
        return self.__add__(WheelTensor(np.full(self.shape, float(other))))

    def __rmul__(self, other: float) -> 'WheelTensor':
        return self.__mul__(other)

    # ── Indexing ──────────────────────────────────────────────────────────
    def __getitem__(self, idx):
        v = self.values[idx]
        b = self.bot_mask[idx]
        if np.isscalar(v):
            return WheelScalar(float(v), bool(b))
        return WheelTensor(v, b)

    def __setitem__(self, idx, value):
        if isinstance(value, WheelScalar):
            self.values[idx] = 0.0 if value.is_bot else value._value
            self.bot_mask[idx] = value.is_bot
        elif isinstance(value, WheelTensor):
            self.values[idx] = value.values
            self.bot_mask[idx] = value.bot_mask

    # ── Conversion ────────────────────────────────────────────────────────
    def to_numpy(self) -> np.ndarray:
        """Export to numpy. ⊥ → NaN, ∞ → np.inf."""
        out = self.values.copy()
        out[self.bot_mask] = np.nan
        return out

    def to_numpy_safe(self, bot_fill: float = 0.0, inf_fill: float = 1e9) -> np.ndarray:
        """Export replacing ⊥ and ∞ with finite values."""
        out = self.values.copy()
        out[self.bot_mask] = bot_fill
        out[np.isinf(out) & ~self.bot_mask] = inf_fill
        return out

    # ── Display ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        items = []
        flat_vals = self.values.flat
        flat_bots = self.bot_mask.flat
        for v, b in zip(flat_vals, flat_bots):
            if b:         items.append('⊥')
            elif np.isinf(v): items.append('∞' if v > 0 else '-∞')
            else:         items.append(f'{v:.4f}')
        inner = ', '.join(items)
        return f'WheelTensor([{inner}])'

    def status(self) -> str:
        n = self.values.size
        nb = self.bot_count()
        ni = self.inf_count()
        nf = n - nb - ni
        return (f'WheelTensor shape={self.shape} | '
                f'finite={nf} | ∞={ni} | ⊥={nb} | '
                f'{"CLEAN" if self.is_clean else "HAS SPECIAL ELEMENTS"}')
