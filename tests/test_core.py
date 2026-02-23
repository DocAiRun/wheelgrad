"""Basic tests for WheelGrad core arithmetic."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wheelgrad.core import WheelScalar, WheelTensor
from wheelgrad.ops import wheel_softmax, wheel_layernorm

def test_wheel_scalar_division():
    assert str(WheelScalar(5.0) / WheelScalar(0.0)) == '∞'
    assert str(WheelScalar(0.0) / WheelScalar(0.0)) == '⊥'
    assert (WheelScalar(6.0) / WheelScalar(2.0))._value == 3.0

def test_wheel_bottom_absorbs():
    bot = WheelScalar.bottom()
    assert (bot + WheelScalar(5.0)).is_bot
    assert (bot * WheelScalar(5.0)).is_bot

def test_wheel_tensor_division():
    t = WheelTensor([1.0, 0.0, 3.0])
    z = WheelTensor([2.0, 0.0, 0.0])
    r = t / z
    assert not r.bot_mask[0]   # 1/2 = 0.5
    assert r.bot_mask[1]       # 0/0 = ⊥
    assert np.isinf(r.values[2])  # 3/0 = ∞

def test_softmax_nan_input():
    logits = np.array([1.0, np.nan, 2.0, 0.5])
    result = wheel_softmax(logits)
    assert result.has_bot  # NaN → ⊥, not silent

def test_softmax_normal():
    logits = np.array([2.1, 0.8, -0.3, 1.5])
    result = wheel_softmax(logits)
    assert result.is_clean
    assert abs(result.to_numpy_safe().sum() - 1.0) < 1e-6

def test_layernorm_zero_std():
    x = np.array([[5.0, 5.0, 5.0, 5.0]])
    result = wheel_layernorm(x)
    assert result.has_bot  # std=0 → ⊥, not biased epsilon

if __name__ == '__main__':
    test_wheel_scalar_division()
    test_wheel_bottom_absorbs()
    test_wheel_tensor_division()
    test_softmax_nan_input()
    test_softmax_normal()
    test_layernorm_zero_std()
    print("All tests passed ✓")
