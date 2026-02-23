"""
WheelGrad — Algebraic Stability for Neural Networks
====================================================
Wheel arithmetic applied to deep learning numerical stability.
Eliminates NaN propagation without epsilon hacks — by algebraic construction.

Based on: Carlström (2004) "Wheels — On Division by Zero"
          Bergstra & Tucker (2021) "The Wheel of Rational Numbers as an ADT"

Usage:
    from wheelgrad import WheelTensor, wheel_softmax, wheel_layernorm
"""

from .core import WheelTensor, WheelScalar
from .ops import wheel_softmax, wheel_layernorm, wheel_attention, wheel_log
from .utils import encode_wheel, decode_wheel, wheel_status_report

__version__ = "0.1.0"
__author__  = "WheelGrad Contributors"
__license__ = "MIT"

__all__ = [
    "WheelTensor", "WheelScalar",
    "wheel_softmax", "wheel_layernorm", "wheel_attention", "wheel_log",
    "encode_wheel", "decode_wheel", "wheel_status_report",
]
