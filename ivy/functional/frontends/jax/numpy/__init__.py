# flake8: noqa
from ivy import (
    bool,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
    complex64,
    complex128,
    complex256,
    bfloat16,
)


jax_numpy_casting_table = {
    bool: [
        bool,
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float16,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
        bfloat16,
    ],
    int8: [
        int8,
        int16,
        int32,
        int64,
        float16,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
        bfloat16,
    ],
    int16: [
        int16,
        int32,
        int64,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    int32: [
        int32,
        int64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    int64: [
        int64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    uint8: [
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float16,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
        bfloat16,
    ],
    uint16: [
        int32,
        int64,
        uint16,
        uint32,
        uint64,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    uint32: [
        int64,
        uint32,
        uint64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    uint64: [
        uint64,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    float16: [
        float16,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    float32: [
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
    float64: [
        float64,
        complex64,
        complex128,
        complex256,
    ],
    complex64: [complex64, complex128, complex256, bfloat16],
    complex128: [complex128, complex256, bfloat16],
    complex256: [complex256, bfloat16],
    bfloat16: [
        bfloat16,
        float32,
        float64,
        complex64,
        complex128,
        complex256,
    ],
}


from . import fft
from . import linalg
from . import creation
from .creation import *
from . import name_space_functions
from .name_space_functions import *
from . import dtype
from .dtype import *

from .._src.numpy.lax_numpy import _rewriting_take
