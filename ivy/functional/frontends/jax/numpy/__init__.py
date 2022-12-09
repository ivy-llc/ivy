# flake8: noqa
import ivy


jax_numpy_promotion_table = {
    ivy.bool: [
        ivy.bool,
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.int8: [
        ivy.int8,
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint8,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.int16: [
        ivy.int16,
        ivy.int32,
        ivy.int64,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.int32: [
        ivy.int32,
        ivy.int64,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.int64: [
        ivy.int64,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.uint8: [
        ivy.uint8,
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.uint16: [
        ivy.uint16,
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.uint32: [
        ivy.uint32,
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.uint64: [
        ivy.uint64,
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.float16: [
        ivy.float16,
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
    ],
    ivy.float32: [
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.float64: [
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
        ivy.bfloat16,
    ],
    ivy.complex64: [ivy.complex64, ivy.complex128, ivy.complex256, ivy.bfloat16],
    ivy.complex128: [ivy.complex128, ivy.complex256, ivy.bfloat16],
    ivy.complex256: [ivy.complex256, ivy.bfloat16],
    ivy.bfloat16: [
        ivy.float32,
        ivy.float64,
        ivy.complex64,
        ivy.complex128,
        ivy.complex256,
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
