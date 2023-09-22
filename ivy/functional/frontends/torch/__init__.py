# global
import sys
from numbers import Number
from typing import Union, Tuple, Iterable

# local
import ivy
from ivy.utils.exceptions import handle_exceptions
from ivy.functional.frontends import set_frontend_to_specific_version


# Constructing dtypes are required as ivy.<dtype>
# will change dynamically on the backend and may not be available
int8 = ivy.IntDtype("int8")
int16 = ivy.IntDtype("int16")
int32 = ivy.IntDtype("int32")
int64 = ivy.IntDtype("int64")
uint8 = ivy.UintDtype("uint8")
uint16 = ivy.UintDtype("uint16")
uint32 = ivy.UintDtype("uint32")
uint64 = ivy.UintDtype("uint64")
bfloat16 = ivy.FloatDtype("bfloat16")
float16 = ivy.FloatDtype("float16")
float32 = ivy.FloatDtype("float32")
float64 = ivy.FloatDtype("float64")
complex64 = ivy.ComplexDtype("complex64")
complex128 = ivy.ComplexDtype("complex128")
bool = ivy.Dtype("bool")

# type aliases
char = int8
short = int16
int = int32
long = int64
half = float16
float = float32
double = float64

# data type promotion
torch_promotion_table = {
    (uint8, uint8): uint8,
    (uint8, int8): int16,
    (uint8, int16): int16,
    (uint8, int32): int32,
    (uint8, int64): int64,
    (uint8, float16): float16,
    (uint8, float32): float32,
    (uint8, float64): float64,
    (uint8, bool): uint8,
    (uint8, bfloat16): bfloat16,
    (uint8, complex64): complex64,
    (uint8, complex128): complex128,
    (int8, uint8): int16,
    (int8, int8): int8,
    (int8, int16): int16,
    (int8, int32): int32,
    (int8, int64): int64,
    (int8, float16): float16,
    (int8, float32): float32,
    (int8, float64): float64,
    (int8, bool): int8,
    (int8, bfloat16): bfloat16,
    (int8, complex64): complex64,
    (int8, complex128): complex128,
    (int16, uint8): int16,
    (int16, int8): int16,
    (int16, int16): int16,
    (int16, int32): int32,
    (int16, int64): int64,
    (int16, float16): float16,
    (int16, float32): float32,
    (int16, float64): float64,
    (int16, bool): int16,
    (int16, bfloat16): bfloat16,
    (int16, complex64): complex64,
    (int16, complex128): complex128,
    (int32, uint8): int32,
    (int32, int8): int32,
    (int32, int16): int32,
    (int32, int32): int32,
    (int32, int64): int64,
    (int32, float16): float16,
    (int32, float32): float32,
    (int32, float64): float64,
    (int32, bool): int32,
    (int32, bfloat16): bfloat16,
    (int32, complex64): complex64,
    (int32, complex128): complex128,
    (int64, uint8): int64,
    (int64, int8): int64,
    (int64, int16): int64,
    (int64, int32): int64,
    (int64, int64): int64,
    (int64, float16): float16,
    (int64, float32): float32,
    (int64, float64): float64,
    (int64, bool): int64,
    (int64, bfloat16): bfloat16,
    (int64, complex64): complex64,
    (int64, complex128): complex128,
    (float16, uint8): float16,
    (float16, int8): float16,
    (float16, int16): float16,
    (float16, int32): float16,
    (float16, int64): float16,
    (float16, float16): float16,
    (float16, float32): float32,
    (float16, float64): float64,
    (float16, bool): float16,
    (float16, bfloat16): float32,
    (float16, complex64): complex64,
    (float16, complex128): complex128,
    (float32, uint8): float32,
    (float32, int8): float32,
    (float32, int16): float32,
    (float32, int32): float32,
    (float32, int64): float32,
    (float32, float16): float32,
    (float32, float32): float32,
    (float32, float64): float64,
    (float32, bool): float32,
    (float32, bfloat16): float32,
    (float32, complex64): complex64,
    (float32, complex128): complex128,
    (float64, uint8): float64,
    (float64, int8): float64,
    (float64, int16): float64,
    (float64, int32): float64,
    (float64, int64): float64,
    (float64, float16): float64,
    (float64, float32): float64,
    (float64, float64): float64,
    (float64, bool): float64,
    (float64, bfloat16): float64,
    (float64, complex64): complex128,
    (float64, complex128): complex128,
    (bool, uint8): uint8,
    (bool, int8): int8,
    (bool, int16): int16,
    (bool, int32): int32,
    (bool, int64): int64,
    (bool, float16): float16,
    (bool, float32): float32,
    (bool, float64): float64,
    (bool, bool): bool,
    (bool, bfloat16): bfloat16,
    (bool, complex64): complex64,
    (bool, complex128): complex128,
    (bfloat16, uint8): bfloat16,
    (bfloat16, int8): bfloat16,
    (bfloat16, int16): bfloat16,
    (bfloat16, int32): bfloat16,
    (bfloat16, int64): bfloat16,
    (bfloat16, float16): float32,
    (bfloat16, float32): float32,
    (bfloat16, float64): float64,
    (bfloat16, bool): bfloat16,
    (bfloat16, bfloat16): bfloat16,
    (bfloat16, complex64): complex64,
    (bfloat16, complex128): complex128,
    (complex64, uint8): complex64,
    (complex64, int8): complex64,
    (complex64, int16): complex64,
    (complex64, int32): complex64,
    (complex64, int64): complex64,
    (complex64, float16): complex64,
    (complex64, float32): complex64,
    (complex64, float64): complex128,
    (complex64, bool): complex64,
    (complex64, bfloat16): complex64,
    (complex64, complex64): complex64,
    (complex64, complex128): complex128,
    (complex128, uint8): complex128,
    (complex128, int8): complex128,
    (complex128, int16): complex128,
    (complex128, int32): complex128,
    (complex128, int64): complex128,
    (complex128, float16): complex128,
    (complex128, float32): complex128,
    (complex128, float64): complex128,
    (complex128, bool): complex128,
    (complex128, bfloat16): complex128,
    (complex128, complex64): complex128,
    (complex128, complex128): complex128,
}


@handle_exceptions
def promote_types_torch(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
    /,
) -> ivy.Dtype:
    """
    Promote the datatypes type1 and type2, returning the data type they promote to.

    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote

    Returns
    -------
    ret
        The type that both input types promote to
    """
    try:
        ret = torch_promotion_table[(ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))]
    except KeyError:
        raise ivy.utils.exceptions.IvyException("these dtypes are not type promotable")
    return ret


@handle_exceptions
def promote_types_of_torch_inputs(
    *args: Union[ivy.Array, Number, Iterable[Number]],
) -> Tuple[ivy.Array, ivy.Array]:
    """
    Promote the dtype of the given native array inputs to a common dtype based on type
    promotion rules.

    While passing float or integer values or any other non-array input
    to this function, it should be noted that the return will be an
    array-like object. Therefore, outputs from this function should be
    used as inputs only for those functions that expect an array-like or
    tensor-like objects, otherwise it might give unexpected results.
    """
    # Ignore type of 0-dim arrays to mimic torch
    args = [ivy.asarray(x) for x in args]
    dtypes = tuple(ivy.default_dtype(item=x).strip("u123456789") for x in args)
    if len(set(dtypes)) == 1:
        target_non_empty_shape_index = ivy.nested_argwhere(
            args, lambda x: not x.shape == (), stop_after_n_found=1
        )
        non_empty_array_val = ivy.multi_index_nest(args, target_non_empty_shape_index)

        empty_shape_index = ivy.nested_argwhere(args, lambda x: x.shape == ())
        empty_array_vals = ivy.multi_index_nest(args, empty_shape_index)

        def nested_multi(x):
            x = ivy.asarray(
                x,
                dtype=non_empty_array_val.dtype,
                device=ivy.default_device(item=non_empty_array_val, as_native=False),
            )
            return x

        ivy.nested_map(empty_array_vals, nested_multi)
    else:
        for arg in args:
            promoted = promote_types_torch(arg.dtype, args[0].dtype)
            arg = ivy.asarray(arg, dtype=promoted)
    return args


from . import nn
from .nn.functional import softmax, relu
from . import tensor
from .tensor import *
from . import blas_and_lapack_ops
from .blas_and_lapack_ops import *
from . import comparison_ops
from .comparison_ops import *
from . import creation_ops
from .creation_ops import *
from . import dtype
from .dtype import *
from . import indexing_slicing_joining_mutating_ops
from .indexing_slicing_joining_mutating_ops import *
from . import locally_disabling_gradient_computation
from .locally_disabling_gradient_computation import *
from . import miscellaneous_ops
from .miscellaneous_ops import *
from . import pointwise_ops
from .pointwise_ops import *
from . import random_sampling
from .random_sampling import *
from . import reduction_ops
from .reduction_ops import *
from . import spectral_ops
from .spectral_ops import *
from . import tensor_functions
from .tensor_functions import *
from . import utilities
from .utilities import *
from . import linalg
from . import func
from .func import *


_frontend_array = tensor

# setting to specific version #
# --------------------------- #

if ivy.is_local():
    module = ivy.utils._importlib.import_cache[__name__]
else:
    module = sys.modules[__name__]

__version__ = set_frontend_to_specific_version(module)
