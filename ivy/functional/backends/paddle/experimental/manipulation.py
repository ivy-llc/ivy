from collections import namedtuple
from typing import Optional, Union, Sequence, Tuple, NamedTuple, List
from numbers import Number
from .. import backend_version
from ivy.func_wrapper import with_unsupported_device_and_dtypes
import paddle
import ivy
import ivy.functional.backends.paddle as paddle_backend

# Code from cephes for i0

_i0A = [
    -4.41534164647933937950e-18,
    3.33079451882223809783e-17,
    -2.43127984654795469359e-16,
    1.71539128555513303061e-15,
    -1.16853328779934516808e-14,
    7.67618549860493561688e-14,
    -4.85644678311192946090e-13,
    2.95505266312963983461e-12,
    -1.72682629144155570723e-11,
    9.67580903537323691224e-11,
    -5.18979560163526290666e-10,
    2.65982372468238665035e-9,
    -1.30002500998624804212e-8,
    6.04699502254191894932e-8,
    -2.67079385394061173391e-7,
    1.11738753912010371815e-6,
    -4.41673835845875056359e-6,
    1.64484480707288970893e-5,
    -5.75419501008210370398e-5,
    1.88502885095841655729e-4,
    -5.76375574538582365885e-4,
    1.63947561694133579842e-3,
    -4.32430999505057594430e-3,
    1.05464603945949983183e-2,
    -2.37374148058994688156e-2,
    4.93052842396707084878e-2,
    -9.49010970480476444210e-2,
    1.71620901522208775349e-1,
    -3.04682672343198398683e-1,
    6.76795274409476084995e-1,
]

_i0B = [
    -7.23318048787475395456e-18,
    -4.83050448594418207126e-18,
    4.46562142029675999901e-17,
    3.46122286769746109310e-17,
    -2.82762398051658348494e-16,
    -3.42548561967721913462e-16,
    1.77256013305652638360e-15,
    3.81168066935262242075e-15,
    -9.55484669882830764870e-15,
    -4.15056934728722208663e-14,
    1.54008621752140982691e-14,
    3.85277838274214270114e-13,
    7.18012445138366623367e-13,
    -1.79417853150680611778e-12,
    -1.32158118404477131188e-11,
    -3.14991652796324136454e-11,
    1.18891471078464383424e-11,
    4.94060238822496958910e-10,
    3.39623202570838634515e-9,
    2.26666899049817806459e-8,
    2.04891858946906374183e-7,
    2.89137052083475648297e-6,
    6.88975834691682398426e-5,
    3.36911647825569408990e-3,
    8.04490411014108831608e-1,
]


def moveaxis(
    a: paddle.Tensor,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if isinstance(source, tuple):
        source = list(source)
    if isinstance(destination, tuple):
        source = list(destination)
    if a.dtype in [paddle.int8, paddle.int16, paddle.uint8]:
        return paddle.moveaxis(a.cast("float32"), source, destination).cast(a.dtype)
    return paddle.moveaxis(a, source, destination)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "float16",
                "complex64",
                "complex128",
                "bool",
            )
        }
    },
    backend_version,
)
def heaviside(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return paddle.heaviside(x1, x2)


def flipud(
    m: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if copy:
        m = m.clone()
    if m.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        return paddle.flip(m.cast("float32"), axis=0).cast(m.dtype)
    return paddle.flip(m, axis=0)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("int16", "float16")}},
    backend_version,
)
def vstack(
    arrays: Sequence[paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        if arrays[0].ndim >= 2:
            return ivy.concat(arrays, axis=0)
        else:
            return ivy.stack(arrays, axis=0)


def hstack(
    arrays: Sequence[paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        if arrays[0].ndim >= 2:
            return ivy.concat(arrays, axis=1)
        else:
            return ivy.concat(arrays, axis=0)


def rot90(
    m: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    k: Optional[int] = 1,
    axes: Optional[Tuple[int, int]] = (0, 1),
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if copy:
        m = m.clone()
    if (k % 4) and m.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        return paddle.rot90(m.cast("float32"), k=k, axes=axes).cast(m.dtype)
    return paddle.rot90(m, k=k, axes=axes)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("complex64", "complex128")}},
    backend_version,
)
def top_k(
    x: paddle.Tensor,
    k: int,
    /,
    *,
    axis: int = -1,
    largest: Optional[bool] = True,
    sorted: bool = True,
    out: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    k = min(k, x.shape[axis])
    topk_res = NamedTuple(
        "top_k", [("values", paddle.Tensor), ("indices", paddle.Tensor)]
    )
    with ivy.ArrayMode(False):
        indices = ivy.argsort(x, axis=axis, descending=largest)
        indices = paddle.index_select(indices, paddle.arange(end=k), axis)
        if not sorted:
            indices = paddle.sort(indices, axis=axis)
        val = ivy.take_along_axis(x, indices, axis)
        return topk_res(val, indices)


def fliplr(
    m: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if copy:
        m = m.clone()
    if m.dtype in [paddle.int8, paddle.int16, paddle.uint8, paddle.float16]:
        return paddle.flip(m.cast("float32"), axis=1).cast(m.dtype)
    return paddle.flip(m, axis=1)


def i0(
    x: paddle.Tensor,
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    def _i0_1(x):
        return paddle_backend.multiply(
            paddle_backend.exp(x),
            _chbevl(paddle_backend.subtract(paddle_backend.divide(x, 2.0), 2.0), _i0A),
        )

    def _i0_2(x):
        return paddle_backend.divide(
            paddle_backend.multiply(
                paddle_backend.exp(x),
                _chbevl(
                    paddle_backend.subtract(paddle_backend.divide(32.0, x), 2.0), _i0B
                ),
            ),
            paddle_backend.sqrt(x),
        )

    def _chbevl(x, vals):
        b0 = vals[0]
        b1 = 0.0

        for i in range(1, len(vals)):
            b2 = b1
            b1 = b0
            b0 = paddle_backend.add(
                paddle_backend.subtract(paddle_backend.multiply(x, b1), b2), vals[i]
            )
        return paddle_backend.multiply(0.5, paddle_backend.subtract(b0, b2))

    x = paddle_backend.abs(x)
    return paddle_backend.where(paddle_backend.less_equal(x, 8.0), _i0_1(x), _i0_2(x))


def flatten(
    x: paddle.Tensor,
    /,
    *,
    copy: Optional[bool] = None,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    order: Optional[str] = "C",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if copy:
        x = x.clone()
    if x.ndim == 0:
        return x

    def _flatten(x, start_dim, end_dim):
        if x.dtype in [
            paddle.float16,
            paddle.complex64,
            paddle.complex128,
            paddle.bool,
        ]:
            if paddle.is_complex(x):
                return paddle.complex(
                    paddle.flatten(x.real(), start_axis=start_dim, stop_axis=end_dim),
                    paddle.flatten(x.imag(), start_axis=start_dim, stop_axis=end_dim),
                )
            return paddle.flatten(
                x.cast("float32"), start_axis=start_dim, stop_axis=end_dim
            ).cast(x.dtype)
        return paddle.flatten(x, start_axis=start_dim, stop_axis=end_dim)

    if order == "F":
        with ivy.ArrayMode(False):
            x = ivy.permute_dims(x, list(reversed(range(x.ndim))))
            ret = _flatten(x, start_dim, end_dim)
            return ivy.permute_dims(ret, list(reversed(range(ret.ndim))))
    return _flatten(x, start_dim, end_dim)


def vsplit(
    ary: paddle.Tensor,
    indices_or_sections: Union[int, Sequence[int], paddle.Tensor],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[paddle.Tensor]:
    if ary.ndim < 2:
        raise ivy.exceptions.IvyError(
            "vsplit only works on arrays of 2 or more dimensions"
        )
    return ivy.split(ary, copy=copy, num_or_size_splits=indices_or_sections, axis=0)


def dsplit(
    ary: paddle.Tensor,
    indices_or_sections: Union[int, Sequence[int], paddle.Tensor],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[paddle.Tensor]:
    if ary.ndim < 3:
        raise ivy.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=2)


def atleast_1d(
    *arys: paddle.Tensor, copy: Optional[bool] = False
) -> List[paddle.Tensor]:
    res = []
    for ary in arys:
        ary = ivy.array(ary, copy=copy).data
        if ary.ndim < 1:
            with ivy.ArrayMode(False):
                res.append(ivy.expand_dims(ary, axis=0))
        else:
            res.append(ary)
    if len(res) == 1:
        return res[0]
    return res


def dstack(
    arrays: Sequence[paddle.Tensor],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        arrays = ivy.atleast_2d(*arrays)
        if not isinstance(arrays, list):
            arrays = [arrays]
        if arrays[0].ndim < 3:
            return ivy.stack(arrays, axis=-1)
        else:
            return ivy.concat(arrays, axis=2)


def atleast_2d(
    *arys: paddle.Tensor, copy: Optional[bool] = False
) -> List[paddle.Tensor]:
    res = []
    for ary in arys:
        ary = ivy.array(ary, copy=copy).data
        if ary.ndim < 2:
            with ivy.ArrayMode(False):
                res.append(ivy.expand_dims(ary, axis=list(range(2 - ary.ndim))))
        else:
            res.append(ary)
    if len(res) == 1:
        return res[0]
    return res


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("float16",)}},
    backend_version,
)
def atleast_3d(
    *arys: Union[paddle.Tensor, bool, Number], copy: Optional[bool] = False
) -> List[paddle.Tensor]:
    res = []
    for ary in arys:
        ary = ivy.array(ary, copy=copy).data
        if ary.ndim == 0:
            result = ary.reshape((1, 1, 1))
        elif ary.ndim == 1:
            result = ary[None, :, None]
        elif ary.ndim == 2:
            result = ary[:, :, None]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("int8",)}},
    backend_version,
)
def take_along_axis(
    arr: paddle.Tensor,
    indices: paddle.Tensor,
    axis: int,
    /,
    *,
    mode: str = "fill",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if arr.ndim != indices.ndim:
        raise ivy.utils.exceptions.IvyException(
            "arr and indices must have the same number of dimensions;"
            + f" got {arr.ndim} vs {indices.ndim}"
        )
    indices = indices.cast("int64")
    if mode not in ["clip", "fill", "drop"]:
        raise ValueError(
            f"Invalid mode '{mode}'. Valid modes are 'clip', 'fill', 'drop'."
        )
    arr_shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    if mode == "clip":
        max_index = arr.shape[axis] - 1
        with ivy.ArrayMode(False):
            indices = ivy.clip(indices, 0, max_index)
    elif mode in ("fill", "drop"):
        if "float" in str(arr.dtype) or "complex" in str(arr.dtype):
            fill_value = float("nan")
        elif "uint" in str(arr.dtype):
            fill_value = paddle.iinfo(arr.dtype).max
        elif "int" in str(arr.dtype):
            fill_value = -paddle.iinfo(arr.dtype).max - 1
        else:
            raise TypeError(
                f"Invalid dtype '{arr.dtype}'. Valid dtypes are 'float', 'complex',"
                " 'uint', 'int'."
            )

        with ivy.ArrayMode(False):
            indices = ivy.where(
                (indices < 0) | (indices >= arr.shape[axis]), -1, indices
            )
            arr_shape = list(arr_shape)
            arr_shape[axis] = 1
            fill_arr = ivy.full(arr_shape, fill_value, dtype=arr.dtype)
            arr = ivy.concat([arr, fill_arr], axis=axis)
            indices = ivy.where(indices < 0, arr.shape[axis] + indices, indices)

    if arr.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
        paddle.complex64,
        paddle.complex128,
        paddle.bool,
    ]:
        if paddle.is_complex(arr):
            return paddle.complex(
                paddle.take_along_axis(arr.real(), indices, axis),
                paddle.take_along_axis(arr.imag(), indices, axis),
            )
        return paddle.take_along_axis(arr.cast("float32"), indices, axis).cast(
            arr.dtype
        )
    return paddle.take_along_axis(arr, indices, axis)


def hsplit(
    ary: paddle.Tensor,
    indices_or_sections: Union[int, Tuple[int, ...]],
    /,
    *,
    copy: Optional[bool] = None,
) -> List[paddle.Tensor]:
    if ary.ndim == 1:
        return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=0)
    return ivy.split(ary, num_or_size_splits=indices_or_sections, axis=1)


def broadcast_shapes(*shapes: Union[List[int], List[Tuple]]) -> Tuple[int]:
    def _broadcast_shape(s1, s2):
        len_1 = len(s1)
        len_2 = len(s2)
        if len_1 == 0:
            return () if len_2 == 0 else s2
        elif len_1 != 0 and len_2 == 0:
            return s1
        else:
            return paddle.broadcast_shape(s1, s2)

    if len(shapes) == 0:
        raise ValueError("shapes=[] must be non-empty")
    elif len(shapes) == 1:
        return shapes[0]
    result = _broadcast_shape(shapes[0], shapes[1])
    for i in range(2, len(shapes)):
        result = _broadcast_shape(result, shapes[i])

    return tuple(result)


def expand(
    x: paddle.Tensor,
    shape: Union[List[int], List[Tuple]],
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    shape = list(shape)

    for i, dim in enumerate(shape):
        if dim < 0:
            shape[i] = x.shape[i]
    if x.ndim == 0:
        if len(shape) == 0:
            return x
        else:
            x = ivy.expand_dims(x, 0)
    if x.ndim > len(shape):
        x = x.reshape([-1])

    if x.dtype in [
        paddle.int8,
        paddle.int16,
        paddle.uint8,
        paddle.float16,
    ]:
        return paddle.expand(x.cast("float32"), shape).cast(x.dtype)

    elif x.dtype in [paddle.complex64, paddle.complex128]:
        x_real = paddle.expand(ivy.real(x).data, shape)
        x_imag = paddle.expand(ivy.imag(x).data, shape)
        return x_real + 1j * x_imag
    else:
        return paddle.expand(x, shape)


def concat_from_sequence(
    input_sequence: Union[Tuple[paddle.Tensor], List[paddle.Tensor]],
    /,
    *,
    new_axis: int = 0,
    axis: int = 0,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    with ivy.ArrayMode(False):
        if new_axis == 0:
            return ivy.concat(input_sequence, axis=axis)
        elif new_axis == 1:
            return ivy.stack(input_sequence, axis=axis)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("int8", "int16", "uint8")}}, backend_version
)
def unique_consecutive(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    Results = namedtuple(
        "Results",
        ["output", "inverse_indices", "counts"],
    )
    x_shape = None
    if axis is None:
        x_shape = x.shape
        x = x.flatten()
        axis = -1
    if axis < 0:
        axis += x.ndim
    split_indices = paddle.flatten(
        paddle.where(
            ivy.current_backend().any(
                paddle.abs(paddle.diff(x, axis=axis)) > 1e-50,
                axis=tuple(i for i in paddle.arange(x.ndim) if i != axis),
            )
        )[0]
        + 1,
    )
    if len(split_indices) > 0:
        split_sizes = (
            [split_indices[0]]
            + [
                split_indices[i] - split_indices[i - 1]
                for i in range(1, len(split_indices))
            ]
            + [x.shape[axis] - split_indices[-1]]
        )
        sub_arrays = paddle.split(
            x,
            split_sizes,
            axis=axis,
        )
    else:
        sub_arrays = [x]
    output = paddle.concat(
        [
            ivy.current_backend().unique_all(sub_array, axis=axis)[0]
            for sub_array in sub_arrays
        ],
        axis=axis,
    )
    counts = paddle.to_tensor([sub_array.shape[axis] for sub_array in sub_arrays])
    inverse_indices = paddle.repeat_interleave(paddle.arange(len(counts)), counts)
    if x_shape:
        inverse_indices = paddle.reshape(inverse_indices, x_shape)
    return Results(
        output.astype(x.dtype),
        inverse_indices,
        counts,
    )
