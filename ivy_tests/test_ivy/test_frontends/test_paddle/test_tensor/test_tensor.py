# global
import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.paddle import Tensor
from ivy_tests.test_ivy.helpers import assert_all_close
from ivy_tests.test_ivy.helpers import handle_frontend_method, BackendHandler
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa E501
    _get_dtype_values_k_axes_for_rot90,
)
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _statistical_dtype_values,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_blas_and_lapack_ops import (
    _get_dtype_and_3dbatch_matrices,
)
from ivy_tests.test_ivy.test_frontends.test_paddle.test_manipulation import (
    _tile_helper,
)

CLASS_TREE = "ivy.functional.frontends.paddle.Tensor"


# --- Helpers --- #
# --------------- #


@st.composite
def _array_and_shape(
    draw,
    *,
    min_num_dims=1,
    max_num_dims=3,
    min_dim_size=1,
    max_dim_size=10,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)

    available_dtypes = draw(helpers.get_dtypes("numeric"))
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    dtype.append("int32")
    shape = draw(
        st.shared(
            helpers.get_shape(
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            ),
            key="shape",
        )
    )
    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
        )
    )
    to_shape = [(None if draw(st.booleans()) else _) for _ in shape]

    return dtype, [array, to_shape]


def _filter_query(query):
    return (
        query.ndim > 1
        if isinstance(query, np.ndarray)
        else (
            not any(isinstance(i, np.ndarray) and i.ndim <= 1 for i in query)
            if isinstance(query, tuple)
            else True
        )
    )


# as_complex
@st.composite
def _get_as_complex_inputs_(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )

    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=(*shape, 2),
            min_value=0,
            max_value=50,
        )
    )
    return x_dtype, x


# clip
@st.composite
def _get_clip_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=1, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
            min_value=0,
            max_value=50,
        )
    )
    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=(1,), min_value=0, max_value=25)
    )
    max = draw(
        helpers.array_values(dtype=x_dtype[0], shape=(1,), min_value=26, max_value=50)
    )
    if draw(st.booleans()):
        min = None
    elif draw(st.booleans()):
        max = None
    return x_dtype, x, min, max


# clip_
@st.composite
def _get_clip_inputs_(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=1, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
            min_value=0,
            max_value=50,
        )
    )
    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=(1,), min_value=0, max_value=25)
    )
    max = draw(
        helpers.array_values(dtype=x_dtype[0], shape=(1,), min_value=26, max_value=50)
    )
    return x_dtype, x, min, max


@st.composite
def _get_dtype_and_3dbatch_matrices_for_matmul(draw):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    batch_size = draw(helpers.ints(min_value=2, max_value=4))
    transpose_x = draw(st.booleans())
    transpose_y = draw(st.booleans())

    mat1_shape = (
        (batch_size, dim_size1, shared_size)
        if not transpose_x
        else (batch_size, shared_size, dim_size1)
    )
    mat2_shape = (
        (batch_size, shared_size, dim_size2)
        if not transpose_y
        else (batch_size, dim_size2, shared_size)
    )

    mat1 = draw(
        helpers.array_values(dtype=dtype[0], shape=mat1_shape, min_value=2, max_value=5)
    )
    mat2 = draw(
        helpers.array_values(dtype=dtype[0], shape=mat2_shape, min_value=2, max_value=5)
    )
    return dtype, mat1, mat2, transpose_x, transpose_y


# cond
@st.composite
def _get_dtype_and_matrix_non_singular(draw, dtypes):
    while True:
        matrix = draw(
            helpers.dtype_and_values(
                available_dtypes=dtypes,
                min_value=-10,
                max_value=10,
                min_num_dims=2,
                max_num_dims=2,
                min_dim_size=1,
                max_dim_size=5,
                shape=st.tuples(st.integers(1, 5), st.integers(1, 5)).filter(
                    lambda x: x[0] == x[1]
                ),
                allow_inf=False,
                allow_nan=False,
            )
        )
        if np.linalg.det(matrix[1][0]) != 0:
            break

    return matrix[0], matrix[1]


@st.composite
def _get_dtype_and_multiplicative_matrices(draw):
    return draw(
        st.one_of(
            _get_dtype_input_and_matrices_for_matmul(),
            _get_dtype_and_3dbatch_matrices_for_matmul(),
        )
    )


@st.composite
def _get_dtype_and_square_matrix(draw):
    dim_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    mat = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size, dim_size), min_value=0, max_value=10
        )
    )
    return dtype, mat


# bmm helper function
@st.composite
def _get_dtype_and_values_bmm(draw):
    # arrays x and y of sizes (b, m, k) and (b, k, n) respectively
    b = draw(helpers.ints(min_value=1, max_value=10))
    k = draw(helpers.ints(min_value=1, max_value=10))
    m = draw(helpers.ints(min_value=1, max_value=10))
    n = draw(helpers.ints(min_value=1, max_value=10))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    x = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(b, m, k), min_value=-10, max_value=10
        )
    )
    y = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(b, k, n), min_value=-10, max_value=10
        )
    )
    return dtype, x, y


# lerp helper function
@st.composite
def _get_dtype_and_values_for_lerp(draw):
    is_tensor = draw(st.booleans())
    if is_tensor:
        input_dtype, x = draw(
            helpers.dtype_and_values(
                num_arrays=3,
                available_dtypes=helpers.get_dtypes("valid"),
                shared_dtype=True,
            )
        )
        return input_dtype, x[0], x[1], x[2]
    else:
        input_dtype, x = draw(
            helpers.dtype_and_values(
                num_arrays=2,
                available_dtypes=helpers.get_dtypes("valid"),
                shared_dtype=True,
            )
        )
        weight = draw(st.floats())
        return input_dtype, x[0], x[1], weight


@st.composite
def _get_dtype_input_and_matrices_for_matmul(draw):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = draw(helpers.ints(min_value=2, max_value=5))
    shared_size = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    transpose_x = draw(st.booleans())
    transpose_y = draw(st.booleans())

    mat1_shape = (shared_size, dim_size1) if transpose_x else (dim_size1, shared_size)
    mat2_shape = (dim_size2, shared_size) if transpose_y else (shared_size, dim_size2)

    mat1 = draw(
        helpers.array_values(dtype=dtype[0], shape=mat1_shape, min_value=2, max_value=5)
    )
    mat2 = draw(
        helpers.array_values(dtype=dtype[0], shape=mat2_shape, min_value=2, max_value=5)
    )
    return dtype, mat1, mat2, transpose_x, transpose_y


@st.composite
def _get_dtype_value1_value2_cov(
    draw,
    available_dtypes,
    min_num_dims,
    max_num_dims,
    min_dim_size,
    max_dim_size,
    abs_smallest_val=None,
    min_value=None,
    max_value=None,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    large_abs_safety_factor=4,
    small_abs_safety_factor=4,
    safety_factor_scale="log",
):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )

    dtype = draw(st.sampled_from(draw(available_dtypes)))

    values = []
    for i in range(1):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    abs_smallest_val=abs_smallest_val,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                    large_abs_safety_factor=large_abs_safety_factor,
                    small_abs_safety_factor=small_abs_safety_factor,
                    safety_factor_scale=safety_factor_scale,
                )
            )
        )

    value = values[0]

    # modifiers: rowVar, bias, ddof
    rowVar = draw(st.booleans())
    ddof = draw(st.booleans())

    numVals = None
    if rowVar is False:
        numVals = -1 if numVals == 0 else 0
    else:
        numVals = 0 if len(shape) == 1 else -1

    fweights = draw(
        helpers.array_values(
            dtype="int64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
        )
    )

    aweights = draw(
        helpers.array_values(
            dtype="float64",
            shape=shape[numVals],
            abs_smallest_val=1,
            min_value=1,
            max_value=10,
            allow_inf=False,
            small_abs_safety_factor=1,
        )
    )

    return [dtype], value, rowVar, ddof, fweights, aweights


@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=1,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=3,
        )
    )

    reshape_shape = draw(helpers.reshape_shapes(shape=shape))

    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )
    return dtypes, x, reshape_shape


# expand helper function
@st.composite
def dtypes_x_shape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            min_dim_size=1,
            min_num_dims=1,
            available_dtypes=["float32"],
            shape=st.shared(
                helpers.get_shape(
                    min_num_dims=1,
                    max_num_dims=6,
                ),
                key="shape",
            ),
        )
    )
    shape = draw(
        st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=6,
            ),
            key="shape",
        )
    )
    return dtypes, x, shape


# --- Main --- #
# ------------ #


# __add__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___add__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__float__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
    ),
)
def test_paddle___float__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    # Numpy doesn't support complex to float conversion
    assume(not np.issubdtype(input_dtypes[0], np.complexfloating))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_paddle___floordiv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1,
    )


# __ge__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__ge__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___ge__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__getitem__",
    dtype_x_index=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ).filter(lambda x: x[0][0] == x[0][-1] and _filter_query(x[-2])),
)
def test_paddle___getitem__(
    dtype_x_index,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, index = dtype_x_index
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"item": index},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __gt__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___gt__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__int__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=-1e15,
        max_value=1e15,
    ),
)
def test_paddle___int__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    # Numpy doesn't support complex to int conversion
    assume(not np.issubdtype(input_dtypes[0], np.complexfloating))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# invert
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__invert__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        max_num_dims=0,
    ),
)
def test_paddle___invert__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __le__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__le__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___le__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __len__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__len__",
    dtype_and_x=_array_and_shape(
        min_num_dims=1,
        max_num_dims=5,
    ),
)
def test_paddle___len__(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# long
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__long__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        max_num_dims=0,
        min_value=-1e15,
        max_value=1e15,
    ),
)
def test_paddle___long__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    frontend,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    # Numpy doesn't support complex to int conversion
    assume(not np.issubdtype(input_dtypes[0], np.complexfloating))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={
            "object": xs[0],
        },
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __lt__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___lt__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __mod__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        min_value=0,
        exclude_min=True,
    ),
)
def test_paddle___mod__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __mul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___mul__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __ne__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_paddle___ne__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __neg__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_paddle___neg__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __or__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___or__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __radd__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___radd__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__rdiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
)
def test_paddle___rdiv__(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __rmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__rmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_paddle___rmul__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __rsub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle___rsub__(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "x": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__rtruediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
)
def test_paddle___rtruediv__(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# __setitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__setitem__",
    dtypes_x_index_val=helpers.dtype_array_query_val(
        available_dtypes=helpers.get_dtypes("valid"),
    ).filter(lambda x: x[0][0] == x[0][-1] and _filter_query(x[-2])),
)
def test_paddle___setitem__(
    dtypes_x_index_val,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, index, val = dtypes_x_index_val
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"item": index, "value": val},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __sub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle___sub__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __xor__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__xor__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle___xor__(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# abs
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_abs(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="acosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_acosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# add_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="add_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
    test_inplace=st.just(True),
)
def test_paddle_add_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="add_n",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=helpers.ints(min_value=1, max_value=5),
        shared_dtype=True,
    ),
)
def test_paddle_add_n(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"inputs": x},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"inputs": x},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# addmm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="addmm",
    dtype_input_xy=_get_dtype_and_3dbatch_matrices(with_input=True, input_3d=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_paddle_addmm(
    *,
    dtype_input_xy,
    beta,
    alpha,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, input, x, y = dtype_input_xy
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": input[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"x": x[0], "y": y[0], "beta": beta, "alpha": alpha},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# all
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="all",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("bool"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_all(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# allclose
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="allclose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    # rtol=1e-05,
    # atol=1e-08,
    # equal_nan=st.booleans(),
)
def test_paddle_allclose(
    dtype_and_x,
    # rtol,
    # atol,
    # equal_nan,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            # "rtol": rtol,
            # "atol": atol,
            # "equal_nan": equal_nan,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="angle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["float64", "complex64", "complex128"],
    ),
)
def test_paddle_angle(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# any
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="any",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_any(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# argmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="argmax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_argmax(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# argmin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="argmin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("valid")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_argmin(
    dtype_x_axis,
    keep_dims,
    on_device,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# argsort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="argsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    descending=st.booleans(),
)
def test_paddle_argsort(
    dtype_x_axis,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "descending": descending,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# as_complex
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="as_complex",
    dtypes_and_x=_get_as_complex_inputs_(),
)
def test_paddle_as_complex(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# as_real
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="as_real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
    ),
)
def test_paddle_as_real(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="asin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_asin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="asinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_asinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# astype
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="astype",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    dtype=st.one_of(helpers.get_dtypes("valid")),
)
def test_paddle_astype(
    dtype_and_x,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    if dtype is None:
        dtype = input_dtype
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dtype": dtype,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="atan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_atan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_and",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_bitwise_and(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


#  bitwise_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_bitwise_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_or",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_bitwise_or(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_xor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bitwise_xor",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_bitwise_xor(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bmm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="bmm",
    dtype_and_x=_get_dtype_and_values_bmm(),
)
def test_paddle_bmm(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, y = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": y},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cast
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cast",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_paddle_cast(
    dtype_and_x,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    if dtype is None:
        dtype = input_dtype
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dtype": dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ceil
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_ceil(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ceil_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="ceil_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_inplace=st.just(True),
)
def test_paddle_ceil_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cholesky
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cholesky",
    dtype_and_x=_get_dtype_and_square_matrix(),
    upper=st.booleans(),
)
def test_paddle_cholesky(
    dtype_and_x,
    upper,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    x = np.matmul(x.T, x) + np.identity(x.shape[0])

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"upper": upper},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="clip",
    input_and_ranges=_get_clip_inputs(),
)
def test_paddle_clip(
    input_and_ranges,
    frontend,
    frontend_method_data,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# clip_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="clip_",
    input_and_ranges=_get_clip_inputs_(),
    test_inplace=st.just(True),
)
def test_paddle_clip_(
    input_and_ranges,
    frontend,
    frontend_method_data,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, min_val, max_val = input_and_ranges
    if min_val > max_val:
        max_value = min_val
        min_value = max_val
    else:
        max_value = max_val
        min_value = min_val

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min_value, "max": max_value},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cond",
    dtype_and_x=_get_dtype_and_matrix_non_singular(dtypes=["float32", "float64"]),
    p=st.sampled_from([None, "fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2]),
)
def test_paddle_cond(
    dtype_and_x,
    p,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"p": p},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# conj
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_paddle_conj(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_cos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_cosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cumprod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
)
def test_paddle_cumprod(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim": axis},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cumsum",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        valid_axis=True,
        force_int_axis=True,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
)
def test_paddle_cumsum(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"axis": axis},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# deg2rad
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="deg2rad",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_deg2rad(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# Tests #
# ----- #


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_device(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.place, ivy.dev(ivy.array(data[0])), as_array=False
    )


# diagonal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="diagonal",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dims_and_offset=helpers.dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
)
def test_paddle_diagonal(
    dtype_and_values,
    dims_and_offset,
    frontend,
    frontend_method_data,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, value = dtype_and_values
    dim1, dim2, offset = dims_and_offset
    input = value[0]
    num_dims = len(np.shape(input))
    assume(dim1 != dim2)
    if dim1 < 0:
        assume(dim1 + num_dims != dim2)
    if dim2 < 0:
        assume(dim1 != dim2 + num_dims)
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"x": input},
        method_input_dtypes=[input_dtype[0]],
        method_all_as_kwargs_np={
            "offset": offset,
            "axis1": dim1,
            "axis2": dim2,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# digamma
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="digamma",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
        max_value=1e5,
    ),
)
def test_paddle_digamma(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# dim
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="dim",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_dim(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# divide
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="divide",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        safety_factor_scale="log",
        small_abs_safety_factor=32,
    ),
)
def test_paddle_divide(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_dtype(
    dtype_x,
):
    dtype, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.dtype, dtype[0], as_array=False)


# eigvals
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="eigvals",
    dtype_and_x=_get_dtype_and_square_matrix(),
)
def test_paddle_eigvals(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x

    ret, frontend_ret = helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        test_values=False,
    )

    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        # check if Tensor or ivy array
        try:
            ret = ret.ivy_array.to_numpy()
        except AttributeError:
            ret = ivy_backend.to_numpy(ret)
        frontend_ret = [np.asarray(x) for x in frontend_ret]
        # Calculate the magnitude of the complex numbers then sort them for testing
        ret = np.sort(np.abs(ret)).astype(np.float64)
        frontend_ret = np.sort(np.abs(frontend_ret)).astype(np.float64)

        assert_all_close(
            ret_np=ret,
            ret_from_gt_np=frontend_ret,
            backend=backend_fw,
            ground_truth_backend=frontend,
            atol=1e-2,
            rtol=1e-2,
        )


# equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="equal",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_equal(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


#  equal_all
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="equal_all",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-np.inf,
        max_value=np.inf,
        shared_dtype=True,
        safety_factor_scale="log",
        small_abs_safety_factor=32,
    ),
)
def test_paddle_equal_all(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# erf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="erf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_erf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# exp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="exp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_exp(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# exp_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="exp_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_paddle_exp_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fill_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="fill_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_inf=False,
    ),
    dtype_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=(1,),
        min_value=0,
        max_value=10,
    ),
)
def test_paddle_fill_(
    dtype_and_x,
    dtype_v,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    value_dtype, v = dtype_v
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=value_dtype,
        method_all_as_kwargs_np={"value": v[0].item()},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# floor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="floor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_floor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# floor_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="floor_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_paddle_floor_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# floor_divide
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="floor_divide",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=2,
        shared_dtype=True,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="linear",
    ),
)
def test_paddle_floor_divide(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    # Absolute tolerance is 1,
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="fmax",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_fmax(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="fmin",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_fmin(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# frac
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="frac",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="valid"),
        num_arrays=1,
        max_value=1e6,
        min_value=-1e6,
    ),
)
def test_paddle_frac(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# gather
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="gather",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_gather(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater_than
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="greater_than",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        safety_factor_scale="log",
        small_abs_safety_factor=32,
    ),
)
def test_paddle_greater_than(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# imag
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="imag",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_imag(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# inner
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="inner",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_inner(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# is_complex
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="is_complex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_is_complex(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# is_floating_point
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="is_floating_point",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=["int16", "int32", "int64", "float32", "float64"],
    ),
)
def test_paddle_is_floating_point(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# is_tensor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="is_tensor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
    ),
)
def test_paddle_is_tensor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# isclose
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isclose",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_isclose(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


#  isfinite
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isfinite",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_isfinite(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


#  isinf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_isinf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# isnan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="isnan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_isnan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# lerp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="lerp",
    dtypes_and_x=_get_dtype_and_values_for_lerp(),
)
def test_paddle_lerp(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, y, weight = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": y,
            "weight": weight,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# lerp_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="lerp_",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=3, shared_dtype=True
    ),
    test_inplace=st.just(True),
)
def test_paddle_lerp_(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
            "weight": x[2],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="less_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_paddle_less_equal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


#  less_than
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="less_than",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_less_than(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_log(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log10
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="log10",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_log10(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_and",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_logical_and(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"self": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


#  logical_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_logical_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_or
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_or",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_logical_or(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_xor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="logical_xor",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_logical_xor(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# max
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="max",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("valid")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=False,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_max(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# mean
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="mean",
    dtype_x_axis=_statistical_dtype_values(function="mean"),
    keepdim=st.booleans(),
)
def test_paddle_mean(
    dtype_x_axis,
    keepdim,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis, *_ = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keepdim,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        backend_to_test=backend_fw,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_minimum(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="mod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
        min_value=0,
        exclude_min=True,
    ),
)
def test_paddle_mod(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_multiply(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_ndim(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(x.ndim, data[0].ndim, as_array=False)


# neg
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="neg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_paddle_neg(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# nonzero
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="nonzero",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="float"),
        min_num_dims=1,
        allow_inf=True,
    ),
)
def test_paddle_nonzero(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# not_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="not_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_paddle_not_equal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "x": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="numel",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
    ),
)
def test_paddle_numel(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# numpy
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="numpy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        min_dim_size=2,
    ),
)
def test_paddle_numpy(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# pow
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="pow",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
)
def test_paddle_pow(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# prod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="prod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        min_value=-5,
        max_value=5,
        valid_axis=True,
        force_int_axis=True,
        allow_inf=False,
    ),
    keep_dims=st.booleans(),
)
def test_paddle_prod(
    dtype_x_axis,
    keep_dims,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
            "keepdim": keep_dims,
            "dtype": x[0].dtype,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# rad2deg
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="rad2deg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_rad2deg(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="real",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="valid"),
        num_arrays=2,
        min_num_dims=1,
        allow_inf=True,
    ),
)
def test_paddle_real(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reciprocal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_reciprocal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reciprocal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reciprocal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_reciprocal_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# remainder
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="remainder",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_paddle_remainder(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# remainder_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="remainder_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_inplace=st.just(True),
)
def test_paddle_remainder_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reshape
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reshape",
    dtype_x_shape=_reshape_helper(),
)
def test_paddle_reshape(
    dtype_x_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, shape = dtype_x_shape
    assume(len(shape) != 0)
    shape = {
        "shape": shape,
    }
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=shape,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reshape_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="reshape_",
    dtype_x_shape=_reshape_helper(),
)
def test_paddle_reshape_(
    dtype_x_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, shape = dtype_x_shape
    assume(len(shape) != 0)
    shape = {
        "shape": shape,
    }
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=shape,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# rot90
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="rot90",
    dtype_m_k_axes=_get_dtype_values_k_axes_for_rot90(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=6,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_paddle_rot90(
    dtype_m_k_axes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, values, k, axes = dtype_m_k_axes

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": values,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "k": k,
            "axes": axes,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# round_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="round_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_round_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# rsqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="rsqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_rsqrt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# rsqrt_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="rsqrt_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_inplace=st.just(True),
)
def test_paddle_rsqrt_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_paddle_shape(dtype_x):
    _, data, shape = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(
        x.ivy_array.shape, ivy.Shape(shape), as_array=False
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_sign(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_sin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_sinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=st.one_of(helpers.get_dtypes("float")),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    descending=st.booleans(),
)
def test_paddle_sort(
    dtype_x_axis,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "object": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
            "descending": descending,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# split
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="split",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_split(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_sqrt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sqrt_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="sqrt_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_inplace=st.just(True),
)
def test_paddle_sqrt_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# square
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="square",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_square(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# squeeze_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="squeeze_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    test_inplace=st.just(True),
)
def test_paddle_squeeze_(
    dtype_value,
    axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# stanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="stanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    scale_a=st.floats(1e-5, 1e5),
    scale_b=st.floats(1e-5, 1e5),
)
def test_paddle_stanh(
    dtype_and_x,
    frontend_method_data,
    scale_a,
    scale_b,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={
            "scale_a": scale_a,
            "scale_b": scale_b,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# std
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="std",
    dtype_and_x=_statistical_dtype_values(function="std"),
    keepdim=st.booleans(),
)
def test_paddle_std(
    dtype_and_x,
    keepdim,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
            "unbiased": bool(correction),
            "keepdim": keepdim,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        backend_to_test=backend_fw,
        method_flags=method_flags,
        on_device=on_device,
    )


# subtract
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="subtract",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_subtract(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# subtract_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="subtract_",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
    test_inplace=st.just(True),
)
def test_paddle_subtract_(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# t
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="t",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        max_num_dims=2,
    ),
)
def test_paddle_t(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_tanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tanh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="tanh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_tanh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __div__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="__div__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
)
def test_paddle_tensor__div__(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="acos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_paddle_tensor_acos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# add
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2, shared_dtype=True
    ),
)
def test_paddle_tensor_add(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# chunk
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="chunk",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        min_value=0.1,
        max_value=1e02,
        force_int_axis=True,
        valid_axis=True,
    ),
    chunks=st.integers(
        min_value=1,
        max_value=8,
    ),
)
def test_paddle_tensor_chunk(
    dtype_x_axis,
    chunks,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    is_remainder = x[0].shape[axis] % chunks != 0
    axis_solvable = len(x[0].shape) + axis < 0
    if is_remainder or axis_solvable:
        assume(False)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "chunks": chunks,
            "axis": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cov
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="cov",
    dtype_x1_corr_cov=_get_dtype_value1_value2_cov(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_paddle_tensor_cov(
    dtype_x1_corr_cov,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    dtype, x, rowvar, ddof, fweights, aweights = dtype_x1_corr_cov
    helpers.test_frontend_method(
        init_input_dtypes=["float64", "int64", "float64"],
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["int64", "float64"],
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={
            "rowvar": rowvar,
            "ddof": ddof,
            "fweights": fweights,
            "aweights": aweights,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        rtol_=1e-3,
        atol_=1e-3,
        frontend=frontend,
        on_device=on_device,
    )


# expand
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="expand",
    dtype_x_shape=dtypes_x_shape(),
)
def test_paddle_tensor_expand(
    dtype_x_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, shape = dtype_x_shape
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "shape": shape,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# flatten
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="flatten",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axes=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        min_size=2,
        max_size=2,
        unique=False,
        force_tuple=True,
    ),
)
def test_paddle_tensor_flatten(
    dtype_value,
    axes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "start_axis": axes[0],
            "stop_axis": axes[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# floor_mod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="floor_mod",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=2,
        shared_dtype=True,
    ),
)
def test_paddle_tensor_floor_mod(
    dtypes_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtypes_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"y": x[1]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="heaviside",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
)
def test_paddle_tensor_heaviside(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "x": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "y": x[1],
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


# matmul
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="matmul",
    dtype_tensor1_tensor2=_get_dtype_and_multiplicative_matrices(),
)
def test_paddle_tensor_matmul(
    dtype_tensor1_tensor2,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    dtype, tensor1, tensor2, transpose_x, transpose_y = dtype_tensor1_tensor2

    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": tensor1,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "y": tensor2,
            "transpose_x": transpose_x,
            "transpose_y": transpose_y,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# squeeze
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="squeeze",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_paddle_tensor_squeeze(
    dtype_value,
    axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tile
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="tile",
    dt_x_repeats=_tile_helper(),
)
def test_paddle_tensor_tile(
    dt_x_repeats,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtypes, x, repeats = dt_x_repeats
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "repeat_times": repeats,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        backend_to_test=backend_fw,
        method_flags=method_flags,
        on_device=on_device,
    )


# topk
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="topk",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    k=st.data(),
    sorted=st.booleans(),
    largest=st.booleans(),
)
def test_paddle_topk(
    dtype_x_and_axis,
    k,
    sorted,
    largest,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_and_axis
    k = k.draw(st.integers(min_value=1, max_value=x[0].shape[axis]))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "k": k,
            "axis": axis,
            "largest": largest,
            "sorted": sorted,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
    )


# trace
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="trace",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_num_dims=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    offset=st.integers(min_value=-1e04, max_value=1e04),
    axis1=st.integers(min_value=0, max_value=0),
    axis2=st.integers(min_value=1, max_value=1),
)
def test_paddle_trace(
    dtype_and_x,
    offset,
    axis1,
    axis2,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "value": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "offset": offset,
            "axis1": axis1,
            "axis2": axis2,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="trunc",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_paddle_trunc(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unbind
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="unbind",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=2,
        max_dim_size=1,
        force_int_axis=True,
        min_axis=-1,
        max_axis=0,
    ),
)
def test_paddle_unbind(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# unique_consecutive
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="unique_consecutive",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=4,
        max_dim_size=1,
        force_int_axis=True,
        min_axis=-1,
        max_axis=0,
    ),
)
def test_paddle_unique_consecutive(
    dtype_x_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# unsqueeze
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="unsqueeze",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_paddle_unsqueeze(
    dtype_value,
    axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unsqueeze_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="unsqueeze_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    test_inplace=st.just(True),
)
def test_paddle_unsqueeze_(
    dtype_value,
    axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# var
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="var",
    dtype_and_x=_statistical_dtype_values(function="var"),
    keepdim=st.booleans(),
)
def test_paddle_var(
    dtype_and_x,
    keepdim,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "axis": axis,
            "unbiased": bool(correction),
            "keepdim": keepdim,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        backend_to_test=backend_fw,
        method_flags=method_flags,
        on_device=on_device,
    )


# zero_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="paddle.to_tensor",
    method_name="zero_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_paddle_zero_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )
