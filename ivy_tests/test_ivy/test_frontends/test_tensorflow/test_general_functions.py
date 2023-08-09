# global
from hypothesis import strategies as st, assume
import numpy as np
from tensorflow import errors as tf_errors

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.tensorflow.general_functions import _num_to_bit_list
from ivy_tests.test_ivy.test_frontends.test_numpy.test_creation_routines.test_from_shape_or_value import (  # noqa : E501
    _input_fill_and_dtype,
)
from ivy_tests.test_ivy.test_frontends.test_tensorflow.test_tensor import (
    _array_and_shape,
)  # noqa : E501
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _matrix_rank_helper
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import (  # noqa
    _get_splits,
)


@st.composite
def _get_clip_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=-50, max_value=5)
    )
    max = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=6, max_value=50)
    )

    return x_dtype, x, min, max


# argsort
@handle_frontend_test(
    fn_tree="tensorflow.argsort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    direction=st.sampled_from(["ASCENDING", "DESCENDING"]),
)
def test_tensorflow_argsort(
    *,
    dtype_input_axis,
    direction,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        values=input[0],
        axis=axis,
        direction=direction,
    )


# clip_by_value
@handle_frontend_test(
    fn_tree="tensorflow.clip_by_value",
    input_and_ranges=_get_clip_inputs(),
    test_with_out=st.just(False),
)
def test_tensorflow_clip_by_value(
    *,
    input_and_ranges,
    frontend,
    test_flags,
    backend_fw,
    fn_tree,
    on_device,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        t=x[0],
        clip_value_min=min,
        clip_value_max=max,
    )


@st.composite
def _get_global_norm_clip_inputs(draw):
    t_list_dtype, t_list = draw(
        helpers.dtype_and_values(
            num_arrays=2,
            min_num_dims=1,
            shared_dtype=True,
            min_value=-100,
            max_value=100,
            dtype=["float32"] * 2,
        )
    )

    norm_dtype, norm = draw(
        helpers.dtype_and_values(
            shape=(1,),
            shared_dtype=True,
            min_value=0,
            exclude_min=True,
            max_value=100,
            dtype=["float32"],
        )
    )

    global_norm_dtype, global_norm = draw(
        helpers.dtype_and_values(
            shape=(1,),
            shared_dtype=True,
            min_value=0,
            exclude_min=True,
            max_value=100,
            dtype=["float32"],
        )
    )
    include_global = draw(st.booleans())
    if not include_global:
        global_norm_dtype, global_norm = None, None
    return t_list_dtype, t_list, norm_dtype, norm, global_norm_dtype, global_norm


# clip_by_global_norm
@handle_frontend_test(
    fn_tree="tensorflow.clip_by_global_norm",
    input_and_norm=_get_global_norm_clip_inputs(),
    test_with_out=st.just(False),
)
def test_tensorflow_clip_by_global_norm(
    *,
    input_and_norm,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    (
        t_list_dtype,
        t_list,
        norm_dtype,
        norm,
        global_norm_dtype,
        global_norm,
    ) = input_and_norm

    input_dtypes = [t_list_dtype[0], norm_dtype[0]]
    use_norm = None
    if global_norm_dtype:
        input_dtypes.append(global_norm_dtype[0])
        use_norm = global_norm[0]

    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        t_list=t_list,
        clip_norm=norm[0],
        use_norm=use_norm,
    )


@st.composite
def _get_norm_clip_inputs(draw):
    dtype = draw(helpers.get_dtypes("numeric", full=False))

    x_dtype, x, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=dtype,
            min_num_dims=1,
            min_value=-100,
            max_value=100,
            force_int_axis=True,
            valid_axis=True,
        )
    )
    norm_dtype, norm = draw(
        helpers.dtype_and_values(available_dtypes=dtype, shape=(1,))
    )
    return x_dtype[0], x, axis, norm


# clip_by_norm
@handle_frontend_test(
    fn_tree="tensorflow.clip_by_norm",
    input_and_norm=_get_norm_clip_inputs(),
    test_with_out=st.just(False),
)
def test_tensorflow_clip_by_norm(
    *,
    input_and_norm,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    x_dtype, x, axis, norm = input_and_norm
    helpers.test_frontend_function(
        input_dtypes=[x_dtype, x_dtype],
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        t=x[0],
        clip_norm=norm[0],
        axes=axis,
    )


# eye
@handle_frontend_test(
    fn_tree="tensorflow.eye",
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    batch_shape=st.lists(
        helpers.ints(min_value=1, max_value=10), min_size=1, max_size=2
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_tensorflow_eye(
    *,
    n_rows,
    n_cols,
    batch_shape,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        num_rows=n_rows,
        num_columns=n_cols,
        batch_shape=batch_shape,
        dtype=dtype[0],
    )


# foldl
@handle_frontend_test(
    fn_tree="tensorflow.foldl",
    fn=st.sampled_from(
        [
            lambda a, b: a + b,
            lambda a, b: a - b,
            lambda a, b: a * b,
        ],
    ),
    initializer=st.one_of(st.none(), st.floats(min_value=-1000, max_value=1000)),
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        min_value=-1000,
        max_value=1000,
        max_dim_size=10,
        max_num_dims=4,
        min_dim_size=1,
        min_num_dims=1,
    ),
    parallel_iterations=st.just(10),
    swap_memory=st.booleans(),
    name=st.none(),
)
def test_tensorflow_foldl(
    *,
    fn,
    initializer,
    dtype_and_values,
    frontend,
    backend_fw,
    fn_tree,
    test_flags,
    parallel_iterations,
    swap_memory,
    name,
):
    dtype, elems = dtype_and_values
    elems = np.atleast_1d(elems)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        fn=fn,
        elems=elems,
        initializer=initializer,
        backend_to_test=backend_fw,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        name=name,
        frontend=frontend,
        fn_tree=fn_tree,
        test_flags=test_flags,
    )


# ones
@handle_frontend_test(
    fn_tree="tensorflow.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_ones(
    shape,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )


# full
@handle_frontend_test(
    fn_tree="tensorflow.fill",
    shape=helpers.get_shape(),
    input_fill_dtype=_input_fill_and_dtype(),
)
def test_tensorflow_fill(
    shape,
    input_fill_dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, _, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-05,
        dims=shape,
        value=fill,
    )


# einsum
@handle_frontend_test(
    fn_tree="tensorflow.einsum",
    eq_n_op_n_shp=st.sampled_from(
        [
            ("ii", (np.arange(25).reshape(5, 5),), ()),
            ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
            ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,)),
        ]
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_tensorflow_einsum(
    *,
    eq_n_op_n_shp,
    dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    eq, operands, _ = eq_n_op_n_shp
    kw = {}
    i = 0
    for x_ in operands:
        kw["x{}".format(i)] = x_
        i += 1
    # len(operands) + 1 because of the equation
    test_flags.num_positional_args = len(operands) + 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        equation=eq,
        **kw,
    )


@st.composite
def _reshape_helper(draw):
    shape = draw(helpers.get_shape(min_num_dims=1))
    reshape_shape = draw(helpers.reshape_shapes(shape=shape))
    dtype = draw(helpers.array_dtypes(num_arrays=1))
    x = draw(helpers.array_values(dtype=dtype[0], shape=shape))
    return x, dtype, reshape_shape


# reshape
@handle_frontend_test(
    fn_tree="tensorflow.reshape",
    input_x_shape=_reshape_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_reshape(
    *,
    input_x_shape,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    x, x_dtype, shape = input_x_shape
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x,
        shape=shape,
    )


@st.composite
def _x_cast_dtype_shape(draw):
    x_dtype = draw(helpers.get_dtypes("valid", full=False))
    x_dtype, x = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=st.shared(helpers.get_shape(), key="value_shape"),
            large_abs_safety_factor=10,
            small_abs_safety_factor=10,
            safety_factor_scale="log",
        ),
    )
    to_shape = draw(
        helpers.reshape_shapes(shape=st.shared(helpers.get_shape(), key="value_shape")),
    )
    cast_dtype = x_dtype[0]
    # known tensorflow bug when trying to cast to a different type
    # https://github.com/tensorflow/tensorflow/issues/39554
    # cast_dtype = draw(
    #     helpers.get_dtypes("valid", full=False)
    #     .map(lambda t: t[0])
    #     .filter(lambda t: ivy.can_cast(x_dtype[0], t))
    # )
    return x_dtype, x, cast_dtype, to_shape


# size
# output_dtype not generated as tf only accepts tf dtypes
@handle_frontend_test(
    fn_tree="tensorflow.size",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), max_num_dims=4
    ),
    # output_dtype=st.sampled_from(["int32", "int64"]),
    test_with_out=st.just(False),
)
def test_tensorflow_size(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,  # output_dtype
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        # out_type=output_dtype,
    )


# constant
@handle_frontend_test(
    fn_tree="tensorflow.constant",
    all_args=_x_cast_dtype_shape(),
    test_with_out=st.just(False),
)
def test_tensorflow_constant(
    *,
    all_args,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtype, x, cast_dtype, to_shape = all_args
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x[0].tolist() if x[0].ndim > 0 else x[0].item(),
        dtype=cast_dtype,
        shape=to_shape,
    )


# convert_to_tensor
@handle_frontend_test(
    fn_tree="tensorflow.convert_to_tensor",
    dtype_x_cast=_x_cast_dtype_shape(),
    dtype_hint=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_convert_to_tensor(
    *,
    dtype_x_cast,
    dtype_hint,
    backend_fw,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    x_dtype, x, cast_dtype, _ = dtype_x_cast
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x[0],
        dtype=cast_dtype,
        dtype_hint=dtype_hint[0],
    )


# rank
@handle_frontend_test(
    fn_tree="tensorflow.rank",
    dtype_and_x=_matrix_rank_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_rank(
    *,
    dtype_and_x,
    backend_fw,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x, _ = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# ones_like
@handle_frontend_test(
    fn_tree="tensorflow.ones_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_ones_like(
    dtype_and_x,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dtype=dtype[0],
    )


# identity
@handle_frontend_test(
    fn_tree="tensorflow.identity",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_identity(
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# zeros_like
@handle_frontend_test(
    fn_tree="tensorflow.zeros_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_zeros_like(
    dtype_and_x,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dtype=dtype[0],
    )


# expand_dims
@handle_frontend_test(
    fn_tree="tensorflow.expand_dims",
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
def test_tensorflow_expand_dims(
    *,
    dtype_value,
    axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        axis=axis,
    )


# identity_n
@handle_frontend_test(
    fn_tree="tensorflow.identity_n",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), max_num_dims=5
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_identity_n(
    dtype_and_x,
    frontend,
    test_flags,
    backend_fw,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x,
    )


# Squeeze
@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="value_shape"))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0, None)
    return draw(st.sampled_from(valid_axes))


@handle_frontend_test(
    fn_tree="tensorflow.squeeze",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
)
def test_tensorflow_squeeze_general(
    *,
    dtype_value,
    axis,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, xs = dtype_value
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=xs[0],
        axis=axis,
    )


# concat
@handle_frontend_test(
    fn_tree="tensorflow.concat",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=st.integers(min_value=1, max_value=4),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_concat(
    *,
    dtype_input_axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        values=x,
        axis=axis,
    )


# cond
@handle_frontend_test(
    fn_tree="tensorflow.cond",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    pred_cond=st.booleans(),
    var=st.integers(min_value=1, max_value=100),
    test_with_out=st.just(False),
)
def test_tensorflow_cond(
    *,
    dtype_and_x,
    pred_cond,
    var,
    test_flags,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
):
    _test_true_fn = lambda: var + var

    _test_false_fn = lambda: var * var

    input_dtype, _ = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        pred=pred_cond,
        true_fn=_test_true_fn,
        false_fn=_test_false_fn,
    )


# zeros
@handle_frontend_test(
    fn_tree="tensorflow.zeros",
    input=helpers.get_shape(
        allow_none=False,
        min_num_dims=0,
        max_num_dims=10,
        min_dim_size=0,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_tensorflow_zeros(
    *,
    input,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        shape=input,
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
    )


# shape
@handle_frontend_test(
    fn_tree="tensorflow.shape",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    output_dtype=st.sampled_from(["int32", "int64"]),
)
def test_tensorflow_shape(
    *,
    dtype_and_x,
    output_dtype,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        out_type=output_dtype,
    )


@handle_frontend_test(
    fn_tree="tensorflow.shape_n",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), min_num_dims=1, max_num_dims=5
    ),
    output_dtype=st.sampled_from(["int32", "int64"]),
)
def test_tensorflow_shape_n(
    *,
    dtype_and_x,
    output_dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        out_type=output_dtype,
    )


@handle_frontend_test(
    fn_tree="tensorflow.ensure_shape",
    dtype_and_x=_array_and_shape(min_num_dims=0, max_num_dims=5),
)
def test_tensorflow_ensure_shape(
    *,
    dtype_and_x,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        x=x[0],
        shape=x[1],
    )


# range
@handle_frontend_test(
    fn_tree="tensorflow.range",
    start=helpers.ints(min_value=-50, max_value=0),
    limit=helpers.ints(min_value=1, max_value=50),
    delta=helpers.ints(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
    test_with_out=st.just(False),
)
def test_tensorflow_range(
    *,
    start,
    limit,
    delta,
    dtype,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        on_device=on_device,
        fn_tree=fn_tree,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        start=start,
        limit=limit,
        delta=delta,
        dtype=dtype[0],
    )


# sort
@handle_frontend_test(
    fn_tree="tensorflow.sort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    descending=st.sampled_from(["ASCENDING", "DESCENDING"]),
)
def test_tensorflow_sort(
    *,
    dtype_input_axis,
    descending,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        values=input[0],
        axis=axis,
        direction=descending,
    )


# searchsorted
@handle_frontend_test(
    fn_tree="tensorflow.searchsorted",
    dtype_x_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
    ),
    side=st.sampled_from(["left", "right"]),
    out_type=st.sampled_from(["int32", "int64"]),
)
def test_tensorflow_searchsorted(
    dtype_x_v,
    side,
    out_type,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs = dtype_x_v
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        sorted_sequence=np.sort(xs[0]),
        values=xs[1],
        side=side,
        out_type=out_type,
    )


# stack
@handle_frontend_test(
    fn_tree="tensorflow.stack",
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays"),
        shape=helpers.get_shape(min_num_dims=1),
        shared_dtype=True,
        valid_axis=True,
        allow_neg_axes=True,
        force_int_axis=True,
    ),
)
def test_tensorflow_stack(
    dtype_values_axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, values, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        values=values,
        axis=axis,
    )


# is_tensor
@handle_frontend_test(
    fn_tree="tensorflow.is_tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_tensorflow_is_tensor(
    *,
    dtype_and_x,
    backend_fw,
    frontend,
    test_flags,
    fn_tree,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        x=x[0],
    )


# gather
@handle_frontend_test(
    fn_tree="tensorflow.gather",
    params_indices_axis_batch_dims=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_tensorflow_gather(
    *,
    params_indices_axis_batch_dims,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtypes, params, indices, axis, batch_dims = params_indices_axis_batch_dims
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        params=params,
        indices=indices,
        axis=axis,
        batch_dims=batch_dims,
    )


# gather_nd
@handle_frontend_test(
    fn_tree="tensorflow.gather_nd",
    params_indices_axis_batch_dims=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        min_num_dims=5,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=5,
        indices_same_dims=False,
    ),
)
def test_tensorflow_gather_nd(
    *,
    params_indices_axis_batch_dims,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, params, indices, axis, batch_dims = params_indices_axis_batch_dims
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        params=params,
        indices=indices,
        batch_dims=batch_dims,
    )


@st.composite
def _pad_helper(draw):
    mode = draw(
        st.sampled_from(
            [
                "CONSTANT",
                "REFLECT",
                "SYMMETRIC",
            ]
        )
    )
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            ret_shape=True,
            min_num_dims=1,
            min_value=-100,
            max_value=100,
        )
    )
    ndim = len(shape)
    min_dim = min(shape)
    paddings = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=min_dim - 1),
                st.integers(min_value=0, max_value=min_dim - 1),
            ),
            min_size=ndim,
            max_size=ndim,
        )
    )
    constant_values = draw(st.integers(min_value=0, max_value=4))
    return dtype, input[0], paddings, mode, constant_values


# pad
@handle_frontend_test(
    fn_tree="tensorflow.pad",
    aliases=["tensorflow.compat.v1.pad"],
    dtype_and_values_and_other=_pad_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_pad(
    *,
    dtype_and_values_and_other,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, tensor, paddings, mode, constant_values = dtype_and_values_and_other
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=tensor,
        paddings=paddings,
        mode=mode,
        constant_values=constant_values,
    )


# transpose
@st.composite
def _get_perm_helper(draw):
    shape = draw(st.shared(helpers.get_shape(min_num_dims=1), key="shape"))
    dimensions = [x for x in range(len(shape))]
    perm = draw(st.permutations(dimensions))
    return perm


@handle_frontend_test(
    fn_tree="tensorflow.transpose",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    perm=_get_perm_helper(),
    conjugate=st.booleans(),
    test_with_out=st.just(False),
)
def test_tensorflow_transpose(
    *,
    dtype_and_x,
    perm,
    conjugate,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        perm=perm,
        conjugate=conjugate,
    )


@st.composite
def _strided_slice_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            ret_shape=True,
        ),
    )
    ndims = len(shape)
    masks = draw(
        st.lists(
            st.integers(min_value=0, max_value=2**ndims - 1), min_size=5, max_size=5
        ).filter(lambda x: bin(x[2])[2:].count("1") <= min(len(shape) - 1, 1))
    )
    begin, end, strides = [], [], []
    for i in shape:
        begin += [draw(st.integers(min_value=0, max_value=i - 1))]
        end += [draw(st.integers(min_value=0, max_value=i - 1))]
        if begin[-1] < end[-1]:
            strides += [draw(st.integers(min_value=1, max_value=i))]
        else:
            strides += [draw(st.integers(max_value=-1, min_value=-i))]
    ellipsis_mask = _num_to_bit_list(masks[2], ndims)
    for i, v in enumerate(ellipsis_mask):
        if v == 1:
            skip = draw(st.integers(min_value=0, max_value=ndims))
            begin, end, strides = map(
                lambda x: x[:i] + x[i + skip :] if i + skip < ndims else x[:i],
                [begin, end, strides],
            )
            break
    return dtype, x, np.array(begin), np.array(end), np.array(strides), masks


# strided_slice
@handle_frontend_test(
    fn_tree="tensorflow.strided_slice",
    dtype_x_params=_strided_slice_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_strided_slice(
    *,
    dtype_x_params,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, begin, end, strides, masks = dtype_x_params
    try:
        helpers.test_frontend_function(
            input_dtypes=dtype + 3 * ["int64"] + 5 * ["int32"],
            frontend=frontend,
            backend_to_test=backend_fw,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            input_=x[0],
            begin=begin,
            end=end,
            strides=strides,
            begin_mask=masks[0],
            end_mask=masks[1],
            ellipsis_mask=masks[2],
            new_axis_mask=masks[3],
            shrink_axis_mask=masks[4],
        )
    except tf_errors.InvalidArgumentError:
        assume(False)
    except Exception as e:
        if (
            hasattr(e, "message")
            and "only stride 1 allowed on non-range indexing" in e.message
        ):
            assume(False)
        raise e


@st.composite
def _slice_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            min_num_dims=1,
            ret_shape=True,
        ),
    )
    begin, size = [], []
    for i in shape:
        begin += [draw(st.integers(min_value=0, max_value=i - 1))]
        size += [draw(st.integers(min_value=0, max_value=i - begin[-1]))]
    return dtype, x, np.array(begin), np.array(size)


# slice
@handle_frontend_test(
    fn_tree="tensorflow.slice",
    dtype_x_params=_slice_helper(),
    test_with_out=st.just(False),
)
def test_tensorflow_slice(
    *,
    dtype_x_params,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x, begin, size = dtype_x_params
    helpers.test_frontend_function(
        input_dtypes=dtype + 3 * ["int64"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input_=x[0],
        begin=begin,
        size=size,
    )


@st.composite
def _linspace_helper(draw):
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=0,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
    )

    dtype = draw(st.sampled_from(["float32", "float64"]))

    # Param: start
    start = draw(
        helpers.array_values(
            dtype=dtype,
            shape=shape,
            min_value=-5.0,
            max_value=5.0,
        ),
    )

    # Param:stop
    stop = draw(
        helpers.array_values(
            dtype=dtype,
            shape=shape,
            min_value=-4.0,
            max_value=10.0,
        ),
    )

    return [dtype] * 2, start, stop


# linspace
@handle_frontend_test(
    fn_tree="tensorflow.linspace",
    dtype_and_params=_linspace_helper(),
    num=helpers.ints(min_value=2, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=0),
)
def test_tensorflow_linspace(
    *,
    dtype_and_params,
    num,
    axis,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, start, stop = dtype_and_params
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        start=start,
        stop=stop,
        num=num,
        axis=axis,
        on_device=on_device,
    )


# no_op
@handle_frontend_test(
    fn_tree="tensorflow.no_op",
    dtype=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_no_op(
    *,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
    )


# realdiv
@handle_frontend_test(
    fn_tree="tensorflow.realdiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-20,
        max_value=20,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_realdiv(
    *,
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    # todo: test for complex numbers
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# tile
@st.composite
def _multiple_shape_helper(draw):
    input_dtype, input_array, input_shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"), ret_shape=True
        )
    )
    input_dims = len(input_shape)

    dt_n_multiples = draw(
        helpers.dtype_and_values(
            available_dtypes=["int32", "int64"],
            min_value=0,
            max_value=10,
            shape=draw(
                helpers.get_shape(
                    min_num_dims=1,
                    max_num_dims=1,
                    min_dim_size=input_dims,
                    max_dim_size=input_dims,
                )
            ),
        )
    )
    return input_dtype, input_array, dt_n_multiples


@handle_frontend_test(fn_tree="tensorflow.tile", all_arguments=_multiple_shape_helper())
def test_tensorflow_tile(
    *,
    all_arguments,
    test_flags,
    frontend,
    fn_tree,
    on_device,
    backend_fw,
):
    input_dtype, input_matrix, dt_and_multiples = all_arguments
    dt_mul, multiples = dt_and_multiples
    helpers.test_frontend_function(
        input_dtypes=input_dtype + dt_mul,
        input=input_matrix[0],
        multiples=multiples[0],
        test_flags=test_flags,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
    )


# one_hot
@handle_frontend_test(
    fn_tree="tensorflow.one_hot",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
        min_value=0,
        max_value=10,
    ),
)
def test_tensorflow_one_hot(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    fn_tree,
    test_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    depth = 10
    helpers.test_frontend_function(
        input_dtypes=["uint8", "int32", "int64"],
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        indices=x[0],
        depth=depth,
    )


@st.composite
def _boolean_mask_helper(draw):
    tensor_shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=3,
            max_num_dims=5,
            min_dim_size=1,
            max_dim_size=10,
        ),
    )

    dtype = draw(st.sampled_from(["float32", "float64"]))

    # Param: tensor
    # tensor = draw(
    #     helpers.array_values(
    #         dtype=dtype, shape=tensor_shape, min_value=-5.0, max_value=5.0
    #     ),
    # )

    dtype, tensor, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=[dtype],
            shape=tensor_shape,
            min_value=-5.0,
            max_value=5.0,
            force_int_axis=True,
            valid_axis=True,
        )
    )
    mask_dim = draw(helpers.ints(min_value=1, max_value=len(tensor_shape) - axis))
    mask_shape = tensor_shape[axis : mask_dim + axis]

    # Param:stop
    mask = draw(
        helpers.array_values(
            allow_nan=False,
            dtype="bool",
            shape=mask_shape,
        ),
    )
    return [dtype[0], "bool"], tensor, mask, axis


# boolean_mask
@handle_frontend_test(
    fn_tree="tensorflow.boolean_mask",
    dtype_and_values=_boolean_mask_helper(),
)
def test_tensorflow_boolean_mask(
    *,
    dtype_and_values,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, tensor, mask, axis = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=tensor[0],
        mask=mask,
        axis=axis,
    )


# where
@handle_frontend_test(
    fn_tree="tensorflow.where",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
        min_value=0,
        max_value=10,
        min_num_dims=1,
    ),
)
def test_tensorflow_where_no_xy(
    *,
    dtype_and_input,
    frontend,
    backend_fw,
    fn_tree,
    test_flags,
    on_device,
):
    input_dtype, [condition] = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        condition=condition,
    )


# where
@handle_frontend_test(
    fn_tree="tensorflow.where",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("bool"),
        num_arrays=3,
        min_value=0,
        max_value=10,
        min_num_dims=1,
    ),
    dim_remove_from_x=st.integers(),
    dim_remove_from_y=st.integers(),
)
def test_tensorflow_where_with_xy(
    *,
    dtype_and_input,
    dim_remove_from_x,
    dim_remove_from_y,
    frontend,
    backend_fw,
    fn_tree,
    test_flags,
    on_device,
):
    input_dtype, [condition, x, y] = dtype_and_input
    if input_dtype != ["bool", "bool", "bool"]:
        return
    for _ in range(min(len(x.shape) - 1, dim_remove_from_x)):
        x = x[0]
    for _ in range(min(len(y.shape) - 1, dim_remove_from_y)):
        y = y[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        condition=condition,
        x=x,
        y=y,
    )


# roll
@handle_frontend_test(
    fn_tree="tensorflow.roll",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    shift=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_tuple=True,
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_tuple=True,
    ),
)
def test_tensorflow_roll(
    *,
    dtype_and_values,
    shift,
    axis,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_and_values
    if isinstance(shift, int) and isinstance(axis, tuple):
        axis = axis[0]
    if isinstance(shift, tuple) and isinstance(axis, tuple):
        if len(shift) != len(axis):
            mn = min(len(shift), len(axis))
            shift = shift[:mn]
            axis = axis[:mn]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
        shift=shift,
        axis=axis,
    )


# split
@handle_frontend_test(
    fn_tree="tensorflow.split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    num_or_size_splits=_get_splits(allow_none=False, min_num_dims=1),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_split(
    *,
    dtype_value,
    num_or_size_splits,
    axis,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=value[0],
        num_or_size_splits=num_or_size_splits,
        axis=axis,
    )


# repeat
@handle_frontend_test(
    fn_tree="tensorflow.repeat",
    dtypes_and_value_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        max_num_dims=5,
        max_dim_size=10,
        valid_axis=True,
        force_int_axis=True,
    ),
    repeats=helpers.ints(min_value=1, max_value=5),
)
def test_tensorflow_repeat(
    *,
    dtypes_and_value_and_axis,
    repeats,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    input_dtypes, x, axis = dtypes_and_value_and_axis
    repeats = repeats
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        repeats=repeats,
        axis=axis,
    )


# unstack
@handle_frontend_test(
    fn_tree="tensorflow.unstack",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        max_dim_size=1,
    ),
    number_positional_args=st.just(1),
    axis=st.integers(-1, 0),
    test_with_out=st.just(False),
)
def test_tensorflow_unstack(
    *,
    dtypes_values,
    axis,
    on_device,
    fn_tree,
    backend_fw,
    frontend,
    test_flags,
):
    x_dtype, x = dtypes_values
    axis = axis
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x[0],
        axis=axis,
    )


# reverse
@st.composite
def reverse_helper(draw):
    dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_num_dims=1,
            max_num_dims=8,
            ret_shape=True,
        )
    )
    axis_dtype, axis = draw(
        helpers.dtype_and_values(
            available_dtypes=["int32", "int64"],
            min_num_dims=1,
            max_num_dims=1,
            min_value=-(len(shape) - 1),
            max_value=len(shape) - 1,
            shape=(1,),
        )
    )
    return dtype, x, axis_dtype, axis


@handle_frontend_test(
    fn_tree="tensorflow.reverse",
    dtype_x_axis=reverse_helper(),
)
def test_tensorflow_reverse(
    *,
    dtype_x_axis,
    frontend,
    backend_fw,
    fn_tree,
    test_flags,
    on_device,
):
    dtype, x, axis_dtype, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=dtype + axis_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
        axis=axis[0],
    )


# scan
@handle_frontend_test(
    fn_tree="tensorflow.scan",
    dtypes_values=helpers.dtype_and_values(
        available_dtypes=["float32"], num_arrays=1, min_num_dims=2, max_dim_size=3
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_scan(
    *,
    dtypes_values,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    def _test_fn(a, x):
        return a + x

    x_dtype, elems = dtypes_values
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        fn=_test_fn,
        elems=elems[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.norm",
    aliases=["tensorflow.norm"],
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=4,
        min_axis=-3,
        max_axis=2,
    ),
    ord=st.sampled_from([1, 2, np.inf]),
    keepdims=st.booleans(),
)
def test_tensorflow_norm(
    *,
    dtype_values_axis,
    ord,
    keepdims,
    backend_fw,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, axis = dtype_values_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
        ord=ord,
        axis=axis,
        keepdims=keepdims,
    )


@handle_frontend_test(
    fn_tree="tensorflow.unique",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=["int64", "int32"],
        min_value=1,
        max_value=100,
        min_dim_size=1,
        max_dim_size=10,
        min_num_dims=1,
        max_num_dims=1,
    ),
    test_with_out=st.just([False]),
)
def test_tensorflow_unique(
    *,
    dtype_x,
    frontend,
    backend_fw,
    fn_tree,
    test_flags,
    on_device,
):
    dtype, x = dtype_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        x=x[0],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
    )


@handle_frontend_test(
    fn_tree="tensorflow.unique_with_counts",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            )
        ),
        dtype=["int32", "int64"],
    ),
    output_dtype=st.sampled_from(["int32", "int64"]),
)
def test_tensorflow_unique_with_counts(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
    output_dtype,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        x=x[0],
        input_dtypes=input_dtype,
        out_idx=output_dtype,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        frontend=frontend,
        test_flags=test_flags,
        on_device=on_device,
    )


@handle_frontend_test(
    fn_tree="tensorflow.while_loop",
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=1,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=10,
        shared_dtype=True,
        min_value=-100,
        max_value=100,
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_tensorflow_while_loop(
    *,
    dtype_and_x,
    test_flags,
    on_device,
    backend_fw,
    fn_tree,
    frontend,
):
    def _test_cond_fn(x):
        def any_negative_real(arr):
            for elem in arr:
                if isinstance(elem, (int, float)) and elem < 0:
                    return True
                elif isinstance(elem, complex):
                    return False
                elif isinstance(elem, (list, tuple)):
                    if any_negative_real(elem):
                        return True
            return False

        return any_negative_real(x)

    def _test_body_fn(x):
        return x + 1

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        cond=_test_cond_fn,
        body=_test_body_fn,
        loop_vars=(x[0],),
    )


# truncatediv
@handle_frontend_test(
    fn_tree="tensorflow.truncatediv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-20,
        max_value=20,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_truncatediv(
    *,
    dtype_and_x,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


# Truncatemod
@handle_frontend_test(
    fn_tree="tensorflow.truncatemod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_truncatemod(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[0], 0)))
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        y=x[1],
    )


@handle_frontend_test(
    fn_tree="tensorflow.unravel_index",
    indices=helpers.array_values(
        dtype=helpers.get_dtypes("integer"), shape=(1, 2), min_value=0, max_value=49
    ),
    dims=helpers.array_values(
        dtype=helpers.get_dtypes("integer"), shape=(1, 2), min_value=50
    ),
)
def test_tensorflow_unravel_index(
    *, indices, dims, frontend, test_flags, fn_tree, on_device, backend_fw
):
    helpers.test_frontend_function(
        input_dtypes=["int32"],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        indices=indices[0],
        dims=dims[0],
    )

@handle_frontend_test(
    fn_tree="tensorflow.zeros_initializer",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_zeros_initializer(
    shape,
    dtype,
    frontend,
    backend_fw,
    test_flags,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )