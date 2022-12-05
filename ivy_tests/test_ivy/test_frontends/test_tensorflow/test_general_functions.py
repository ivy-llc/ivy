# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import _matrix_rank_helper


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
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
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
)
def test_tensorflow_clip_by_value(
    *,
    input_and_ranges,
    as_variable,
    native_array,
    frontend,
    fn_tree,
    on_device,
    num_positional_args,
):
    x_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        t=x[0],
        clip_value_min=min,
        clip_value_max=max,
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
    as_variable,
    native_array,
    with_out,
    frontend,
    fn_tree,
    on_device,
    num_positional_args,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        num_rows=n_rows,
        num_columns=n_cols,
        batch_shape=batch_shape,
        dtype=dtype[0],
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
)
def test_tensorflow_ones(
    shape,
    dtype,
    as_variable,
    native_array,
    num_positional_args,
    frontend,
    fn_tree,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
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
    as_variable,
    with_out,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    eq, operands, _ = eq_n_op_n_shp
    kw = {}
    i = 0
    for x_ in operands:
        kw["x{}".format(i)] = x_
        i += 1
    # len(operands) + 1 because of the equation
    num_positional_args = len(operands) + 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        equation=eq,
        **kw,
    )


@st.composite
def _constant_helper(draw):
    x_dtype = draw(helpers.get_dtypes("valid", full=False))
    x_dtype, x = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
            shape=st.shared(helpers.get_shape(), key="value_shape"),
        ),
    )
    to_shape = draw(
        helpers.reshape_shapes(shape=st.shared(helpers.get_shape(), key="value_shape")),
    )
    cast_dtype = x_dtype[0]  # draw(
    #     helpers.get_dtypes("valid", full=False)
    #     .map(lambda t: t[0])
    #     .filter(lambda t: ivy.can_cast(x_dtype[0], t))
    # )
    return x_dtype, x, cast_dtype, to_shape


# constant
@handle_frontend_test(
    fn_tree="tensorflow.constant",
    all_args=_constant_helper(),
)
def test_tensorflow_constant(
    *,
    all_args,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    x_dtype, x, cast_dtype, to_shape = all_args
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        value=x[0],
        dtype=cast_dtype,
        shape=to_shape,
    )


@st.composite
def _convert_to_tensor_helper(draw):
    x_dtype = draw(helpers.get_dtypes("valid", full=False))
    x_dtype, x = draw(
        helpers.dtype_and_values(
            dtype=x_dtype,
        )
    )
    cast_dtype = x_dtype[0]  # draw(
    #     helpers.get_dtypes("valid", full=False)
    #     .map(lambda t: t[0])
    #     .filter(lambda t: ivy.can_cast(x_dtype[0], t))
    # )
    return x_dtype, x, cast_dtype


# convert_to_tensor
@handle_frontend_test(
    fn_tree="tensorflow.convert_to_tensor",
    dtype_x_cast=_convert_to_tensor_helper(),
    dtype_hint=helpers.get_dtypes("valid", full=False),
)
def test_tensorflow_convert_to_tensor(
    *,
    dtype_x_cast,
    dtype_hint,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    x_dtype, x, cast_dtype = dtype_x_cast
    helpers.test_frontend_function(
        input_dtypes=x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
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
)
def test_tensorflow_rank(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# ones_like
@handle_frontend_test(
    fn_tree="tensorflow.ones_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_tensorflow_ones_like(
    dtype_and_x,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        dtype=dtype[0],
    )


# zeros_like
@handle_frontend_test(
    fn_tree="tensorflow.zeros_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_tensorflow_zeros_like(
    dtype_and_x,
    dtype,
    as_variable,
    native_array,
    num_positional_args,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
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
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, value = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=value[0],
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
)
def test_tensorflow_concat(
    *,
    dtype_input_axis,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        values=x,
        axis=axis,
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
    as_variable,
    native_array,
    with_out,
    frontend,
    fn_tree,
    on_device,
    num_positional_args,
):
    helpers.test_frontend_function(
        shape=input,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
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
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        out_type=output_dtype,
    )


# range
@handle_frontend_test(
    fn_tree="tensorflow.range",
    start=helpers.ints(min_value=-50, max_value=0),
    limit=helpers.ints(min_value=1, max_value=50),
    delta=helpers.ints(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
)
def test_tensorflow_range(
    *,
    start,
    limit,
    delta,
    dtype,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        on_device=on_device,
        fn_tree=fn_tree,
        frontend=frontend,
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
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
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
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtypes, xs = dtype_x_v
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        sorted_sequence=np.sort(xs[0]),
        values=xs[1],
        side=side,
        out_type=out_type,
    )
