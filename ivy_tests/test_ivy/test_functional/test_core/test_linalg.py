"""Collection of tests for unified linear algebra functions."""

# global
import sys

import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


@st.composite
def dtype_value1_value2_axis(
    draw,
    available_dtypes,
    min_value=None,
    max_value=None,
    allow_inf=True,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
    specific_dim_size=3,
):
    # For cross product, a dim with size 3 is required
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    axis = draw(st.integers(0, len(shape)))
    # make sure there is a dim with specific dim size
    shape = list(shape)
    shape = shape[:axis] + [specific_dim_size] + shape[axis:]
    shape = tuple(shape)

    dtype = draw(st.sampled_from(available_dtypes))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                )
            )
        )

    value1, value2 = values[0], values[1]
    return dtype, value1, value2, axis


@st.composite
def _get_dtype_value1_value2_axis_for_tensordot(
    draw,
    available_dtypes,
    min_value=None,
    max_value=None,
    allow_inf=True,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
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
    axis = draw(st.integers(1, len(shape)))

    dtype = draw(st.sampled_from(available_dtypes))

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                )
            )
        )

    value1, value2 = values[0], values[1]
    value1 = np.asarray(value1, dtype=dtype)
    value2 = np.asarray(value2, dtype=dtype)
    if not isinstance(axis, list):
        value2 = value2.transpose(
            [k for k in range(len(shape) - axis, len(shape))]
            + [k for k in range(0, len(shape) - axis)]
        )
    return dtype, value1, value2, axis


@st.composite
def _get_dtype_and_matrix(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(st.shared(st.sampled_from(ivy_np.valid_float_dtypes)))
    random_size = draw(st.integers(min_value=2, max_value=4))
    batch_shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=3))
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [random_size, random_size]),
            min_value=2,
            max_value=5,
        )
    )


@st.composite
def _get_first_matrix_and_dtype(draw):
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(st.sampled_from(ivy_np.valid_numeric_dtypes), key="shared_dtype")
    )
    shared_size = draw(
        st.shared(st.integers(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(st.integers(min_value=2, max_value=4))
    batch_shape = draw(
        st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=3), key="shape")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [random_size, shared_size]),
            min_value=2,
            max_value=5,
        )
    )


@st.composite
def _get_second_matrix_and_dtype(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(st.sampled_from(ivy_np.valid_numeric_dtypes), key="shared_dtype")
    )
    shared_size = draw(
        st.shared(st.integers(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(st.integers(min_value=2, max_value=4))
    batch_shape = draw(
        st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=3), key="shape")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [shared_size, random_size]),
            min_value=2,
            max_value=5,
        )
    )


# vector_to_skew_symmetric_matrix
@st.composite
def _get_dtype_and_vector(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(st.sampled_from(ivy_np.valid_numeric_dtypes))
    batch_shape = draw(helpers.get_shape(min_num_dims=2, max_num_dims=4))
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [3]),
            min_value=2,
            max_value=5,
        )
    )


@given(
    dtype_x=_get_dtype_and_vector(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="vector_to_skew_symmetric_matrix"
    ),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_vector_to_skew_symmetric_matrix(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vector_to_skew_symmetric_matrix",
        vector=np.asarray(x, dtype=input_dtype),
    )


# matrix_power
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=50,
        shape=st.integers(2, 8).map(lambda x: tuple([x, x])),
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="matrix_power"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    n=st.integers(1, 8),
)
def test_matrix_power(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    n,
):
    dtype, x = dtype_x

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_power",
        x=np.asarray(x, dtype=dtype),
        n=n,
    )


# matmul
@given(
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="matmul"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
)
def test_matmul(
    x,
    y,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype1, x_1 = x
    input_dtype2, y_1 = y
    input_dtype = [input_dtype1, input_dtype2]
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matmul",
        x1=np.asarray(x_1, dtype=input_dtype1),
        x2=np.asarray(y_1, dtype=input_dtype2),
    )


# det
@given(
    dtype_x=_get_dtype_and_matrix(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="det"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_det(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="det",
        x=np.asarray(x, dtype=input_dtype),
    )


# eigh
@given(
    dtype_x=_get_dtype_and_matrix(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="eigh"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_eigh(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    x = np.asarray(x, dtype=input_dtype)
    results = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="eigh",
        x=x,
        test_values=False,
    )
    if results is None:
        return

    ret, ret_from_np = results
    # flattened array returns
    ret_np_flat, ret_from_np_flat = helpers.get_flattened_array_returns(
        ret=ret, ret_from_gt=ret_from_np
    )

    # value test
    for ret_np, ret_from_np in zip(ret_np_flat, ret_from_np_flat):
        helpers.assert_all_close(
            np.abs(ret_np), np.abs(ret_from_np), rtol=1e-2, atol=1e-2
        )


# eigvalsh
@given(
    dtype_x=_get_dtype_and_matrix(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="eigvalsh"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_eigvalsh(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="eigvalsh",
        x=np.asarray(x, dtype=input_dtype),
    )


# inv
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=50,
        shape=st.integers(2, 20).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="inv"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_inv(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="inv",
        x=np.asarray(x, dtype=input_dtype),
    )


# matrix_transpose
@given(
    dtype_x=_get_first_matrix_and_dtype(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="matrix_transpose"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_matrix_transpose(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_transpose",
        x=np.asarray(x, dtype=input_dtype),
    )


# outer
@given(
    dtype_xy=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=2,
        min_value=1,
        max_value=50,
        min_num_dims=1,
        max_num_dims=1,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="outer"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
)
def test_outer(
    dtype_xy,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    types, arrays = dtype_xy
    type1, type2 = types
    x1, x2 = arrays
    input_dtype = [type1, type2]
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="outer",
        x1=np.asarray(x1, input_dtype[0]),
        x2=np.asarray(x2, input_dtype[1]),
    )


# slogdet
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=50,
        shape=st.integers(2, 20).map(lambda x: tuple([x, x])),
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="slogdet"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_slogdet(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="slogdet",
        x=np.asarray(x, dtype=input_dtype),
    )


# solve
@st.composite
def _get_first_matrix(draw):
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(st.sampled_from(ivy_np.valid_float_dtypes), key="shared_dtype")
    )
    shared_size = draw(
        st.shared(st.integers(min_value=2, max_value=4), key="shared_size")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([shared_size, shared_size]),
            min_value=2,
            max_value=5,
        ).filter(lambda x: np.linalg.cond(x) < 1 / sys.float_info.epsilon)
    )


@st.composite
def _get_second_matrix(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(st.sampled_from(ivy_np.valid_float_dtypes), key="shared_dtype")
    )
    shared_size = draw(
        st.shared(st.integers(min_value=2, max_value=4), key="shared_size")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype, shape=tuple([shared_size, 1]), min_value=2, max_value=5
        )
    )


@given(
    x=_get_first_matrix(),
    y=_get_second_matrix(),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="solve"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
)
def test_solve(
    x,
    y,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype1, x1 = x
    input_dtype2, x2 = y
    input_dtype = [input_dtype1, input_dtype2]

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="solve",
        x1=np.asarray(x1, dtype=input_dtype1),
        x2=np.asarray(x2, dtype=input_dtype2),
    )


# svdvals
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=50,
        min_num_dims=2,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="svdvals"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_svdvals(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="svdvals",
        x=np.asarray(x, dtype=input_dtype),
    )


# tensordot
@given(
    dtype_x1_x2_axis=_get_dtype_value1_value2_axis_for_tensordot(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=3,
        max_num_dims=8,
        min_dim_size=1,
        max_dim_size=15,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="tensordot"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
)
def test_tensordot(
    dtype_x1_x2_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    (
        dtype,
        x1,
        x2,
        axis,
    ) = dtype_x1_x2_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tensordot",
        x1=x1,
        x2=x2,
        axes=axis,
    )


# trace
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=50,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="trace"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    offset=st.integers(-10, 10),
)
def test_trace(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    offset,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="trace",
        x=np.asarray(x, dtype=dtype),
        offset=offset,
    )


# vecdot
@given(
    dtype_x1_x2_axis=dtype_value1_value2_axis(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=6,
        min_dim_size=1,
        max_dim_size=10,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="vecdot"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
)
def test_vecdot(
    dtype_x1_x2_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x1, x2, axis = dtype_x1_x2_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vecdot",
        x1=np.asarray(x1, dtype=dtype),
        x2=np.asarray(x2, dtype=dtype),
        axis=axis,
    )


# vector_norm
@given(
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=5,
        min_axis=-2,
        max_axis=1,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="vector_norm"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    kd=st.booleans(),
    ord=st.integers(1, 2),
)
def test_vector_norm(
    dtype_values_axis,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    kd,
    ord,
):
    dtype, x, axis = dtype_values_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vector_norm",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
        keepdims=kd,
        ord=ord,
    )


# pinv
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="pinv"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    rtol=st.floats(1e-5, 1e-3),
)
def test_pinv(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    rtol,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="pinv",
        x=np.asarray(x, dtype=dtype),
        rtol=rtol,
    )


# qr
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="qr"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    mode=st.sampled_from(("reduced", "complete")),
)
def test_qr(
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    mode,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="qr",
        x=np.asarray(x, dtype=dtype),
        mode=mode,
    )


# svd
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="svd"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    fm=st.booleans(),
)
def test_svd(
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    fm,
):
    dtype, x = dtype_x

    results = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="svd",
        x=np.asarray(x, dtype=dtype),
        full_matrices=fm,
        test_values=False,
    )
    if results is None:
        return

    ret, ret_from_np = results
    # flattened array returns
    ret_np_flat, ret_from_np_flat = helpers.get_flattened_array_returns(
        ret=ret, ret_from_gt=ret_from_np
    )

    # value test
    for ret_np, ret_from_np in zip(ret_np_flat, ret_from_np_flat):
        num_cols = ret_np.shape[-2]
        for col_idx in range(num_cols):
            ret_np_col = ret_np[..., col_idx, :]
            ret_np_col = np.where(ret_np_col[..., 0:1] < 0, ret_np_col * -1, ret_np_col)
            ret_from_np_col = ret_from_np[..., col_idx, :]
            ret_from_np_col = np.where(
                ret_from_np_col[..., 0:1] < 0, ret_from_np_col * -1, ret_from_np_col
            )
            helpers.assert_all_close(ret_np_col, ret_from_np_col, rtol=1e-1, atol=1e-1)


# matrix_norm
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="matrix_norm"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    kd=st.booleans(),
    ord=st.integers(1, 2) | st.sampled_from(("fro", "nuc")),
)
def test_matrix_norm(
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    kd,
    ord,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_norm",
        x=np.asarray(x, dtype=dtype),
        ord=ord,
        keepdims=kd,
    )


# matrix_rank
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes[1:],
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=3,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="matrix_rank"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    rtol=st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
)
def test_matrix_rank(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    rtol,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_rank",
        x=np.asarray(x, dtype=dtype),
        rtol=1e-04,
    )


# cholesky
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=10,
        shape=st.integers(2, 5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1])) != 0
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="cholesky"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    upper=st.booleans(),
)
def test_cholesky(
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    upper,
):
    dtype, x = dtype_x
    x = np.asarray(x, dtype=dtype)
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cholesky",
        x=x,
        upper=upper,
    )


# cross
@given(
    dtype_x1_x2_axis=dtype_value1_value2_axis(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=3,
        max_dim_size=3,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="cross"),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
)
def test_cross(
    dtype_x1_x2_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x1, x2, axis = dtype_x1_x2_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cross",
        x1=np.asarray(x1, dtype=dtype),
        x2=np.asarray(x2, dtype=dtype),
        axis=axis,
    )


# diagonal
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=50,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="diagonal"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    offset=st.integers(-10, 50),
    axes=st.lists(st.integers(-2, 1), min_size=2, max_size=2, unique=True).filter(
        lambda axes: axes[0] % 2 != axes[1] % 2
    ),
)
def test_diagonal(
    dtype_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    offset,
    axes,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="diagonal",
        x=np.asarray(x, dtype=dtype),
        offset=offset,
        axis1=axes[0],
        axis2=axes[1],
    )
