"""Collection of tests for unified linear algebra functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# vector_to_skew_symmetric_matrix
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
)
def test_vector_to_skew_symmetric_matrix(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
):
    if "float16" or "int8" in input_dtype:
        return
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
        vector=np.random.uniform(size=(a, 3)).astype(input_dtype[0]),
    )


# matrix_power
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_float_dtypes), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    n=st.integers(-10, 10),
)
def test_matrix_power(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    n,
):
    if "float16" in input_dtype:
        return
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_power",
        test_rtol=1e-02,
        test_atol=1e-02,
        x=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
        n=n,
    )


# matmul
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=2
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
    c=st.integers(1, 50),
    seed=st.integers(0, 2**16 - 1),
)
def test_matmul(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    c,
    seed,
):
    np.random.seed(seed)
    if "float16" or "int8" in input_dtype:
        return
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
        test_rtol=5e-02,
        test_atol=5e-02,
        x1=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(b, c)).astype(input_dtype[1]),
    )


# det
@given(
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
)
def test_det(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
):
    if "float16" in input_dtype:
        return
    x = np.random.uniform(size=(b, a, a)).astype(input_dtype)
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
        test_rtol=1e-03,
        test_atol=1e-03,
        x=x,
    )


# eigh
@given(
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="eigh"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(2, 6),
)
def test_eigh(
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
):
    if "float16" in input_dtype:
        return
    x = np.random.uniform(size=(a, a)).astype(input_dtype)
    x = (x + x.T) / 2
    ret, ret_from_np = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="eigh",
        test_rtol=1e-02,
        test_atol=1e-02,
        x=x,
        test_values=False,
    )

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
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 5),
    b=st.integers(1, 5),
)
def test_eigvalsh(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
):
    if "float16" in input_dtype:
        return
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
        test_rtol=0.01,
        test_atol=0.01,
        x=np.random.uniform(size=(b, a, a)).astype(input_dtype),
    )


# inv
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_float_dtypes), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
)
def test_inv(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
):
    if "float16" in input_dtype:
        return
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
        test_rtol=1e-02,
        test_atol=1e-02,
        x=np.random.uniform(size=(b, a, a)).astype(input_dtype[0]),
    )


# matrix_transpose
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
)
def test_matrix_transpose(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
):
    if "float16" or "int8" in input_dtype:
        return
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
        x=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
    )


# outer
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=2
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
)
def test_outer(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
):
    if "float16" or "int8" in input_dtype:
        return
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
        x1=np.random.uniform(size=a).astype(input_dtype[0]),
        x2=np.random.uniform(size=b).astype(input_dtype[1]),
    )


# slogdet
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_float_dtypes), length=1
    ),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
)
def test_slogdet(
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
):
    if "float16" in input_dtype:
        return
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
        x=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
        test_rtol=1e-3,
        test_atol=1e-3,
    )


# solve
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_float_dtypes), length=2
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
)
def test_solve(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
):
    if "float16" in input_dtype:
        return
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
        test_rtol=1e-03,
        test_atol=1e-03,
        x1=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(a, 1)).astype(input_dtype[1]),
    )


# svdvals
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
)
def test_svdvals(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
):
    if "float16" or "int8" in input_dtype:
        return
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
        x=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
    )


# tensordot
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=2
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
    a=st.integers(1, 50) | st.tuples(st.lists(st.integers()), st.lists(st.integers())),
    b=st.integers(1, 50),
    c=st.integers(1, 50),
    d=st.integers(1, 50),
)
def test_tensordot(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    c,
    d,
):
    if "float16" or "int8" in input_dtype:
        return
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tensordot",
        axes=a,
        x1=np.random.uniform(size=(b, c)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(c, d)).astype(input_dtype[1]),
    )


# trace
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
    c=st.integers(1, 50),
    offset=st.integers(-10, 10),
)
def test_trace(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    c,
    offset,
):
    if "float16" or "int8" in input_dtype:
        return
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="trace",
        offset=offset,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
    )


# vecdot
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=2
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
    a=st.integers(-1, 50),
    b=st.integers(1, 50),
    c=st.integers(1, 50),
)
def test_vecdot(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    c,
):
    if "float16" or "int8" in input_dtype:
        return
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vecdot",
        axes=a,
        x1=np.random.uniform(size=(b, c)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(b, b)).astype(input_dtype[1]),
    )


# vector_norm
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 3],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    # axis=st.integers(-3, 5),
    kd=st.booleans(),
    ord=st.integers() | st.floats(),
)
def test_vector_norm(
    array_shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    kd,
    ord,
):
    if "float16" in input_dtype:
        return
    shape = tuple(array_shape)
    x = np.random.uniform(size=shape).astype(input_dtype)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vector_norm",
        x=x,
        axis=None,
        keepdims=kd,
        ord=ord,
    )


# pinv
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_pinv(
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    if "float16" in input_dtype:
        return
    shape = tuple(array_shape)
    x = np.random.uniform(size=shape).astype(input_dtype)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="pinv",
        test_rtol=1e-04,
        test_atol=1e-04,
        x=x,
    )


# qr
@given(
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="qr"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    mode=st.sampled_from(("reduced", "complete")),
    a=st.integers(2, 5),
    b=st.integers(2, 5),
)
def test_qr(
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    mode,
):
    if "float16" in input_dtype:
        return
    x = np.random.uniform(size=(a, b)).astype(input_dtype)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="qr",
        x=x,
        mode=mode,
    )


# svd
@given(
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    fm=st.booleans(),
)
def test_svd(
    array_shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    fm,
):
    if "float16" in input_dtype:
        return
    shape = tuple(array_shape)
    x = np.random.uniform(size=shape).astype(input_dtype)
    ret, ret_from_np = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="svd",
        x=x,
        full_matrices=fm,
        test_values=False,
    )
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
    array_shape=helpers.lists(
        arg=st.integers(1, 5),
        min_size="num_dims",
        max_size="num_dims",
        size_bounds=[2, 5],
    ),
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    kd=st.booleans(),
    ord=st.integers(1, 2) | st.sampled_from(("fro", "nuc")),
)
def test_matrix_norm(
    array_shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    kd,
    ord,
):
    if "float16" in input_dtype:
        return
    shape = tuple(array_shape)
    x = np.random.uniform(size=shape).astype(input_dtype)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_norm",
        x=x,
        keepdims=kd,
        ord=ord,
    )


# matrix_rank
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_float_dtypes[1:]), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
    c=st.integers(1, 50),
    rtol=st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
)
def test_matrix_rank(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    c,
    rtol,
):

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_rank",
        test_atol=1e-04,
        test_rtol=1e-04,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
        rtol=rtol,
    )


# cholesky
@given(
    input_dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(2, 5),
    upper=st.booleans(),
)
def test_cholesky(
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    upper,
):
    if "float16" in input_dtype:
        return
    x = np.random.uniform(size=(a, a)).astype(input_dtype)
    x = np.matmul(x, x.T + 1e-3)  # make symmetric positive-definite
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cholesky",
        test_rtol=1e-02,
        test_atol=1e-02,
        x=x,
        upper=upper,
    )


# cross
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=2
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
    container=helpers.list_of_length(x=st.booleans(), length=2),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
    axis=st.integers(-1, 50),
)
def test_cross(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    axis,
):
    if "float16" or "int8" in input_dtype:
        return
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cross",
        axis=axis,
        x1=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(a, b)).astype(input_dtype[1]),
    )


# diagonal
@given(
    input_dtype=helpers.list_of_length(
        x=st.sampled_from(ivy_np.valid_numeric_dtypes), length=1
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
    offset=st.integers(-10, 50),
    axes=st.lists(st.integers(-2, 50), min_size=2, max_size=2, unique=True),
)
def test_diagonal(
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    a,
    b,
    offset,
    axes,
):
    if "float16" or "int8" in input_dtype:
        return
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="diagonal",
        offset=offset,
        axis1=axes[0],
        axis2=axes[1],
        x=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
    )
