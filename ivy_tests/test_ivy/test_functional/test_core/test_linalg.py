"""Collection of tests for unified linear algebra functions."""

# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# vector_to_skew_symmetric_matrix
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_numeric_dtypes), 1),
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
    if "float16" in input_dtype or "int8" in input_dtype:
        return
    helpers.test_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "vector_to_skew_symmetric_matrix",
        vector=np.random.uniform(size=(a, 3)).astype(input_dtype[0]),
    )


# matrix_power
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 1),
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matrix_power",
        test_rtol=1e-02,
        test_atol=1e-02,
        x=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
        n=n,
    )


# matmul
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_numeric_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matmul",
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "det",
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
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "eigh",
        test_rtol=1e-02,
        test_atol=1e-02,
        x=x,
        test_values=False,
    )

    # flattened array returns
    ret_np_flat, ret_from_np_flat = helpers.get_flattened_array_returns(
        ret, ret_from_np
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "eigvalsh",
        test_rtol=0.01,
        test_atol=0.01,
        x=np.random.uniform(size=(b, a, a)).astype(input_dtype),
    )


# inv
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 1),
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "inv",
        test_rtol=1e-02,
        test_atol=1e-02,
        x=np.random.uniform(size=(b, a, a)).astype(input_dtype[0]),
    )


# matrix_transpose
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_numeric_dtypes), 1),
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matrix_transpose",
        x=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
    )


# outer
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_numeric_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "outer",
        x1=np.random.uniform(size=a).astype(input_dtype[0]),
        x2=np.random.uniform(size=b).astype(input_dtype[1]),
    )


# slogdet
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 1),
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
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "slogdet",
        x=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
        test_rtol=1e-3,
        test_atol=1e-3,
    )


# solve
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "solve",
        test_rtol=1e-03,
        test_atol=1e-03,
        x1=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(a, 1)).astype(input_dtype[1]),
    )


# svdvals
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_numeric_dtypes), 1),
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
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "svdvals",
        x=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
    )


# tensordot
@given(
    dtype_x1_x2_axis=helpers.dtype_value1_value2_axis(
        ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=50,
    ),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
    dtype, x1, x2, axis, = dtype_x1_x2_axis
    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "tensordot",
        x1=np.asarray(x1, dtype=dtype),
        x2=np.asarray(x2, dtype=dtype).swapaxes(-1,-2),
        axes=axis,
    )


# trace
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=50,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
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
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "trace",
        x=np.asarray(x, dtype=dtype),
        offset=offset,
    )


# vecdot
@given(
    dtype_x1_x2_axis=helpers.dtype_value1_value2_axis(
        ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=50,
    ),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "vecdot",
        x1=np.asarray(x1, dtype=dtype),
        x2=np.asarray(x2, dtype=dtype),
        axis=axis
    )


# vector_norm
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    # axis=st.integers(-3, 5),
    kd=st.booleans(),
    ord=st.integers(1, 2),
)
def test_vector_norm(
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
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "vector_norm",
        x=np.asarray(x, dtype=dtype),
        axis=None,
        keepdims=kd,
        ord=ord,
    )


# pinv
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
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
):
    dtype, x = dtype_x
    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "pinv",
        test_rtol=1e-04,
        test_atol=1e-04,
        x=np.asarray(x, dtype=dtype),
    )


# qr
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_float_dtypes,
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
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "qr",
        x=np.asarray(x, dtype=dtype),
        mode=mode,
    )


# svd
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_float_dtypes,
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
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
    try:
        ret, ret_from_np = helpers.test_function(
            dtype,
            as_variable,
            False,
            num_positional_args,
            native_array,
            container,
            instance_method,
            fw,
            "svd",
            test_values=False,
            x=np.asarray(x, dtype=dtype),
            full_matrices=fm,
        )
    except TypeError:
        return

    # flattened array returns
    ret_np_flat, ret_from_np_flat = helpers.get_flattened_array_returns(
        ret, ret_from_np
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
        ivy_np.valid_float_dtypes,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
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
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matrix_norm",
        x=np.asarray(x, dtype=dtype),
        keepdims=kd,
        ord=ord,
    )


# matrix_rank
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_float_dtypes[1:],
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=3,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
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
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matrix_rank",
        test_atol=1e-04,
        test_rtol=1e-04,
        x=np.asarray(x, dtype=dtype),
        rtol=rtol,
    )


# cholesky
# Todo: this test is not passed
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_float_dtypes,
        min_value=0,
        max_value=1000,
        shape=st.integers(2, 5).map(lambda x: tuple([x, x]))
    ).filter(lambda dtype_x: np.linalg.det(np.asarray(dtype_x[1])) != 0),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 1),
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
    x = np.matmul(x, x.T) + np.identity(x.shape[0]) * 1e-3  # make symmetric positive-definite

    helpers.test_function(
        dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "cholesky",
        test_rtol=1e-02,
        test_atol=1e-02,
        x=x,
        upper=upper,
    )


# cross
@given(
    dtype_x1_x2_axis=helpers.dtype_value1_value2_axis(
        ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=50,
    ),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "cross",
        x1=np.asarray(x1, dtype=dtype),
        x2=np.asarray(x2, dtype=dtype),
        axis=axis,
    )


# diagonal
@given(
    dtype_x=helpers.dtype_and_values(
        ivy_np.valid_numeric_dtypes,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=50,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    offset=st.integers(-10, 50),
    axes=st.lists(st.integers(-2, 1), min_size=2, max_size=2, unique=True)
           .filter(lambda axes: axes[0] % 2 != axes[1] % 2),
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
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "diagonal",
        x=np.asarray(x, dtype=dtype),
        offset=offset,
        axis1=axes[0],
        axis2=axes[1],

    )
