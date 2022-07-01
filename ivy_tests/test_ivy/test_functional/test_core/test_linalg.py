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
    if "float16" or "int8" in input_dtype:
        return
    helpers.test_array_function(
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matrix_power",
        rtol=1e-03,
        atol=1e-03,
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matmul",
        rtol=5e-02,
        atol=5e-02,
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "det",
        rtol=1e-04,
        atol=1e-04,
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "eigh",
        rtol=1e-02,
        atol=1e-02,
        x=x,
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "eigvalsh",
        rtol=0.01,
        atol=0.01,
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "inv",
        rtol=1e-02,
        atol=1e-02,
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
    helpers.test_array_function(
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
    helpers.test_array_function(
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
    helpers.test_array_function(
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "solve",
        rtol=1e-04,
        atol=1e-04,
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
    helpers.test_array_function(
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
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_numeric_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "tensordot",
        axes=a,
        x1=np.random.uniform(size=(b, c)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(c, d)).astype(input_dtype[1]),
    )


# trace
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "trace",
        offset=offset,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
    )


# vecdot
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_numeric_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "vecdot",
        axes=a,
        x1=np.random.uniform(size=(b, c)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(b, b)).astype(input_dtype[1]),
    )


# vector_norm
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[2, 3]
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "vector_norm",
        x=x,
        axis=None,
        keepdims=kd,
        ord=ord,
    )


# pinv
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[2, 5]
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "pinv",
        rtol=1e-04,
        atol=1e-04,
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "qr",
        x=x,
        mode=mode,
    )


# svd
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[2, 5]
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "svd",
        rtol=1e-01,
        atol=1e-01,
        x=x,
        full_matrices=fm,
    )


# matrix_norm
@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[2, 5]
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matrix_norm",
        x=x,
        keepdims=kd,
        ord=ord,
    )


# matrix_rank
@given(
    input_dtype=helpers.list_of_length(
        st.sampled_from(ivy_np.valid_float_dtypes[1:]), 1
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
):
    if "float16" in input_dtype:
        return
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "matrix_rank",
        atol=1e-04,
        rtol=1e-04,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        False,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "cholesky",
        rtol=1e-02,
        atol=1e-02,
        x=x,
        upper=upper,
    )


# cross
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "cross",
        axis=axis,
        x1=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
        x2=np.random.uniform(size=(a, b)).astype(input_dtype[1]),
    )


# diagonal
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "diagonal",
        offset=offset,
        axis1=axes[0],
        axis2=axes[1],
        x=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
    )
