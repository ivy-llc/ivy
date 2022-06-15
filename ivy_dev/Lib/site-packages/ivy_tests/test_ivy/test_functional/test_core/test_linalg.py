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
    if fw == "torch" and input_dtype == "float16":
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
        n=n,
        x=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
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
        x=np.random.uniform(size=(b, a, a)).astype(input_dtype[0]),
    )


# eigh
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
def test_eigh(
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
        "eigh",
        x=np.random.uniform(size=(b, a, a)).astype(input_dtype[0]),
    )


# eigvalsh
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
        x=np.random.uniform(size=(b, a, a)).astype(input_dtype[0]),
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
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
)
def test_slogdet(
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
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 1),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
    axis=st.integers(-10, 10) | st.tuples(st.lists(st.integers())),
    kd=st.booleans(),
    ord=st.integers() | st.floats(),
)
def test_vector_norm(
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
    kd,
    ord,
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
        "vector_norm",
        axis=axis,
        keepdims=kd,
        ord=ord,
        x=np.random.uniform(size=(a, b)).astype(input_dtype[0]),
    )


# pinv
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
    c=st.integers(1, 50),
    seed=st.integers(0, 2**4 - 1),
)
def test_pinv(
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
    if "float16" in input_dtype:
        return
    np.random.seed(seed)
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
        rtol=5e-02,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
    )


# qr
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
    c=st.integers(1, 50),
    mode=st.sampled_from(("reduced", "complete")),
)
def test_qr(
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
    mode,
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
        "qr",
        mode=mode,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
    )


# svd
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
    c=st.integers(1, 50),
    fm=st.booleans(),
)
def test_svd(
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
    fm,
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
        "svd",
        full_matrices=fm,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
    )


# matrix_norm
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
    c=st.integers(1, 50),
    kd=st.booleans(),
    ord=st.integers(1, 10)
    | st.floats(1, 10)
    | st.sampled_from(("fro", "nuc", "float('inf')", "-float('inf')")),
)
def test_matrix_norm(
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
    kd,
    ord,
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
        "matrix_norm",
        keepdims=kd,
        ord=ord,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
    )


# matrix_rank
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
        rtol=5e-02,
        x=np.random.uniform(size=(a, b, c)).astype(input_dtype[0]),
    )


# cholesky
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 1),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 1),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    upper=st.booleans(),
)
def test_cholesky(
    input_dtype,
    as_variable,
    with_out,
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
    helpers.test_array_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "cholesky",
        upper=upper,
        x=np.random.uniform(size=(a, a)).astype(input_dtype[0]),
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
