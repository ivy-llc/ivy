"""Collection of tests for unified linear algebra functions."""

# global
import pytest
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# vector_to_skew_symmetric_matrix
@pytest.mark.parametrize(
    "x",
    [
        [
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
            [[1.0, 2.0, 3.0]],
        ],
        [[1.0, 2.0, 3.0]],
    ],
)
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize("tensor_fn", [ivy.array, helpers.var_fn])
def test_vector_to_skew_symmetric_matrix(x, dtype, tensor_fn, device, call):
    # smoke test
    x = tensor_fn(x, dtype, device)
    ret = ivy.vector_to_skew_symmetric_matrix(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape + (x.shape[-1],)
    # value test
    assert np.allclose(
        call(ivy.vector_to_skew_symmetric_matrix, x),
        ivy.functional.backends.numpy.vector_to_skew_symmetric_matrix(ivy.to_numpy(x)),
    )


# matmul
@given(
    input_dtype=helpers.list_of_length(st.sampled_from(ivy_np.valid_float_dtypes), 2),
    as_variable=helpers.list_of_length(st.booleans(), 2),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="matmul"),
    native_array=helpers.list_of_length(st.booleans(), 2),
    container=helpers.list_of_length(st.booleans(), 2),
    instance_method=st.booleans(),
    a=st.integers(1, 50),
    b=st.integers(1, 50),
    c=st.integers(1, 50),
    seed=st.integers(0, 2**32 - 1),
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


# Still to Add #
# ---------------#

# cholesky
# cross
# det
# diagonal
# eigh
# eigvalsh
# inv
# matrix_norm
# matrix_power
# matrix_rank
# matrix_transpose
# outer
# pinv
# qr
# slogdet
# solve
# svd
# svdvals
# tensordot
# trace
# vecdot
# vector_norm
