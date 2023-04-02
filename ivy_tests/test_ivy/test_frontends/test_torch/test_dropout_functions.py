# global
import numpy as np
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers import handle_frontend_test
import torch
import unittest
# local
import ivy_tests.test_ivy.helpers as helpers


@handle_frontend_test(
    fn_tree="torch.nn.functional.dropout",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    training=st.booleans(),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_dropout(
    *,
    dtype_and_x,
    prob,
    training,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        p=prob,
        training=training,
        test_values=False,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    x = np.asarray(x[0], input_dtype[0])
    for u in ret:
        # cardinality test
        assert u.shape == x.shape

@handle_frontend_test(
    fn_tree="torch.nn.functional.dropout3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    training=st.booleans(),
    test_with_out=st.just(False),
    test_inplace=st.booleans(),
)
def test_torch_dropout3d(*, dtype_and_x, prob, training, on_device, fn_tree, frontend, test_flags):
    input_dtype, x = dtype_and_x
    dropout = torch.nn.Dropout3d(p=prob)
    x = torch.tensor(x[0], dtype=input_dtype[0])

    if training:
        dropout.train()
    else:
        dropout.eval()

    y = dropout(x)
    y = helpers.flatten_and_to_np(ret=y)

    assert y.shape == x.shape
    num_nonzero = np.sum(y != 0.0)
    expected_num_nonzero = np.prod(x.shape) * (1 - prob)
    assert np.abs(num_nonzero - expected_num_nonzero) <= 0.1 * expected_num_nonzero
