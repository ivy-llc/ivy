# global
import numpy as np
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers import handle_frontend_test

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
    test_with_out=st.just(True),
    test_inplace=st.just(False),
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
    fn_tree="torch.nn.functional.dropout2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=4,
        min_dim_size=1,
        max_dim_size=5,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    training=st.booleans(),
    test_with_out=st.just(False),
    test_inplace=st.just(False),
)
def test_torch_dropout2d(
    *,
    dtype_and_x,
    prob,
    training,
    test_flags,
    backend_fw,
    on_device,
    fn_name,
    ground_truth_backend,
):
    dtype, x = dtype_and_x
    ret, gt_ret = helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        test_flags=test_flags,
        test_values=False,
        on_device=on_device,
        fw=backend_fw,
        fn_name=fn_name,
        input=x[0],
        p=prob,
        training=training,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    gt_ret = helpers.flatten_and_to_np(ret=gt_ret)
    x = np.asarray(x[0], dtype[0]).flatten()
    for u, v, w in zip(ret, gt_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape
