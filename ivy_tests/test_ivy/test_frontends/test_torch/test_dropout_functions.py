# global
import numpy as np
from hypothesis import given, strategies as st
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# local
import ivy_tests.test_ivy.helpers as helpers


@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.dropout"
    ),
    with_inplace=st.booleans(),
)
def test_torch_dropout(
    dtype_and_x,
    prob,
    training,
    with_inplace,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x

    ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        with_inplace=with_inplace,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.dropout",
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
