# global
import ivy
from hypothesis import assume, strategies as st
# local
import ivy.functional.frontends.paddle as paddle_frontend
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import math


@handle_frontend_test(
    fn_tree="paddle.nn.functional.channel_shuffle",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=["float32","float64"],
    min_num_dims=4,
    max_num_dims=4,
    ret_shape=True
    ),
    groups=helpers.number_helpers.ints(min_value=1),
    test_with_out=st.just(False)
)
def test_paddle_channel_shuffle(
    *,
    dtype_and_x,
    groups,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x,shape = dtype_and_x
    groups=math.gcd(groups,shape[1])
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        groups=groups
    )
