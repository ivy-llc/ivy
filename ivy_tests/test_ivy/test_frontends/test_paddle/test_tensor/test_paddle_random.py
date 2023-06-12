# global

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# randint
@handle_frontend_test(
    fn_tree="paddle.tensor.randint",
    low=helpers.ints(min_value=0, max_value=10),
    high=helpers.ints(min_value=11, max_value=20),
    dtype=helpers.get_dtypes("integer"),
    shape=helpers.get_shape(),
)
def test_paddle_randint(
    low,
    high,
    dtype,
    frontend,
    test_flags,
    shape,
    fn_tree,
):
    def call():
        helpers.test_frontend_function(
            input_dtypes=dtype,
            frontend=frontend,
            test_values=False,
            fn_tree=fn_tree,
            test_flags=test_flags,
            low=low,
            high=high,
            shape=shape,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape
