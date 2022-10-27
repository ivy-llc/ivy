# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import ivy
import numpy as np


# Helpers #
# ------- #


# dirichlet
@handle_cmd_line_args
@given(
    dtype_and_alpha=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=2),
        shape=st.tuples(
            st.integers(min_value=2, max_value=5),
        ),
        min_value=0,
        max_value=100,
        exclude_min=True,
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    seed=helpers.ints(min_value=0, max_value=100),
    num_positional_args=helpers.num_positional_args(fn_name="dirichlet"),
)
def test_dirichlet(
    dtype_and_alpha,
    size,
    seed,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, alpha = dtype_and_alpha

    def call():
        return helpers.test_function(
            input_dtypes=dtype,
            as_variable_flags=as_variable,
            with_out=with_out,
            num_positional_args=num_positional_args,
            native_array_flags=native_array,
            container_flags=container,
            instance_method=instance_method,
            test_values=False,
            fw=fw,
            fn_name="dirichlet",
            alpha=np.asarray(alpha[0], dtype=dtype[0]),
            size=size,
            seed=seed,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt1 = call()
        assert ivy.any(ret == ret1)
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        u, v = ivy.array(u), ivy.array(v)
        assert ivy.all(ivy.sum(u, axis=-1) == ivy.sum(v, axis=-1))
        assert ivy.all(u >= 0) and ivy.all(u <= 1)
        assert ivy.all(v >= 0) and ivy.all(v <= 1)
