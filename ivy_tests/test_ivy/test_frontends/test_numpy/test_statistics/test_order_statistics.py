from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="numpy.quantile",
    dtype_a_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2,
        small_abs_safety_factor=2,
        safety_factor_scale="log",
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
        valid_axis=True,
        allow_neg_axes=False,
        min_axes_size=1,
    ),
    q=helpers.array_values(
        dtype=helpers.get_dtypes("float"),
        shape=helpers.get_shape(min_dim_size=1, max_num_dims=1, min_num_dims=1),
        min_value=0.0,
        max_value=1.0,
        exclude_max=False,
        exclude_min=False,
    ),
    interpolation=helpers.lists(
        arg=st.sampled_from(["linear", "lower", "higher"]),
        min_size=1,
        max_size=1
    ),
    num_positional_args=helpers.num_positional_args(fn_name="quantile"),
    keep_dims=st.booleans(),
)
def test_numpy_quantile(
        *,
        dtype_a_and_axis,
        interpolation,
        q,
        keep_dims,
        as_variable,
        num_positional_args,
        native_array,
        with_out,
        frontend,
        fn_tree,
        on_device,
):
    input_dtype, x, axis = dtype_a_and_axis
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        q=q,
        axis=axis,
        interpolation=interpolation[0],
        keepdims=keep_dims,
    )
