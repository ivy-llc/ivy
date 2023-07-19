# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_statistical as test_statistical
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _a_bins_weights_range(draw):
    a, bin_size, *_, range, weights, _, dtype_input = draw(
        test_statistical._histogram_helper()
    )
    bin_selectors = [
        "stone",
        "auto",
        "doane",
        "fd",
        "rice",
        "scott",
        "sqrt",
        "sturges",
        bin_size,
    ]
    bins = draw(st.sampled_from(bin_selectors))
    if isinstance(bins, str):
        weights = None
    return a, bins, weights, range


# bincount
@handle_frontend_test(
    fn_tree="numpy.bincount",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=2,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
            ),
            key="a_s_d",
        ),
    ),
    test_with_out=st.just(False),
)
def test_numpy_bincount(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        weights=None,
        minlength=0,
    )


@handle_frontend_test(fn_tree="numpy.histogram_bin_edges", data=_a_bins_weights_range())
def test_numpy_histogram_bin_edges(
    *,
    data,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    a, bins, weights, range = data
    helpers.test_frontend_function(
        frontend=frontend,
        input_dtypes=[a.dtype],
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        bins=bins,
        range=range,
        weights=weights,
    )
