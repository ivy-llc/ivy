# global
from hypothesis import strategies as st
import numpy as np

import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy.functional.frontends.numpy as np_frontend


@st.composite
def _histogram_helper(draw):
    dtype_input = draw(st.sampled_from(draw(helpers.get_dtypes("float"))))
    bins = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=(draw(helpers.ints(min_value=1, max_value=10)),),
            abs_smallest_val=-10,
            min_value=-10,
            max_value=10,
        )
    )
    bins = np.asarray(sorted(set(bins)), dtype=dtype_input)
    if len(bins) == 1:
        bins = int(abs(bins[0]))
        if bins == 0:
            bins = 1
        if dtype_input in draw(helpers.get_dtypes("unsigned")):
            range = (
                draw(
                    helpers.floats(
                        min_value=0, max_value=10, exclude_min=False, exclude_max=False
                    )
                ),
                draw(
                    helpers.floats(
                        min_value=11, max_value=20, exclude_min=False, exclude_max=False
                    )
                ),
            )
        else:
            range = (
                draw(helpers.floats(min_value=-10, max_value=0)),
                draw(helpers.floats(min_value=1, max_value=10)),
            )
        range = draw(st.sampled_from([range, None]))
    else:
        range = None
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=1, min_dim_size=5, max_dim_size=20
        )
    )
    a = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=shape,
            min_value=-20,
            max_value=20,
        )
    )
    weights = draw(
        helpers.array_values(
            dtype=dtype_input,
            shape=shape,
            min_value=-20,
            max_value=20,
        )
    )
    bin_selectors = [
        "stone",
        "fd",
        "auto",
        "doane",
        "rice",
        "scott",
        "sqrt",
        "sturges",
        bins,
    ]
    bins = draw(st.sampled_from(bin_selectors))
    if isinstance(bins, str):
        weights = None
    weights = draw(st.sampled_from([weights, None]))
    density = draw(st.booleans())
    return (a, bins, range, weights, density)


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


@handle_frontend_test(fn_tree="numpy.histogram_bin_edges", data=_histogram_helper())
def test_numpy_histogram_bin_edges(data):
    a, bins, range, weights, density = data
    ret = np_frontend.histogram_bin_edges(
        a=a,
        bins=bins,
        range=range,
        weights=weights,
    )
    ret_gt = np.histogram_bin_edges(
        a=a,
        bins=bins,
        range=range,
        weights=weights,
    )
    assert ivy.allclose(ret.ivy_array, ret_gt)


@handle_frontend_test(fn_tree="numpy.histogram", data=_histogram_helper())
def test_numpy_histogram(data):
    a, bins, range, weights, density = data
    ret = np_frontend.histogram(
        a=a, bins=bins, range=range, weights=weights, density=density
    )
    ret_gt = np.histogram(a=a, bins=bins, range=range, weights=weights, density=density)
    assert ivy.allclose(ret[0].ivy_array, ret_gt[0])
    assert ivy.allclose(ret[1].ivy_array, ret_gt[1])
