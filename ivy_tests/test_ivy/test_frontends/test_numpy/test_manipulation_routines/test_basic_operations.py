# global
from hypothesis import strategies as st
import ivy
import ivy.functional.frontends.numpy as ivy_np
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def generate_copyto_args(draw):
    input_dtypes, xs, casting, _ = draw(
        np_frontend_helpers.dtypes_values_casting_dtype(
            arr_func=[
                lambda: helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("valid"),
                    num_arrays=2,
                    shared_dtype=True,
                    min_num_dims=1,
                )
            ],
        )
    )
    where = draw(np_frontend_helpers.where(shape=xs[0].shape))
    return input_dtypes, xs, casting, where


# copyto
@handle_frontend_test(
    fn_tree="numpy.copyto",
    test_with_out=st.just(False),
    copyto_args=generate_copyto_args(),
)
def test_numpy_copyto(
    copyto_args,
):
    _, xs, casting, where = copyto_args
    if isinstance(where, list) or isinstance(where, tuple):
        where = where[0]

    src_ivy = ivy_np.array(xs[0])
    dst_ivy = ivy_np.array(xs[1])
    ivy_np.copyto(dst_ivy, src_ivy, where=where, casting=casting)

    src_np = np.array(xs[0])
    dst_np = np.array(xs[1])
    np.copyto(dst_np, src_np, where=where, casting=casting)

    assert dst_np.shape == dst_ivy.shape
    # value test
    dst_ = ivy.to_numpy(dst_ivy.ivy_array)
    helpers.assert_all_close(dst_, dst_np)
    assert id(src_ivy) != id(dst_ivy)


# shape
@handle_frontend_test(
    fn_tree="numpy.shape",
    xs_n_input_dtypes_n_unique_idx=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid")
    ),
    test_with_out=st.just(False),
)
def test_numpy_shape(
    *,
    xs_n_input_dtypes_n_unique_idx,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtypes, xs = xs_n_input_dtypes_n_unique_idx
    ret, ret_gt = helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        array=xs[0],
    )
    # Manually compare the shape here because ivy.shape doesn't return an array, so
    # ivy.to_numpy will narrow the bit-width, resulting in different dtypes. This is
    # not an issue with the front-end function, but how the testing framework converts
    # non-array function outputs to arrays.
    assert len(ret) == len(ret_gt)
    for i, j in zip(ret, ret_gt):
        assert i == j
