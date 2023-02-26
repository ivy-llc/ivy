# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# copyto
@handle_frontend_test(
    fn_tree="numpy.copyto",
    dtypes_values_casting=np_frontend_helpers.dtypes_values_casting_dtype(
        arr_func=[
            lambda: helpers.dtype_and_values(
                available_dtypes=helpers.get_dtypes("valid"),
                num_arrays=2,
                shared_dtype=True,
                min_num_dims=1,
            )
        ],
    ),
    test_with_out=st.just(False),
    where=np_frontend_helpers.where(),
)
def test_numpy_copyto(
    dtypes_values_casting,
    where,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs, casting, _ = dtypes_values_casting
    src = xs[0]
    dst = xs[1]
    where, input_dtypes, test_flags = np_frontend_helpers.handle_where_and_array_bools(
        where=where, input_dtype=input_dtypes, test_flags=test_flags
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        src=src,
        dst=dst,
        where=where,
        casting=casting,
        test_values=False,
    )
    w = helpers.flatten_and_to_np(ret=where)
    dst_f = helpers.flatten_and_to_np(ret=dst)
    src_f = helpers.flatten_and_to_np(ret=src)
    for i, j, c in zip(src_f, dst_f, w):
        if c:
            assert i == j


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
