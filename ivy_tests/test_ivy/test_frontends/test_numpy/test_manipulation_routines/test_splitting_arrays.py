# global
from hypothesis import strategies as st, assume
import numpy as np

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import assert_all_close, handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import (  # noqa
    _get_splits,
)


# split
@handle_frontend_test(
    fn_tree="numpy.split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_splits(min_num_dims=1, allow_none=False),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    test_with_out=st.just(False),
)
def test_numpy_split(
    *,
    dtype_value,
    indices_or_sections,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        axis=axis,
        test_values=False,
    )
    ret_flattened = [ivy.to_numpy(r) for r in ret]
    for ret_np, ret_gt in zip(ret_flattened, frontend_ret[0]):
        assert_all_close(ret_np, ret_gt, ground_truth_backend=frontend)


# array_split
@handle_frontend_test(
    fn_tree="numpy.array_split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_splits(min_num_dims=1, allow_none=False),
    axis=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    test_with_out=st.just(False),
)
def test_numpy_array_split(
    *,
    dtype_value,
    indices_or_sections,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    assume(isinstance(indices_or_sections, int))
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        axis=axis,
        test_values=False,
    )
    ret_flattened = [ivy.to_numpy(r) for r in ret]
    for ret_np, ret_gt in zip(ret_flattened, frontend_ret[0]):
        assert_all_close(ret_np, ret_gt, ground_truth_backend=frontend)


# dsplit
@handle_frontend_test(
    fn_tree="numpy.dsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=3), key="value_shape"),
    ),
    indices_or_sections=_get_splits(min_num_dims=3, axis=2, allow_none=False),
    test_with_out=st.just(False),
)
def test_numpy_dsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    if isinstance(indices_or_sections, np.ndarray):
        assume(indices_or_sections.ndim == 0)
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        test_values=False,
    )
    ret_flattened = [ivy.to_numpy(r) for r in ret]
    for ret_np, ret_gt in zip(ret_flattened, frontend_ret[0]):
        assert_all_close(ret_np, ret_gt, ground_truth_backend=frontend)


# vsplit
@handle_frontend_test(
    fn_tree="numpy.vsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(min_num_dims=2, axis=0, allow_none=False),
    test_with_out=st.just(False),
)
def test_numpy_vsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    assume(isinstance(indices_or_sections, int))
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        test_values=False,
    )
    ret_flattened = [ivy.to_numpy(r) for r in ret]
    for ret_np, ret_gt in zip(ret_flattened, frontend_ret[0]):
        assert_all_close(ret_np, ret_gt, ground_truth_backend=frontend)


# hsplit
@handle_frontend_test(
    fn_tree="numpy.hsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(min_num_dims=1, axis=1, allow_none=False),
    test_with_out=st.just(False),
)
def test_numpy_hsplit(
    *,
    dtype_value,
    indices_or_sections,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    if isinstance(indices_or_sections, np.ndarray):
        assume(indices_or_sections.ndim == 0)
    ret, frontend_ret = helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        test_values=False,
    )
    ret_flattened = [ivy.to_numpy(r) for r in ret]
    for ret_np, ret_gt in zip(ret_flattened, frontend_ret[0]):
        assert_all_close(ret_np, ret_gt, ground_truth_backend=frontend)
