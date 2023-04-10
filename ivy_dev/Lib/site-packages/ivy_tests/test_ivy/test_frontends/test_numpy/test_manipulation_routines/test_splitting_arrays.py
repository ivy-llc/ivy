# global
from hypothesis import strategies as st, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa
    _get_split_locations,
)


# split
@handle_frontend_test(
    fn_tree="numpy.split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_split_locations(min_num_dims=1),
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
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        axis=axis,
    )


# array_split
@handle_frontend_test(
    fn_tree="numpy.array_split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_split_locations(min_num_dims=1),
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
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
        axis=axis,
    )


# dsplit
@handle_frontend_test(
    fn_tree="numpy.dsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=3), key="value_shape"),
    ),
    indices_or_sections=_get_split_locations(min_num_dims=3, axis=2),
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
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
    )


# vsplit
@handle_frontend_test(
    fn_tree="numpy.vsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_split_locations(min_num_dims=2, axis=0),
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
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
    )


# hsplit
@handle_frontend_test(
    fn_tree="numpy.hsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_split_locations(min_num_dims=1, axis=1),
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
    # TODO: remove the assumption when these bugfixes are merged and version-pinned
    # https://github.com/tensorflow/tensorflow/pull/59523
    # https://github.com/google/jax/pull/14275
    assume(
        not (
            len(value[0].shape) == 1
            and ivy.current_backend_str() in ("tensorflow", "jax")
        )
    )
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        ary=value[0],
        indices_or_sections=indices_or_sections,
    )
