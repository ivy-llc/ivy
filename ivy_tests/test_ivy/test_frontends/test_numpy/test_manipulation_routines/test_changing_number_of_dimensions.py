# local
import numpy as np

import ivy_tests.test_ivy.helpers as helpers
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers import handle_frontend_test


# squeeze
@st.composite
def _squeeze_helper(draw):
    shape = draw(st.shared(helpers.get_shape(), key="value_shape"))
    valid_axes = []
    for index, axis in enumerate(shape):
        if axis == 1:
            valid_axes.append(index)
    valid_axes.insert(0, None)

    return draw(st.sampled_from(valid_axes))


@handle_frontend_test(
    fn_tree="numpy.squeeze",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=_squeeze_helper(),
)
def test_numpy_squeeze(
    *,
    dtype_and_x,
    axis,
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
        a=x[0],
        axis=axis,
    )


@handle_frontend_test(
    fn_tree="numpy.expand_dims",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        min_size=1,
        max_size=1,
        force_int=True,
    ),
)
def test_numpy_expand_dims(
    *,
    dtype_and_x,
    axis,
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
        a=x[0],
        axis=axis,
    )


# atleast_2d
@handle_frontend_test(
    fn_tree="numpy.atleast_2d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_numpy_atleast_2d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys["arrs{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


# atleast_3d
@handle_frontend_test(
    fn_tree="numpy.atleast_3d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_numpy_atleast_3d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys["arrs{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


# atleast_1d
@handle_frontend_test(
    fn_tree="numpy.atleast_1d",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
    ),
    test_with_out=st.just(False),
)
def test_numpy_atleast_1d(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys["arrs{}".format(i)] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


@handle_frontend_test(
    fn_tree="numpy.broadcast_arrays",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=helpers.ints(min_value=1, max_value=10),
        shared_dtype=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_broadcast_arrays(
    *,
    dtype_value,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    input_dtype, value = dtype_value
    arrys = {}
    for i, v in enumerate(value):
        arrys[f"array{i}"] = v
    test_flags.num_positional_args = len(arrys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arrys,
    )
