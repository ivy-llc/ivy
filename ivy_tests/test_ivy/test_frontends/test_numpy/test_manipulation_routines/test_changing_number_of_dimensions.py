# local
import numpy as np

import ivy_tests.test_ivy.helpers as helpers
from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_dtype import dtypes_shared


# --- Helpers --- #
# --------------- #


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


# broadcast_arrays
@st.composite
def broadcastable_arrays(draw, dtypes):
    num_arrays = st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays")
    shapes = draw(num_arrays.flatmap(helpers.mutually_broadcastable_shapes))
    dtypes = draw(dtypes)
    arrays = []
    for c, (shape, dtype) in enumerate(zip(shapes, dtypes), 1):
        x = draw(helpers.array_values(dtype=dtype, shape=shape), label=f"x{c}").tolist()
        arrays.append(x)
    return arrays


# --- Main --- #
# ------------ #


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
    backend_fw,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys[f"arrs{i}"] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
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
    backend_fw,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys[f"arrs{i}"] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
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
    backend_fw,
):
    input_dtype, arrays = dtype_and_x
    arys = {}
    for i, (array, idtype) in enumerate(zip(arrays, input_dtype)):
        arys[f"arrs{i}"] = np.asarray(array, dtype=idtype)
    test_flags.num_positional_args = len(arys)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **arys,
    )


@handle_frontend_test(
    fn_tree="numpy.broadcast_arrays",
    arrays=broadcastable_arrays(dtypes_shared("num_arrays")),
    input_dtypes=dtypes_shared("num_arrays"),
    test_with_out=st.just(False),
)
def test_numpy_broadcast_arrays(
    *,
    arrays,
    input_dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    args = {}
    for i, (array, dtype) in enumerate(zip(arrays, input_dtypes)):
        args[f"x{i}"] = np.asarray(array, dtype=dtype)
    test_flags.num_positional_args = len(args)
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        **args,
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
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
    )


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
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
    )
