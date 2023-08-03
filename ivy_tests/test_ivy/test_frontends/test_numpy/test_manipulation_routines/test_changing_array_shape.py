# global
import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            ),
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


# reshape
@handle_frontend_test(
    fn_tree="numpy.reshape",
    dtypes_x_shape=dtypes_x_reshape(),
    order=st.sampled_from(["C", "F", "A"]),
)
def test_numpy_reshape(
    *,
    dtypes_x_shape,
    order,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtypes, x, shape = dtypes_x_shape
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        newshape=shape,
        order=order,
    )


@handle_frontend_test(
    fn_tree="numpy.broadcast_to",
    dtype_x_shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), ret_shape=True
    ),
    factor=helpers.ints(min_value=1, max_value=5),
    test_with_out=st.just(False),
)
def test_numpy_broadcast_to(
    *,
    dtype_x_shape,
    factor,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, shape = dtype_x_shape
    broadcast_shape = (factor,) + shape
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=x[0],
        shape=broadcast_shape,
    )


@handle_frontend_test(
    fn_tree="numpy.ravel",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    order=st.sampled_from(["C", "F", "A", "K"]),
    test_with_out=st.just(False),
)
def test_numpy_ravel(
    *,
    dtype_and_x,
    order,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        order=order,
    )


# moveaxis
@handle_frontend_test(
    fn_tree="numpy.moveaxis",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
    ),
    source=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
    destination=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
    test_with_out=st.just(False),
)
def test_numpy_moveaxis(
    *,
    dtype_and_a,
    source,
    destination,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
        source=source,
        destination=destination,
    )


# resize
@st.composite
def dtype_and_resize(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=2,
                max_dim_size=10,
            ),
        )
    )
    new_shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=2,
            max_num_dims=5,
            min_dim_size=2,
            max_dim_size=10,
        ),
    )
    return dtype, x, new_shape


@handle_frontend_test(
    fn_tree="numpy.resize",
    dtypes_x_shape=dtype_and_resize(),
)
def test_numpy_resize(
    *,
    dtypes_x_shape,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, x, new_shape = dtypes_x_shape
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        newshape=new_shape,
    )


# asfarray
@handle_frontend_test(
    fn_tree="numpy.asfarray",
    dtype_and_a=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_numpy_asfarray(
    *,
    dtype_and_a,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
    )


# asarray_chkfinite
@handle_frontend_test(
    fn_tree="numpy.asarray_chkfinite",
    dtype_and_a=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    test_with_out=st.just(False),
)
def test_numpy_asarray_chkfinite(
    *,
    dtype_and_a,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
    )


# require
@handle_frontend_test(
    fn_tree="numpy.require",
    dtype_and_a=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    requirements=st.sampled_from(["C", "F", "A", "O", "W", "E"]),
    like=st.just(None),
    test_with_out=st.just(False),
)
def test_numpy_require(
    *,
    dtype_and_a,
    requirements,
    like,
    on_device,
    fn_tree,
    frontend,
    backend_fw,
    test_flags,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
        dtype=np.dtype(dtype[0]),
        requirements=requirements,
        like=like,
    )


#flatten
@handle_frontend_test(
    fn_tree="numpy.ndarray.flatten",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    order=st.sampled_from(["C", "F", "A", "K"]),
    test_with_out=st.just(False),
)
