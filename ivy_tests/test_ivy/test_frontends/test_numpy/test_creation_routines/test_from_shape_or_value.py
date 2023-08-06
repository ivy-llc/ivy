# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.globals as test_globals
from ivy_tests.test_ivy.helpers import handle_frontend_test, update_backend


# empty
@handle_frontend_test(
    fn_tree="numpy.empty",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_empty(
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        dtype=dtype[0],
    )


# empty_like
@handle_frontend_test(
    fn_tree="numpy.empty_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_empty_like(
    dtype_and_x,
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        prototype=x[0],
        dtype=dtype[0],
        order="K",
        subok=True,
        shape=shape,
    )


# eye
@handle_frontend_test(
    fn_tree="numpy.eye",
    rows=helpers.ints(min_value=3, max_value=10),
    cols=helpers.ints(min_value=3, max_value=10),
    k=helpers.ints(min_value=0, max_value=2),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_eye(
    rows,
    cols,
    k,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        N=rows,
        M=cols,
        k=k,
        dtype=dtype[0],
    )


# identity
@handle_frontend_test(
    fn_tree="numpy.identity",
    n=helpers.ints(min_value=1, max_value=10),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_identity(
    n,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        dtype=dtype[0],
    )


# ones
@handle_frontend_test(
    fn_tree="numpy.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_ones(
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )


# ones_like
@handle_frontend_test(
    fn_tree="numpy.ones_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_ones_like(
    dtype_and_x,
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
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
        dtype=dtype[0],
        order="K",
        subok=True,
        shape=shape,
    )


# zeros
@handle_frontend_test(
    fn_tree="numpy.zeros",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_zeros(
    shape,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype[0],
    )


# zeros_like
@handle_frontend_test(
    fn_tree="numpy.zeros_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_zeros_like(
    dtype_and_x,
    dtype,
    shape,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
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
        dtype=dtype[0],
        order="K",
        subok=True,
        shape=shape,
    )


# full and full_like helper
@st.composite
def _input_fill_and_dtype(draw):
    dtype = draw(helpers.get_dtypes("float", full=False))
    dtype_and_input = draw(helpers.dtype_and_values(dtype=dtype))
    with update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        if ivy_backend.is_uint_dtype(dtype[0]):
            fill_values = draw(st.integers(min_value=0, max_value=5))
        elif ivy_backend.is_int_dtype(dtype[0]):
            fill_values = draw(st.integers(min_value=-5, max_value=5))
        else:
            fill_values = draw(
                helpers.floats(
                    min_value=-5,
                    max_value=5,
                    large_abs_safety_factor=10,
                    small_abs_safety_factor=10,
                    safety_factor_scale="log",
                )
            )
        dtype_to_cast = draw(helpers.get_dtypes("float", full=False))
    return dtype, dtype_and_input[1], fill_values, dtype_to_cast[0]


# full
@handle_frontend_test(
    fn_tree="numpy.full",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    input_fill_dtype=_input_fill_and_dtype(),
    test_with_out=st.just(False),
)
def test_numpy_full(
    shape,
    input_fill_dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        fill_value=fill,
        dtype=dtype_to_cast,
    )


# full_like
@handle_frontend_test(
    fn_tree="numpy.full_like",
    input_fill_dtype=_input_fill_and_dtype(),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    test_with_out=st.just(False),
)
def test_numpy_full_like(
    input_fill_dtype,
    shape,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        fill_value=fill,
        dtype=dtype_to_cast,
        order="K",
        subok=True,
        shape=shape,
    )


@handle_frontend_test(
    fn_tree="numpy.fromfunction",
    shape_and_function=helpers.shape_and_function(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
    test_with_out=st.just(False),
)
def test_numpy_fromfunction(
    shape_and_function,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    shape, function = shape_and_function
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        function=function,
        shape=shape,
        dtype=dtype[0],
    )
