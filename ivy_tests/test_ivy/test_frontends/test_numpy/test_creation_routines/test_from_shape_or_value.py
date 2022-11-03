# global
import ivy
import random
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


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
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_empty(
    shape,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        dtype=dtype,
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
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_empty_like(
    dtype_and_x,
    shape,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        prototype=x[0],
        dtype=dtype,
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
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_eye(
    rows,
    cols,
    k,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        N=rows,
        M=cols,
        k=k,
        dtype=dtype,
    )


# identity
@handle_frontend_test(
    fn_tree="numpy.identity",
    n=helpers.ints(min_value=1, max_value=10),
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_identity(
    n,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        n=n,
        dtype=dtype,
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
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_ones(
    shape,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype,
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
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_ones_like(
    dtype_and_x,
    shape,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        dtype=dtype,
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
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_zeros(
    shape,
    dtypes,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    helpers.test_frontend_function(
        input_dtypes=[dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        dtype=dtype,
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
    dtypes=helpers.get_dtypes("valid"),
)
def test_numpy_zeros_like(
    dtype_and_x,
    dtypes,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    dtype = random.choice(dtypes)
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        dtype=dtype,
        order="K",
        subok=True,
        shape=shape,
    )


# full and full_like helper
@st.composite
def _input_fill_and_dtype(draw):
    dtype = random.choice(draw(helpers.get_dtypes("float")))
    dtype_and_input = draw(helpers.dtype_and_values(dtype=[dtype]))
    if ivy.is_uint_dtype(dtype):
        fill_values = draw(st.integers(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        fill_values = draw(st.integers(min_value=-5, max_value=5))
    else:
        fill_values = draw(st.floats(min_value=-5, max_value=5))
    dtype_to_cast = random.choice(draw(helpers.get_dtypes("float")))
    return [dtype], dtype_and_input[1], fill_values, dtype_to_cast


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
)
def test_numpy_full(
    shape,
    input_fill_dtype,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
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
)
def test_numpy_full_like(
    input_fill_dtype,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, fill, dtype_to_cast = input_fill_dtype
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        fill_value=fill,
        dtype=dtype_to_cast,
        order="K",
        subok=True,
        shape=shape,
    )
