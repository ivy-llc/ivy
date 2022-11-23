# global
import ivy
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(draw(helpers.get_dtypes("valid"))), length=1
            ),
            key="dtype",
        )
    )


# empty
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.empty"
    ),
)
def test_cupy_empty(
    shape,
    dtypes,
    num_positional_args,
    fw,
    native_array,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="empty",
        test_values=False,
        shape=shape,
        dtype=dtypes[0],
    )


# empty_like
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.empty_like"
    ),
)
def test_cupy_empty_like(
    dtype_and_x,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="empty_like",
        test_values=False,
        prototype=x[0],
        dtype=input_dtype,
        order="K",
        subok=True,
        shape=shape,
    )


# eye
@handle_cmd_line_args
@given(
    rows=helpers.ints(min_value=3, max_value=10),
    cols=helpers.ints(min_value=3, max_value=10),
    k=helpers.ints(min_value=0, max_value=2),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.eye"
    ),
)
def test_cupy_eye(
    rows,
    cols,
    k,
    dtypes,
    num_positional_args,
    fw,
    native_array,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="eye",
        N=rows,
        M=cols,
        k=k,
        dtype=dtypes[0],
    )


# identity
@handle_cmd_line_args
@given(
    n=helpers.ints(min_value=1, max_value=10),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.identity"
    ),
)
def test_cupy_identity(
    n,
    dtypes,
    num_positional_args,
    fw,
    native_array,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="identity",
        n=n,
        dtype=dtypes[0],
    )


# ones
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.ones"
    ),
)
def test_cupy_ones(
    shape,
    dtypes,
    num_positional_args,
    fw,
    native_array,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="ones",
        shape=shape,
        dtype=dtypes[0],
    )


# ones_like
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.ones_like"
    ),
)
def test_cupy_ones_like(
    dtype_and_x,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="ones_like",
        a=x[0],
        dtype=input_dtype,
        order="K",
        subok=True,
        shape=shape,
    )


# zeros
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.zeros"
    ),
)
def test_cupy_zeros(
    shape,
    dtypes,
    num_positional_args,
    fw,
    native_array,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="zeros",
        shape=shape,
        dtype=dtypes[0],
    )


# zeros_like
@handle_cmd_line_args
@given(
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
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.zeros_like"
    ),
)
def test_cupy_zeros_like(
    dtype_and_x,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="zeros_like",
        a=x[0],
        dtype=input_dtype,
        order="K",
        subok=True,
        shape=shape,
    )


# full and full_like helpers
@st.composite
def _dtype_and_fill_value(draw):
    dtype = draw(helpers.get_dtypes("numeric", full=False))
    if ivy.is_uint_dtype(dtype):
        return dtype, draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return dtype, draw(helpers.ints(min_value=-5, max_value=5))
    return dtype, draw(helpers.floats(min_value=-5, max_value=5))


# full
@handle_cmd_line_args
@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype_and_fill_value=_dtype_and_fill_value(),
    # dtypes=helpers.get_dtypes("numeric"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.full"
    ),
)
def test_cupy_full(
    shape,
    dtype_and_fill_value,
    # dtypes,
    num_positional_args,
    fw,
    native_array,
):
    dtype, fill_value = dtype_and_fill_value
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="full",
        shape=shape,
        fill_value=fill_value,
        dtype=dtype,
    )


# full_like
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype_and_fill_value=_dtype_and_fill_value(),
    # dtypes=helpers.get_dtypes("numeric", full=False),
    shape=helpers.get_shape(
        allow_none=True,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.cupy.full_like"
    ),
)
def test_cupy_full_like(
    dtype_and_x,
    dtype_and_fill_value,
    # dtypes,
    shape,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    dtype, fill_value = dtype_and_fill_value
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="cupy",
        fn_tree="full_like",
        a=x[0],
        fill_value=fill_value,
        dtype=dtype,
        order="K",
        subok=True,
        shape=shape,
    )
