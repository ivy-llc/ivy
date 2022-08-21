<<<<<<< HEAD
# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.jax as ivy_jax


# add
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_jax.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
    as_variable=helpers.list_of_length(x=st.booleans(), length=2),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.add"
    ),
    native_array=helpers.list_of_length(x=st.booleans(), length=2),
)
def test_jax_lax_add(
    dtype_and_x,
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
        fw=fw,
        frontend="jax",
        fn_name="lax.add",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        y=np.asarray(x[1], dtype=input_dtype[1]),
    )


# tan
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_jax.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.tan"
    ),
    native_array=st.booleans(),
)
def test_jax_lax_tan(
    dtype_and_x,
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
        fw=fw,
        frontend="jax",
        fn_name="lax.tan",
        x=np.asarray(x, dtype=input_dtype),
    )


# noinspection DuplicatedCode
=======
>>>>>>> d5b6172a0147071eaf8aac982234ec9a8fe8ef16
@st.composite
def _sample_castable_numeric_dtype(draw):
    dtype = draw(_dtypes())[0]
<<<<<<< HEAD
    if ivy.is_uint_dtype(dtype):
        return draw(st.integers(0, 5))
    elif ivy.is_int_dtype(dtype):
        return draw(st.integers(-5, 5))
    return draw(st.floats(-5, 5))


@given(
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.full"
    ),
)
def test_jax_lax_full(
    shape,
    fill_value,
    dtypes,
    num_positional_args,
    fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        fw=fw,
        frontend="jax",
        fn_name="lax.full",
        shape=shape,
        fill_value=fill_value,
        dtype=dtypes[0],
    )


# abs
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_jax.valid_float_dtypes),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.abs"
    ),
    native_array=st.booleans(),
)
def test_jax_lax_abs(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

=======
    return draw(
        st.sampled_from(ivy.valid_numeric_dtypes).filter(
            lambda x: ivy.can_cast(dtype, x)
        )
    )


@handle_cmd_line_args
@given(
    dtype_and_x=_dtype_and_values(
        num_arrays=1,
        min_num_dims=1,
        min_value=-5,
        max_value=5,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.convert_element_type"
    ),
    new_dtype=_sample_castable_numeric_dtype(),
)
def test_jax_lax_convert_element_type(
    dtype_and_x, as_variable, num_positional_args, native_array, fw, new_dtype
):
    input_dtype, x = dtype_and_x
>>>>>>> d5b6172a0147071eaf8aac982234ec9a8fe8ef16
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
<<<<<<< HEAD
        fn_name="lax.abs",
        x=np.asarray(x, dtype=input_dtype),
=======
        fn_tree="lax.convert_element_type",
        operand=np.asarray(x, dtype=input_dtype),
        new_dtype=new_dtype,
>>>>>>> d5b6172a0147071eaf8aac982234ec9a8fe8ef16
    )
