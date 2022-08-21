@st.composite
def _sample_castable_numeric_dtype(draw):
    dtype = draw(_dtypes())[0]
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
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.convert_element_type",
        operand=np.asarray(x, dtype=input_dtype),
        new_dtype=new_dtype,
    )
