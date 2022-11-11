from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    n_rows=helpers.ints(min_value=0, max_value=10),
    n_cols=st.none() | helpers.ints(min_value=0, max_value=10),
    k=helpers.ints(min_value=-10, max_value=10),
    num_positional_args=helpers.num_positional_args(fn_name="triu_indices"),
)
def test_triu_indices(
    *,
    n_rows,
    n_cols,
    k,
    device,
    num_positional_args,
    fw,
):
    helpers.test_function(
        input_dtypes=["int32"],
        as_variable_flags=[False],
        with_out=None,
        num_positional_args=num_positional_args,
        native_array_flags=[False],
        container_flags=[False],
        instance_method=False,
        fw=fw,
        fn_name="triu_indices",
        n_rows=n_rows,
        n_cols=n_cols,
        k=k,
        device=device,
    )


# vorbis_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        min_num_dims=1,
        max_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="vorbis_window"),
)
def test_vorbis_window(
    dtype_and_x,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vorbis_window",
        x=x[0],
        dtype=input_dtype,
    )


# hann_window
@handle_cmd_line_args
@given(
    window_length=helpers.ints(min_value=1, max_value=10),
    input_dtype=helpers.get_dtypes("integer"),
    periodic=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="hann_window"),
    dtype=helpers.get_dtypes("float"),
)
def test_hann_window(
    window_length,
    input_dtype,
    periodic,
    dtype,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="hann_window",
        window_length=window_length,
        periodic=periodic,
        dtype=dtype,
    )


# kaiser_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=(1, 1),
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    beta=st.floats(min_value=0, max_value=5),
    dtype=helpers.get_dtypes("float"),
    num_positional_args=helpers.num_positional_args(fn_name="kaiser_window"),
)
def test_kaiser_window(
    dtype_and_x,
    periodic,
    beta,
    dtype,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="kaiser_window",
        window_length=x[0],
        periodic=periodic,
        beta=beta,
        dtype=dtype,
    )


# kaiser_bessel_derived_window
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=(1, 1),
        min_value=1,
        max_value=10,
    ),
    periodic=st.booleans(),
    beta=st.floats(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
    num_positional_args=helpers.num_positional_args(
        fn_name="kaiser_bessel_derived_window"
    ),
)
def test_kaiser_bessel_derived_window(
    dtype_and_x,
    periodic,
    beta,
    dtype,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="kaiser_bessel_derived_window",
        window_length=x[0],
        periodic=periodic,
        beta=beta,
        dtype=dtype,
    )


# hamming_window
@handle_cmd_line_args
@given(
    window_length=helpers.ints(min_value=1, max_value=10),
    input_dtype=helpers.get_dtypes("integer"),
    periodic=st.booleans(),
    alpha=st.floats(min_value=1, max_value=5),
    beta=st.floats(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
    num_positional_args=helpers.num_positional_args(fn_name="hamming_window"),
)
def test_hamming_window(
    window_length,
    input_dtype,
    periodic,
    alpha,
    beta,
    dtype,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="hamming_window",
        window_length=window_length,
        periodic=periodic,
        alpha=alpha,
        beta=beta,
        dtype=dtype,
    )
