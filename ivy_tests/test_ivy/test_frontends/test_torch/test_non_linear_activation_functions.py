# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.torch as ivy_torch


@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(
                    tuple(
                        set(ivy_np.valid_float_dtypes).intersection(
                            set(ivy_torch.valid_float_dtypes)
                        )
                    )
                    + (None,)
                ),
                length=1,
            ),
            key="dtype",
        )
    )
    
 
# relu
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        )
    ),
    as_variable=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="functional.frontends.torch.relu"
    ),
    native_array=st.booleans(),
)
def test_torch_relu(
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
        frontend="torch",
        fn_name="relu",
        input=np.asarray(x, dtype=input_dtype),
    )


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        )
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="sigmoid"),
    native_array=st.booleans(),
)
def test_torch_sigmoid(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="torch",
        fn_name="sigmoid",
        input=np.asarray(x, dtype=input_dtype),
        out=None,
    )


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(
                set(ivy_torch.valid_float_dtypes)
            )
        ),
        min_num_dims=1,
    ),
    as_variable=st.booleans(),
    axis=st.integers(-1, 0),
    dtypes=_dtypes(),
    num_positional_args=helpers.num_positional_args(fn_name="softmax"),
    native_array=st.booleans(),
)
def test_torch_softmax(
    dtype_and_x,
    as_variable,
    axis,
    dtypes,
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
        frontend="torch",
        fn_name="softmax",
        input=np.asarray(x, dtype=input_dtype),
        dim=axis,
        dtype=dtypes[0],
    )

