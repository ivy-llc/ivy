# global

import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _outer_get_dtype_and_data(draw):
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("numeric"))), key="shared_dtype"
        )
    )

    shape = (
        draw(st.integers(min_value=1, max_value=5)),
        draw(st.integers(min_value=1, max_value=5)),
    )
    x = draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=shape,
        )
    )

    data1 = (input_dtype, x)

    shape = (
        draw(st.integers(min_value=1, max_value=5)),
        draw(st.integers(min_value=1, max_value=5)),
    )
    data2 = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )

    return data1, data2


# outer
@handle_cmd_line_args
@given(
    dtype_and_x=_outer_get_dtype_and_data(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.outer"
    ),
)
def test_numpy_outer(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
    fw,
):
    data1, data2 = dtype_and_x
    input_dtype1, x = data1
    input_dtype2, y = data2

    helpers.test_frontend_function(
        input_dtypes=[input_dtype1, input_dtype2],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="outer",
        a=np.array(x, dtype=input_dtype1),
        b=np.array(y, dtype=input_dtype2),
        out=None,
    )
