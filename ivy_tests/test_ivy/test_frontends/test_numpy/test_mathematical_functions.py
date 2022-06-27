# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# add
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes, 2),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    where=st.sampled_from(
        (helpers.dtype_and_values(
            ("bool",), shape=st.shared(helpers.get_shape(), key="shape")), True)),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.add"),
    native_array=st.booleans(),
)
def test_numpy_add(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        "numpy",
        "add",
        x1=np.asarray(x[0], dtype=input_dtype[0]),
        x2=np.asarray(x[1], dtype=input_dtype[1]),
        out=None,
        where=where,
        casting='same_kind',
        order='k',
        dtype=dtype,
        subok=True
    )


# tan
@given(
    dtype_and_x=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
<<<<<<< Updated upstream
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    where=st.sampled_from(
        (helpers.dtype_and_values(
            ("bool",), shape=st.shared(helpers.get_shape(), key="shape")), True)),
=======
>>>>>>> Stashed changes
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.tan"),
    native_array=st.booleans(),
)
def test_numpy_tan(
    dtype_and_x,
<<<<<<< Updated upstream
    dtype,
    where,
=======
>>>>>>> Stashed changes
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        "numpy",
        "tan",
        x=np.asarray(x, dtype=input_dtype),
        out=None,
<<<<<<< Updated upstream
        where=where,
        casting='same_kind',
        order='k',
        dtype=dtype,
=======
        where=True,
        casting='same_kind',
        order='k',
        dtype=None,
>>>>>>> Stashed changes
        subok=True
    )
