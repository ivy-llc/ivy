# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers

@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes= ivy_np.valid_float_dtypes[1:],
        min_num_dims= 2,
        min_value= -10000000000.0,
        max_value= 10000000000.0
    ),
    as_variable= st.booleans(),
    native_array= st.booleans(),
    num_positional_args= helpers.num_positional_args(
      fn_name= "ivy.functional.frontends.tensorflow.matrix_rank"
    ),
    tolr= st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
    data= st.data(),
)
def test_matrix_rank(
  *,
  data,
  dtype_x,
  as_variable,
  num_positional_args,
  native_array,
  tolr,
  fw
):
    input_dtype , x = dtype_x
    helpers.test_frontend_function(
        input_dtypes= input_dtype,
        as_variable_flags= as_variable,
        with_out= False,
        num_positional_args= num_positional_args,
        native_array_flags= native_array,
        fw= fw,
        frontend= "tensorflow",
        fn_name= "linalg.matrix_rank",
        atol= 1.0,
        a= np.asarray(x, dtype=input_dtype),
        tol= tolr
    )
