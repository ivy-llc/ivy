# global
from hypothesis import strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# conv2D
@handle_frontend_test(
  fn_tree = "paddle.nn.functional.conv2D",
  dtype_and_x = helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
  test_with_out = st.just(False),
  filters = helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("int")),
  strides = helpers.ints(min_value=1, max_value=10),
  padding = helpers.ints(min_value=0,max_value=10),
)
def test_paddle_conv2D(
  *,
  dtype_and_x,
  filters,
  strides,
  padding,
  frontend,
  test_flags,
  fn_tree,
  on_device,
):
  input_dtype, x = dtype_and_x
  helpers.test_frontend_function(
    input_dtypes=input_dtype,
    frontend=frontend,
    test_flags=test_flags,
    fn_tree=fn_tree,
    on_device=on_device,
    x=x[0],
    filters=filters,
    strides=strides,
    padding=padding,
  )
