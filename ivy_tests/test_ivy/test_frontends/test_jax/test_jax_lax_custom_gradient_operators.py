# import ivy
# import numpy as np
# from hypothesis import given, strategies as st
from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers

# import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.jax as ivy_jax
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(dtype_and_x=helpers.dtype_and_values(ivy_jax.valid_float_dtypes))
def test_stop_gradient(dtype_and_x):
    pass
