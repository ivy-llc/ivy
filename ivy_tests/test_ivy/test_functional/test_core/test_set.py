# global
import numpy as np
from hypothesis import strategies as st, given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# unique_values
@given(array_shape=helpers.lists(
    st.integers(1, 3),
    min_size="num_dims",
    max_size="num_dims",
    size_bounds=[1, 3]),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans())
def test_unique_values(array_shape,
                       dtype,
                       as_variable,
                       with_out,
                       num_positional_args,
                       native_array,
                       container,
                       instance_method,
                       fw,
                       device):
    if fw == "torch" and ("int" in dtype or "16" in dtype):
        return

    shape = tuple(array_shape)
    x = np.random.uniform(size=shape).astype(dtype)

    helpers.test_array_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "unique_values",
        x=x)


# Still to Add #
# ---------------#

# unique_all
# unique_counts
# unique_inverse
