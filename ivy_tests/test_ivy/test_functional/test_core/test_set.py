# global
import numpy as np
from hypothesis import strategies as st, given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# unique_values
@given(
    array_shape=helpers.lists(
        st.integers(1, 3), min_size="num_dims", max_size="num_dims", size_bounds=[1, 3]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_unique_values(
    array_shape,
    input_dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    if fw == "torch" and ("int" in input_dtype or "16" in input_dtype):
        return

    shape = tuple(array_shape)
    x = np.random.uniform(size=shape).astype(input_dtype)

    helpers.test_array_function(input_dtype, as_variable, with_out, num_positional_args, native_array, container,
                                instance_method, fw, "unique_values", x=x)


@given(
    array_shape=helpers.lists(
        st.integers(1, 5), min_size="num_dims", max_size="num_dims", size_bounds=[1, 5]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_unique_all(
    array_shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    if fw == "torch" and ("int" in input_dtype or "16" in input_dtype):
        return

    shape = tuple(array_shape)
    x = np.random.uniform(size=shape).astype(input_dtype)

    helpers.test_array_function(input_dtype, as_variable, False, num_positional_args, native_array, container,
                                instance_method, fw, "unique_all", x=x)


@given(
    array_shape=helpers.lists(
        st.integers(2, 5), min_size="num_dims", max_size="num_dims", size_bounds=[2, 5]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_unique_counts(
    array_shape,
    input_dtype,
    data,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    if fw == "torch" and ("int" in input_dtype or "16" in input_dtype):
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))

    helpers.test_array_function(input_dtype, as_variable, False, num_positional_args, native_array, container,
                                instance_method, fw, "unique_counts", x=x)


@given(
    array_shape=helpers.lists(
        st.integers(2, 5), min_size="num_dims", max_size="num_dims", size_bounds=[2, 5]
    ),
    input_dtype=st.sampled_from(ivy_np.valid_numeric_dtypes),
    data=st.data(),
    as_variable=st.booleans(),
    num_positional_args=st.integers(0, 2),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_unique_inverse(
    array_shape,
    input_dtype,
    data,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):

    if fw == "torch" and ("int" in input_dtype or "16" in input_dtype):
        return

    x = data.draw(helpers.nph.arrays(shape=array_shape, dtype=input_dtype))

    helpers.test_array_function(input_dtype, as_variable, False, num_positional_args, native_array, container,
                                instance_method, fw, "unique_inverse", x=x)
