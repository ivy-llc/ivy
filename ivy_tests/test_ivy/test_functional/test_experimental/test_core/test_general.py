# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test
import numpy as np

# isin


@st.composite
def _isin_data_generation_helper(draw):
    assume_unique = draw(st.booleans())
    if assume_unique:
        dtype_and_x = helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric"),
                                               num_arrays=2,
                                               shared_dtype=True)\
            .filter(lambda x: np.array_equal(x[1][0], np.unique(x[1][0])))
    else:
        dtype_and_x = helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric"),
                                               num_arrays=2,
                                               shared_dtype=True)
    return assume_unique, draw(dtype_and_x)


@handle_test(
    fn_tree="functional.experimental.isin",
    dtype_and_x=st.builds(lambda x: x[1], _isin_data_generation_helper()),
    num_positional_args=helpers.num_positional_args(fn_name="isin"),
    invert=st.booleans(),
    assume_unique=st.builds(lambda x: x[0], _isin_data_generation_helper()),
)
def test_isin(
    dtype_and_x,
    invert,
    assume_unique,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    instance_method,
    backend_fw,
):
    dtypes, values = dtype_and_x
    elements, test_elements = values
    helpers.test_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        instance_method=instance_method,
        container_flags=container_flags,
        fw=backend_fw,
        fn_name="isin",
        ground_truth_backend='numpy',
        elements=elements,
        test_elements=test_elements,
        invert=invert,
        assume_unique=assume_unique
    )
