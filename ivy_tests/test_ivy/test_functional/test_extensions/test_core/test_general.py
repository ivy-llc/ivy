# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# isin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("numeric"),
                                         num_arrays=2,
                                         shared_dtype=True),
    num_positional_args=helpers.num_positional_args(fn_name="isin"),
    invert=st.booleans(),
    assume_unique=st.booleans()
)
def test_isin(
    dtype_and_x,
    invert,
    assume_unique,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
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
        container_flags=container,
        fw=fw,
        fn_name="isin",
        ground_truth_backend='numpy',
        elements=elements,
        test_elements=test_elements,
        invert=invert,
        assume_unique=assume_unique,
    )
