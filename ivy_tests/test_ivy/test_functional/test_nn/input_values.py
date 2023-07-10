from hypothesis import given
import hypothesis.strategies as st

# Import the function to test
from ivy_tests.test_ivy.helpers import handle_test, get_dtypes
from ivy_tests.test_ivy import test_layer_norm


# Define the strategy for generating the input values
@st.composite
def input_strategy(draw):
    available_dtypes = get_dtypes("float")
    values_tuple = draw(_generate_data_layer_norm(
        available_dtypes=available_dtypes,
    ))
    new_std = draw(st.floats(min_value=0.01, max_value=0.1))
    eps = draw(st.floats(min_value=0.01, max_value=0.1))
    test_flags = draw(st.lists(st.booleans()))
    backend_fw = draw(st.sampled_from(["numpy", "jax", "tensorflow"]))
    fn_name = "functional.ivy.layer_norm"
    on_device = draw(st.sampled_from(["cpu", "gpu"]))
    ground_truth_backend = draw(st.sampled_from(["numpy", "jax", "tensorflow"]))

    return {
        "values_tuple": values_tuple,
        "new_std": new_std,
        "eps": eps,
        "test_flags": test_flags,
        "backend_fw": backend_fw,
        "fn_name": fn_name,
        "on_device": on_device,
        "ground_truth_backend": ground_truth_backend
    }


# Define the test function
@given(input_strategy())
def test_code(input_data):
    test_layer_norm(**input_data)


# Run the test
test_code()

