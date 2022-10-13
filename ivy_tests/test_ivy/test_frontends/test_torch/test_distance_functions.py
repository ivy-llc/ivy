# global
from hypothesis import given
from hypothesis.control import reject

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.hypothesis_helpers.array_helpers as array_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# cosine_similarity
@handle_cmd_line_args
@given(
    dtype_input_axis=array_helpers.dtype_values_axis(
        available_dtypes=["float16", "float32", "float64"],
        num_arrays=2,
        min_num_dims=1,
        min_value=-1e15,
        max_value=1e15,
        abs_smallest_val=1e-15,
        valid_axis=True,
        force_int_axis=True,
    ),
    eps=helpers.floats(min_value=0, max_value=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.cosine_similarity"
    ),
)
def test_torch_cosine_similarity(
    dtype_input_axis,
    eps,
    as_variable,
    num_positional_args,
    native_array,
):
    dtype, values, axis = dtype_input_axis
    try:
        pow(values[0], 2).sum(axis)
        pow(values[1], 2).sum(axis)
    except OverflowError:
        reject()
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.cosine_similarity",
        x1=values[0],
        x2=values[1],
        dim=axis,
        eps=eps,
    )
