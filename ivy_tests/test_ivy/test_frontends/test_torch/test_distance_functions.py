from hypothesis import assume, given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


def _filter_dtypes(input_dtype):
    assume(("bfloat16" not in input_dtype) and ("float16" not in input_dtype))


# Cosine Similarity
@handle_cmd_line_args
@given(
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=2,
        max_value=5,
        min_dim_size=2,
        shared_dtype=True,
        num_arrays=2,
    ),
    dim=st.integers(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.nn.functional.cosine_similarity"
    ),
)
def test_torch_cosine_similarity(
    d_type_and_x,
    dim,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
):
    dtype, x = d_type_and_x
    _filter_dtypes(dtype)
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="nn.functional.cosine_similarity",
        rtol=1e-01,
        x1=x[0],
        x2=x[1],
        dim=dim,
    )
