import hypothesis.extra.numpy as hnp
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _broadcastable_trio(draw):
    dtype = draw(st.sampled_from(ivy_np.valid_numeric_dtypes))

    shapes_st = hnp.mutually_broadcastable_shapes(num_shapes=3, min_dims=1, min_side=1)
    cond_shape, x1_shape, x2_shape = draw(shapes_st).input_shapes
    cond = draw(hnp.arrays(hnp.boolean_dtypes(), cond_shape))
    x1 = draw(hnp.arrays(dtype, x1_shape))
    x2 = draw(hnp.arrays(dtype, x2_shape))
    return cond, x1, x2, dtype


@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    dtype=st.sampled_from(ivy_np.valid_numeric_dtypes + (None,)),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.nonzero"
    ),
)

@handle_cmd_line_args
def test_numpy_nonzero(
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    input_dtype = [input_dtype]

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_name="nonzero",
        x=np.asarray(x, dtype=input_dtype[0]),
    )
