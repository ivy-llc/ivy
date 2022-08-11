import hypothesis.extra.numpy as hnp
from hypothesis import given, strategies as st
import numpy as np

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
    broadcastables=_broadcastable_trio(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.where"
    ),
    data=st.data(),
)
@handle_cmd_line_args
def test_numpy_where(
    *,
    data,
    broadcastables,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    cond, x1, x2, dtype = broadcastables

    helpers.test_frontend_function(
        input_dtypes=["bool", dtype, dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="where",
        cond=cond,
        x1=x1,
        x2=x2,
    )


@given(
    dtype_and_a=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes,),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.nonzero"
    ),
    data=st.data(),
)
@handle_cmd_line_args
def test_numpy_nonzero(
    *,
    data,
    dtype_and_a,
    native_array,
    num_positional_args,
    fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=False,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_name="nonzero",
        a=np.asarray(a, dtype=dtype),
    )
