import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np

@st.composite
def _broadcastable_trio(draw):
    dtype = draw(st.sampled_from(ivy_np.valid_numeric_dtypes))

    shapes_st = hnp.mutually_broadcastable_shapes(num_shapes=3, min_dims=1, min_side=1)
    cond_shape, x1_shape, x2_shape = draw(shapes_st).input_shapes
    cond = draw(hnp.arrays(hnp.boolean_dtypes(), cond_shape))
    x1 = draw(hnp.arrays(dtype, x1_shape))
    x2 = draw(hnp.arrays(dtype, x2_shape))
    return cond, x1, x2

@given(
    broadcastables=_broadcastable_trio(),
    num_positional_args=helpers.num_positional_args(fn_name="where"),
    data=st.data(),
)
def test_numpy_where(
    *,
    data,
    broadcastables,
    as_variable,
    num_positional_args,
    native_array,
    instance_method,
    fw,
):
    cond, x1, x2 = broadcastables

    helpers.test_frontend_function(
        as_variable_flags=as_variable,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        instance_method=instance_method,
        fw=fw,
        frontend="numpy",
        fn_name="where",
        condition=cond,
        x1=x1,
        x2=x2,
    )
