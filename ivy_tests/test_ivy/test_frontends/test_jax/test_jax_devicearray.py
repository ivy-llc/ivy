# global
import numpy as np
from hypothesis import given, strategies as st

# local
from ivy.functional.frontends.jax.devicearray import DeviceArray
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# reshape
@st.composite
def _reshape_helper(draw):
    # generate a shape s.t len(shape) > 0
    shape = draw(helpers.get_shape(min_num_dims=1))

    reshape_shape = draw(helpers.reshape_shapes(shape=shape))

    dtype = draw(helpers.array_dtypes(num_arrays=1))[0]
    x = draw(helpers.array_values(dtype=dtype, shape=shape))

    is_dim = draw(st.booleans())
    if is_dim:
        # generate a permutation of [0, 1, 2, ... len(shape) - 1]
        permut = draw(st.permutations(list(range(len(shape)))))
        return x, dtype, reshape_shape, permut
    else:
        return x, dtype, reshape_shape, None


@handle_cmd_line_args
@given(
    x_reshape_permut=_reshape_helper(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.DeviceArray.reshape"
    ),
)
def test_jax_instance_reshape(
    x_reshape_permut,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    x, dtype, shape, dimensions = x_reshape_permut
    helpers.test_frontend_array_instance_method(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        frontend_class=DeviceArray,
        fn_tree="DeviceArray.reshape",
        self=np.asarray(x, dtype=dtype),
        new_sizes=shape,
        dimensions=dimensions,
    )


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.DeviceArray.add",
    ),
)
def test_jax_instance_add(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_array_instance_method(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        frontend_class=DeviceArray,
        fn_tree="DeviceArray.add",
        self=np.asarray(x[0], dtype=input_dtype[0]),
        other=np.asarray(x[1], dtype=input_dtype[1]),
    )
