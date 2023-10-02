# global
from hypothesis import given, settings

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy.functional.frontends.jax.func_wrapper import (
    inputs_to_ivy_arrays,
    outputs_to_frontend_arrays,
    to_ivy_arrays_and_back,
)
from ivy.functional.frontends.jax.array import Array
import ivy.functional.frontends.jax as jax_frontend


# --- Helpers --- #
# --------------- #


def _fn(x, check_default=False):
    if check_default and jax_frontend.config.jax_enable_x64:
        ivy.utils.assertions.check_equal(
            ivy.default_float_dtype(), "float64", as_array=False
        )
        ivy.utils.assertions.check_equal(
            ivy.default_int_dtype(), "int64", as_array=False
        )
    return x


# --- Main --- #
# ------------ #


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_inputs_to_ivy_arrays(dtype_and_x, backend_fw):
    ivy.set_backend(backend_fw)
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = inputs_to_ivy_arrays(_fn)(input_ivy)
    assert isinstance(output, ivy.Array)
    assert input_ivy.dtype == output.dtype
    assert ivy.all(input_ivy == output)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    output = inputs_to_ivy_arrays(_fn)(input_native)
    assert isinstance(output, ivy.Array)
    assert ivy.as_ivy_dtype(input_native.dtype) == output.dtype
    assert ivy.all(ivy.equal(input_native, output.data))

    # check for frontend array
    input_frontend = Array(x[0])
    output = inputs_to_ivy_arrays(_fn)(input_frontend)
    assert isinstance(output, ivy.Array)
    assert input_frontend.dtype == output.dtype
    assert ivy.all(input_frontend.ivy_array == output)
    ivy.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        num_arrays=2,
    ),
)
@settings(max_examples=200)
def test_jax_numpy_promote_types_of_jax_input(dtype_and_x, backend_fw):
    x_dtype, x = dtype_and_x
    x1, x2 = x
    ivy.set_backend(backend_fw)
    # convert inputs to ivy arrays
    input_ivy1 = ivy.array(x1, dtype=x_dtype[0])
    input_ivy2 = ivy.array(x2, dtype=x_dtype[1])

    # check promoted type of arrays
    jax_frontend.config.jax_enable_x64 = True
    promoted_type1, promoted_type2 = jax_frontend.numpy.promote_types_of_jax_inputs(
        input_ivy1, input_ivy2
    )
    assert promoted_type1.dtype == promoted_type2.dtype

    try:
        import jax

        x1_dtype, x2_dtype = jax.numpy.dtype(x_dtype[0]), jax.numpy.dtype(x_dtype[1])
        promoted_type_jax = jax.numpy.promote_types(x1_dtype, x2_dtype)
    except ImportError:
        promoted_type_jax = None
    if promoted_type_jax is not None:
        assert str(promoted_type1.dtype) == str(promoted_type_jax)
    jax_frontend.config.jax_enable_x64 = False

    ivy.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_outputs_to_frontend_arrays(dtype_and_x, backend_fw):
    ivy.set_backend(backend_fw)
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = outputs_to_frontend_arrays(_fn)(input_ivy, check_default=True)
    assert isinstance(output, Array)
    assert input_ivy.dtype == output.dtype
    assert ivy.all(input_ivy == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []
    ivy.previous_backend()


@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ),
)
def test_jax_to_ivy_arrays_and_back(dtype_and_x, backend_fw):
    ivy.set_backend(backend_fw)
    x_dtype, x = dtype_and_x

    # check for ivy array
    input_ivy = ivy.array(x[0], dtype=x_dtype[0])
    output = to_ivy_arrays_and_back(_fn)(input_ivy, check_default=True)
    assert isinstance(output, Array)
    assert input_ivy.dtype == output.dtype
    assert ivy.all(input_ivy == output.ivy_array)

    # check for native array
    input_native = ivy.native_array(input_ivy)
    output = to_ivy_arrays_and_back(_fn)(input_native, check_default=True)
    assert isinstance(output, Array)
    assert ivy.as_ivy_dtype(input_native.dtype) == output.dtype
    assert ivy.all(ivy.equal(input_native, output.ivy_array.data))

    # check for frontend array
    input_frontend = Array(x[0])
    output = to_ivy_arrays_and_back(_fn)(input_frontend, check_default=True)
    assert isinstance(output, Array)
    assert str(input_frontend.dtype) == str(output.dtype)
    assert ivy.all(input_frontend.ivy_array == output.ivy_array)

    assert ivy.default_float_dtype_stack == ivy.default_int_dtype_stack == []
    ivy.previous_backend()
