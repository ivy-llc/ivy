# global
import math

import numpy as np
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


# helpers
@st.composite
def _get_repeat_interleaves_args(
    draw, *, available_dtypes, valid_axis, max_num_dims, max_dim_size
):
    values_dtype, values, axis, shape = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            valid_axis=valid_axis,
            force_int_axis=True,
            shape=draw(
                helpers.get_shape(
                    allow_none=False,
                    min_num_dims=0,
                    max_num_dims=max_num_dims,
                    min_dim_size=1,
                    max_dim_size=max_dim_size,
                )
            ),
            ret_shape=True,
        )
    )

    if axis is None:
        generate_repeats_as_integer = draw(st.booleans())
        num_repeats = 1 if generate_repeats_as_integer else math.prod(tuple(shape))
    else:
        num_repeats = shape[axis]

    repeats_dtype, repeats = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=10,
            shape=[num_repeats],
        )
    )

    # Output size is an optional parameter accepted by Torch for optimisation
    use_output_size = draw(st.booleans())
    output_size = np.sum(repeats) if use_output_size else None

    return [values_dtype, repeats_dtype], values, repeats, axis, output_size


# --- Main --- #
# ------------ #


# stft
@handle_frontend_test(
    fn_tree="torch.stft",
    dtypes_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=2,
        min_value=1,
    ),
    test_with_out=st.just(False),
)
def test_torch_stft(
    dtypes_and_x,
    frontend,
    fn_tree,
    on_device,
    test_flags,
    backend_fw,
):
    input_dtypes, x = dtypes_and_x
    helpers.test_frontend_function(
        input_dtypes=["float64"],
        frontend=frontend,
        fn_tree=fn_tree,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        input=x[0],
    )
