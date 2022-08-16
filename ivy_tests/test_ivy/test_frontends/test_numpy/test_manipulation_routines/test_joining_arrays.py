# global
import numpy as np
from hypothesis import given, settings, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    input_dtypes = draw(helpers.array_dtypes())
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


@st.composite
def _dtype_n_with_out(draw):
    dtype = draw(st.sampled_from(ivy_np.valid_float_dtypes + (None,)))
    if dtype is None:
        return dtype, draw(st.booleans())
    return dtype, False


# concat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    as_variable=helpers.array_bools(),
    dtype_n_with_out=_dtype_n_with_out(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.concatenate"
    ),
    native_array=helpers.array_bools(),
)
@settings(max_examples=1)
def test_numpy_concatenate(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    dtype_n_with_out,
    num_positional_args,
    native_array,
    fw,
):
    dtype, with_out = dtype_n_with_out
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="concatenate",
        arrays=xs,
        axis=unique_idx,
        out=None,
        dtype=dtype,
    )
