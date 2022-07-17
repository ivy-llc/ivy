# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(st.integers(1, 4), key="num_dims"))
    num_arrays = draw(st.shared(st.integers(2, 4), key="num_arrays"))
    common_shape = draw(
        helpers.lists(st.integers(2, 3), min_size=num_dims - 1, max_size=num_dims - 1)
    )
    unique_idx = draw(helpers.integers(0, num_dims - 1))
    unique_dims = draw(
        helpers.lists(st.integers(2, 3), min_size=num_arrays, max_size=num_arrays)
    )
    xs = list()
    input_dtypes = draw(helpers.array_dtypes(shared_dtype=True))
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# concat
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    as_variable=helpers.array_bools(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cat"
    ),
    native_array=helpers.array_bools(),
    with_out=st.booleans(),
)
def test_torch_cat(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    xs = [np.asarray(x, dtype=dt) for x, dt in zip(xs, input_dtypes)]
    helpers.test_frontend_function(
        input_dtypes,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        "torch",
        "cat",
        tensors=xs,
        dim=unique_idx,
        out=None,
    )
