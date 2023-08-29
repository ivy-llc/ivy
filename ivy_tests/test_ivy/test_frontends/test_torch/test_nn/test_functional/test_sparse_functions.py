# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

inf = float("inf")


# --- Helpers --- #
# --------------- #


@st.composite
def get_dtype_num_classes(draw):
    dtype_and_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            num_arrays=1,
            min_value=1,
            max_value=10,
            max_num_dims=0,
        )
    )
    input_dtype, x = dtype_and_x
    print(max(x))
    num_classes = draw(st.integers(min_value=max(x) + 1, max_value=10))

    return (num_classes, dtype_and_x)


# embedding
@handle_frontend_test(
    fn_tree="torch.nn.functional.embedding",
    dtypes_indices_weights=helpers.embedding_helper(),
    max_norm=st.floats(min_value=0.1, max_value=5, exclude_min=True),
    p=st.one_of(
        st.sampled_from([inf, -inf]),
        st.integers(min_value=1, max_value=2),
        st.floats(min_value=1.0, max_value=2.0),
    ),
    test_with_out=st.just(False),
)
def test_torch_embedding(
    *,
    dtypes_indices_weights,
    max_norm,
    p,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtypes, indices, weight, padding_idx = dtypes_indices_weights
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=indices,
        weight=weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=p,
    )


# one_hot
@handle_frontend_test(
    fn_tree="torch.nn.functional.one_hot",
    num_classes_dtype_x_axis=get_dtype_num_classes(),
)
def test_torch_one_hot(
    *,
    num_classes_dtype_x_axis,
    frontend,
    fn_tree,
    test_flags,
    backend_fw,
    on_device,
):
    num_classes, values = num_classes_dtype_x_axis
    input_dtype, x = values
    test_flags.num_positional_args += 1
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        tensor=x[0],
        num_classes=num_classes,
    )
