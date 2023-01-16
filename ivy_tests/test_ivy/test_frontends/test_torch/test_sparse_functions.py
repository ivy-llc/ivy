# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

inf = float("inf")


@st.composite
def _embedding_helper(draw):
    dtype_weight, weight = draw(
        helpers.dtype_and_values(
            available_dtypes=[
                x
                for x in draw(helpers.get_dtypes("numeric"))
                if "float" in x or "complex" in x
            ],
            min_num_dims=2,
            max_num_dims=2,
            min_dim_size=1,
            min_value=-1e04,
            max_value=1e04,
        )
    )
    num_embeddings, embedding_dim = weight[0].shape
    dtype_indices, indices = draw(
        helpers.dtype_and_values(
            available_dtypes=["int32", "int64"],
            min_num_dims=2,
            min_dim_size=1,
            min_value=0,
            max_value=num_embeddings - 1,
        ).filter(lambda x: x[1][0].shape[-1] == embedding_dim)
    )
    padding_idx = draw(st.integers(min_value=0, max_value=num_embeddings - 1))
    return dtype_indices + dtype_weight, indices[0], weight[0], padding_idx


# embedding
@handle_frontend_test(
    fn_tree="torch.nn.functional.embedding",
    dtypes_indices_weights=_embedding_helper(),
    max_norm=st.floats(min_value=0, exclude_min=True),
    p=st.one_of(
        st.sampled_from([inf, -inf]),
        st.integers(min_value=1, max_value=2),
        st.floats(min_value=1.0, max_value=2.0),
    ),
)
def test_torch_embedding(
    *,
    dtypes_indices_weights,
    max_norm,
    p,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, indices, weight, padding_idx = dtypes_indices_weights
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=indices,
        weight=weight,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=p,
    )
