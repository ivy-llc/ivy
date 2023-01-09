# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _embedding_helper(draw):
    dtype_weight, weight = draw(helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
    ))
    num_embeddings, embedding_dim = weight[0].shape
    dtype_indices, indices = draw(helpers.dtype_and_values(
        available_dtypes=['int32', 'int64'],
        min_num_dims=2,
        min_dim_size=1,
        min_value=0,
        max_value=num_embeddings-1,
    ).filter(lambda x: x[1][0].shape[-1] == embedding_dim))
    return dtype_indices+dtype_weight, indices[0], weight[0]


# embedding
@handle_frontend_test(
    fn_tree="torch.nn.functional.embedding",
    dtypes_indices_weights=_embedding_helper(),
)
def test_torch_embedding(
    *,
    dtypes_indices_weights,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtypes, indices, weight = dtypes_indices_weights
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
    )
