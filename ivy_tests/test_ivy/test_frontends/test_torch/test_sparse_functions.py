# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test

inf = float("inf")


# embedding
@handle_frontend_test(
    fn_tree="torch.nn.functional.embedding",
    dtypes_indices_weights=helpers.embedding_helper(),
    max_norm=st.floats(min_value=0, max_value=5, exclude_min=True),
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
):
    dtypes, indices, weight, padding_idx = dtypes_indices_weights
    helpers.test_frontend_function(
        input_dtypes=dtypes,
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
