from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Cosine Similarity
@handle_frontend_test(
    fn_tree="torch.nn.functional.cosine_similarity",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=2,
        max_value=5,
        min_dim_size=2,
        shared_dtype=True,
        num_arrays=2,
    ),
    dim=st.integers(min_value=-1, max_value=0),
)
def test_torch_cosine_similarity(
    *,
    d_type_and_x,
    dim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        x1=x[0],
        x2=x[1],
        dim=dim,
    )


# Pairwise Distance
@handle_frontend_test(
    fn_tree="torch.nn.functional.pairwise_distance",
    d_type_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_dim_size=2,
        max_dim_size=5,
        min_num_dims=2,
        min_value=2,
        max_value=5,
        allow_inf=False,
    ),
    p=st.integers(min_value=0, max_value=2),
    keepdim=st.booleans(),
)
def test_torch_pairwise_distance(
    *,
    d_type_and_x,
    p,
    keepdim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
):
    dtype, x = d_type_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        x1=x[0],
        x2=x[1],
        p=p,
        keepdim=keepdim,
    )
