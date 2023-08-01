# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# log_poisson_loss
@handle_test(
    fn_tree="functional.ivy.experimental.log_poisson_loss",
    dtype_and_targets=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=3,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    dtype_and_log_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        small_abs_safety_factor=4,
        safety_factor_scale="log",
        min_value=0,
        max_value=3,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=3,
    ),
    compute_full_loss=st.sampled_from([True, False]),
    test_with_out=st.just(False),
)
def test_log_poisson_loss(
    *,
    dtype_and_targets,
    dtype_and_log_input,
    compute_full_loss,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    targets_dtype, targets = dtype_and_targets
    log_input_dtype, log_input = dtype_and_log_input
    helpers.test_function(
        input_dtypes=targets_dtype + log_input_dtype,
        test_flags=test_flags,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        targets=targets[0],
        log_input=log_input[0],
        compute_full_loss=compute_full_loss,
        atol_=1e-2,
    )


# Test cases for gumbel_softmax
@handle_test(
    fn_tree="functional.ivy.experimental.gumbel_softmax",
    logits=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-10,
        max_value=10,
        allow_inf=False,
        exclude_min=True,
        exclude_max=True,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=5,
    ),
    tau=helpers.floats(min_value=0.1, max_value=10),
    hard=st.booleans(),
    eps=helpers.floats(min_value=1e-20, max_value=1e-5),
    dim=helpers.ints(min_value=-1, max_value=1),
)
def test_gumbel_softmax(
    logits,
    tau,
    hard,
    eps,
    dim,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    dtype, inputs = logits
    # Call the gumbel_softmax function with the given inputs and parameters
    helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        logits=inputs[0],
        tau=tau,
        hard=hard,
        eps=eps,
        dim=dim,
    )
