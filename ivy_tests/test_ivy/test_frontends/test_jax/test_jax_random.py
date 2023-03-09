# global
import jax
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _prepare_n_probs_shape_seed(draw):
    prob_dtype = draw(helpers.get_dtypes("float", full=False))
    batch_size = draw(helpers.ints(min_value=1, max_value=5))
    num_samples = draw(helpers.ints(min_value=1, max_value=20))
    seed = draw(helpers.ints(min_value=1, max_value=20))
    probs = draw(
        helpers.array_values(
            dtype=prob_dtype[0],
            shape=[batch_size, num_samples],
            min_value=1.0013580322265625e-05,
            max_value=1.0,
            exclude_min=True,
            large_abs_safety_factor=2,
            safety_factor_scale="linear",
        )
    )
    return prob_dtype, batch_size, num_samples, seed, probs


@handle_frontend_test(
    fn_tree="jax.random.bernoulli",
    everything=_prepare_n_probs_shape_seed(),
    test_with_out=st.just(False),
)
def test_jax_random_bernoulli(
    *,
    everything,
    test_flags,
    on_device,
    fn_tree,
    frontend,
):
    prob_dtype, batch_size, num_samples, seed, probs = everything
    helpers.test_frontend_function(
        input_dtypes=prob_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        key=jax.random.PRNGKey(seed),
        p=probs,
        shape=(batch_size, num_samples),
    )

