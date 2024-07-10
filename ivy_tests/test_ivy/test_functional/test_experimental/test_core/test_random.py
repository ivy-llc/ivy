# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test, BackendHandler


@handle_test(
    fn_tree="functional.ivy.experimental.bernoulli",
    dtype_and_probs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        min_value=0,
        max_value=1,
        min_num_dims=0,
    ),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
    ground_truth_backend="torch",
)
def test_bernoulli(
    *, dtype_and_probs, seed, test_flags, backend_fw, fn_name, on_device
):
    dtype, probs = dtype_and_probs
    # torch doesn't support half precision on CPU
    assume(
        not ("torch" in str(backend_fw) and "float16" in dtype and on_device == "cpu")
    )
    ret_np_flat_from_target, ret_np_from_gt_flat = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        test_values=False,
        return_flat_np_arrays=True,
        probs=probs[0],
        logits=None,
        shape=None,
        seed=seed,
    )
    helpers.assert_same_type_and_shape([ret_np_flat_from_target, ret_np_from_gt_flat])


# beta
@handle_test(
    fn_tree="functional.ivy.experimental.beta",
    dtype_and_alpha_beta=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=1,
        max_num_dims=2,
        num_arrays=2,
        exclude_min=True,
    ),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_beta(
    *,
    dtype_and_alpha_beta,
    seed,
    backend_fw,
    fn_name,
    on_device,
    test_flags,
):
    dtype, alpha_beta = dtype_and_alpha_beta
    if "float16" in dtype:
        return
    ret, ret_gt = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        test_values=False,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        alpha=alpha_beta[0],
        beta=alpha_beta[1],
        shape=None,
        dtype=dtype[0],
        seed=seed,
    )
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(
        ret=ret_gt, backend=test_flags.ground_truth_backend
    )
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        for u, v in zip(ret, ret_gt):
            assert ivy_backend.all(u >= 0)
            assert ivy_backend.all(u <= 1)
            assert ivy_backend.all(v >= 0)
            assert ivy_backend.all(v <= 1)


# dirichlet
@handle_test(
    fn_tree="functional.ivy.experimental.dirichlet",
    dtype_and_alpha=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(
            st.integers(min_value=2, max_value=5),
        ),
        min_value=0,
        max_value=100,
        exclude_min=True,
    ),
    size=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=5)
    ),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_dirichlet(
    *, dtype_and_alpha, size, seed, test_flags, backend_fw, fn_name, on_device
):
    dtype, alpha = dtype_and_alpha
    assume("bfloat16" not in dtype)

    def call():
        return helpers.test_function(
            input_dtypes=dtype,
            test_flags=test_flags,
            test_values=False,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            on_device=on_device,
            alpha=alpha[0],
            size=size,
            seed=seed,
        )

    ret, ret_gt = call()
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        if seed:
            ret1, ret_gt1 = call()
            assert ivy_backend.any(ret == ret1)
        ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
        ret_gt = helpers.flatten_and_to_np(
            ret=ret_gt, backend=test_flags.ground_truth_backend
        )
        for u, v in zip(ret, ret_gt):
            u, v = ivy_backend.array(u), ivy_backend.array(v)
            assert ivy_backend.all(
                ivy_backend.sum(u, axis=-1) == ivy_backend.sum(v, axis=-1)
            )
            assert ivy_backend.all(u >= 0)
            assert ivy_backend.all(u <= 1)
            assert ivy_backend.all(v >= 0)
            assert ivy_backend.all(v <= 1)


# gamma
@handle_test(
    fn_tree="functional.ivy.experimental.gamma",
    dtype_and_alpha_beta=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=1,
        max_num_dims=2,
        num_arrays=2,
        exclude_min=True,
    ),
    seed=helpers.ints(min_value=0, max_value=100),
    test_gradients=st.just(False),
)
def test_gamma(
    *, dtype_and_alpha_beta, seed, test_flags, backend_fw, fn_name, on_device
):
    dtype, alpha_beta = dtype_and_alpha_beta
    if "float16" in dtype:
        return
    ret, ret_gt = helpers.test_function(
        input_dtypes=dtype,
        test_flags=test_flags,
        test_values=False,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        alpha=alpha_beta[0],
        beta=alpha_beta[1],
        shape=None,
        dtype=dtype[0],
        seed=seed,
    )
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(
        ret=ret_gt, backend=test_flags.ground_truth_backend
    )
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        for u, v in zip(ret, ret_gt):
            assert ivy_backend.all(u >= 0)
            assert ivy_backend.all(v >= 0)


# poisson
# TODO: Enable gradient tests (test_gradients) once random generation
#   is unified
@handle_test(
    fn_tree="functional.ivy.experimental.poisson",
    dtype_and_lam=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        min_value=-2,
        max_value=5,
        min_num_dims=0,
    ),
    dtype=helpers.get_dtypes("float", full=False),
    seed=helpers.ints(min_value=0, max_value=100),
    fill_value=helpers.floats(min_value=0, max_value=1),
    test_gradients=st.just(False),
)
def test_poisson(
    *,
    dtype_and_lam,
    dtype,
    seed,
    fill_value,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    lam_dtype, lam = dtype_and_lam

    def call():
        return helpers.test_function(
            input_dtypes=lam_dtype,
            test_flags=test_flags,
            on_device=on_device,
            backend_to_test=backend_fw,
            fn_name=fn_name,
            test_values=False,
            lam=lam[0],
            shape=None,
            dtype=dtype[0],
            seed=seed,
            fill_value=fill_value,
        )

    ret, ret_gt = call()
    if seed:
        ret1, ret_gt1 = call()
        with BackendHandler.update_backend(backend_fw) as ivy_backend:
            assert ivy_backend.any(ret == ret1)
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(
        ret=ret_gt, backend=test_flags.ground_truth_backend
    )
    for u, v in zip(ret, ret_gt):
        assert u.dtype == v.dtype
        assert u.shape == v.shape
