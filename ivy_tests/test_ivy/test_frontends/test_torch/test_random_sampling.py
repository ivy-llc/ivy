# global
from hypothesis import strategies as st
import importlib

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@st.composite
def _pop_size_num_samples_replace_n_probs(draw):
    prob_dtype = draw(helpers.get_dtypes("float", full=False))
    batch_size = draw(helpers.ints(min_value=1, max_value=5))
    replace = draw(st.booleans())
    num_samples = draw(helpers.ints(min_value=1, max_value=20))
    probs = draw(
        helpers.array_values(
            dtype=prob_dtype[0],
            shape=[batch_size, num_samples],
            min_value=1.0013580322265625e-05,
            max_value=1.0,
            exclude_min=True,
        )
    )
    return prob_dtype, batch_size, num_samples, replace, probs


# multinomial
@handle_frontend_test(
    fn_tree="torch.multinomial",
    everything=_pop_size_num_samples_replace_n_probs(),
)
def test_torch_multinomial(
    *,
    everything,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    prob_dtype, batch_size, num_samples, replace, probs = everything

    def call():
        return helpers.test_frontend_function(
            input_dtypes=prob_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            input=probs,
            num_samples=num_samples,
            replacement=replace,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.manual_seed",
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_torch_manual_seed(
    *,
    seed,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    # just test calling the function
    frontend_fw = importlib.import_module(fn_tree[25 : fn_tree.rfind(".")])
    split_index = fn_tree.rfind(".")
    _, fn_name = fn_tree[:split_index], fn_tree[split_index + 1 :]
    frontend_fw.__dict__[fn_name](seed)


@handle_frontend_test(
    fn_tree="torch.poisson",
    dtype_and_lam=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        min_value=0,
        max_value=100,
        min_num_dims=0,
        max_num_dims=10,
        min_dim_size=1,
    ),
)
def test_torch_poisson(
    *,
    dtype_and_lam,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    lam_dtype, lam = dtype_and_lam

    def call():
        return helpers.test_frontend_function(
            input_dtypes=lam_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            input=lam[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


# randint
@handle_frontend_test(
    fn_tree="torch.randint",
    low=helpers.ints(min_value=0, max_value=10),
    high=helpers.ints(min_value=11, max_value=20),
    size=helpers.get_shape(),
    dtype=helpers.get_dtypes("integer"),
)
def test_torch_randint(
    *,
    low,
    high,
    size,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    def call():
        helpers.test_frontend_function(
            input_dtypes=dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_values=False,
            fn_tree=fn_tree,
            test_flags=test_flags,
            low=low,
            high=high,
            size=size,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.rand",
    dtype=helpers.get_dtypes("float", full=False),
    size=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_torch_rand(*, dtype, size, frontend, fn_tree, test_flags, backend_fw):
    size = {f"size{i}": size[i] for i in range(len(size))}
    test_flags.num_positional_args = len(size)

    def call():
        return helpers.test_frontend_function(
            input_dtypes=dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_values=False,
            fn_tree=fn_tree,
            test_flags=test_flags,
            **size,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.normal",
    dtype_and_mean=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1000,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
    ),
    dtype_and_std=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=1000,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=2,
    ),
)
def test_torch_normal(
    *,
    dtype_and_mean,
    dtype_and_std,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    mean_dtype, mean = dtype_and_mean
    _, std = dtype_and_std

    def call():
        return helpers.test_frontend_function(
            input_dtypes=mean_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            mean=mean[0],
            std=std[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.rand_like",
    dtype=helpers.get_dtypes("float", full=False),
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_torch_rand_like(
    dtype_and_x, dtype, *, frontend, fn_tree, test_flags, backend_fw
):
    input_dtype, input = dtype_and_x

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_values=False,
            fn_tree=fn_tree,
            test_flags=test_flags,
            input=input[0],
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.randn",
    dtype=helpers.get_dtypes("float", full=False),
    size=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_torch_randn(*, dtype, size, frontend, fn_tree, test_flags, backend_fw):
    size = {f"size{i}": size[i] for i in range(len(size))}
    test_flags.num_positional_args = len(size)

    def call():
        return helpers.test_frontend_function(
            input_dtypes=dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_values=False,
            fn_tree=fn_tree,
            test_flags=test_flags,
            **size,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.randn_like",
    dtype=helpers.get_dtypes("float", full=False),
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=10,
    ),
)
def test_torch_randn_like(
    dtype_and_x, dtype, *, frontend, fn_tree, test_flags, backend_fw
):
    input_dtype, input = dtype_and_x

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_values=False,
            fn_tree=fn_tree,
            test_flags=test_flags,
            input=input[0],
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.bernoulli",
    dtype_and_probs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False),
        min_value=0,
        max_value=1,
        min_num_dims=0,
    ),
)
def test_torch_bernoulli(
    dtype_and_probs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, probs = dtype_and_probs

    def call():
        return helpers.test_frontend_function(
            input_dtypes=dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            input=probs[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


# randperm
@handle_frontend_test(
    fn_tree="torch.randperm",
    n=st.integers(min_value=0, max_value=10),
    dtype=helpers.get_dtypes("integer", full=False),
)
def test_torch_randperm(
    *,
    n,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    def call():
        return helpers.test_frontend_function(
            input_dtypes=dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            test_values=False,
            n=n,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


@handle_frontend_test(
    fn_tree="torch.randint_like",
    dtype=helpers.get_dtypes("signed_integer", full=False),
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=1,
        max_dim_size=10,
    ),
    low=helpers.ints(min_value=0, max_value=10),
    high=helpers.ints(min_value=11, max_value=20),
)
def test_torch_randint_like(
    dtype_and_x, low, high, *, dtype, frontend, fn_tree, test_flags, backend_fw
):
    input_dtype, input = dtype_and_x

    def call():
        return helpers.test_frontend_function(
            input_dtypes=input_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_values=False,
            fn_tree=fn_tree,
            test_flags=test_flags,
            input=input[0],
            low=low,
            high=high,
            dtype=dtype[0],
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


# set_rng_state
@handle_frontend_test(
    fn_tree="torch.set_rng_state",
    new_state=helpers.dtype_and_values(
        available_dtypes=("int64", "int32"),
        min_value=0,
        max_value=10,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        max_dim_size=1,
    ),
)
def test_torch_set_rng_state(
    *,
    new_state,
    frontend,
    test_flags,
    fn_tree,
):
    dtype, new_state = new_state
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_values=False,
        fn_tree=fn_tree,
        test_flags=test_flags,
        state=new_state[0],
    )
