from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# random_sample
@handle_frontend_test(
    fn_tree="tensorflow.random.uniform",
    shape=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=1,
        max_value=5,
        min_num_dims=1,
        max_num_dims=1,
    ),
    minval=helpers.ints(min_value=0, max_value=3),
    maxval=helpers.ints(min_value=4, max_value=10),
    dtype=helpers.get_dtypes("float", full=False),
    seed=helpers.ints(min_value=0, max_value=10),
    test_with_out=st.just(False),
)
def test_tensorflow_uniform(
    shape,
    minval,
    maxval,
    dtype,
    seed,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, shape = shape
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape[0],
        minval=minval,
        maxval=maxval,
        dtype=dtype[0],
        seed=seed,
    )


# random_normal
@handle_frontend_test(
    fn_tree="tensorflow.random.normal",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    mean=st.floats(allow_nan=False, allow_infinity=False, width=32),
    stddev=st.floats(allow_nan=False, allow_infinity=False, width=32, min_value=0),
    dtype=helpers.get_dtypes("float", full=False),
    seed=helpers.ints(min_value=0, max_value=10),
    test_with_out=st.just(False),
)
def test_tensorflow_normal(
    frontend,
    fn_tree,
    on_device,
    shape,
    mean,
    stddev,
    dtype,
    seed,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        mean=mean,
        stddev=stddev,
        dtype=dtype[0],
        seed=seed,
    )


# random_shuffle
@handle_frontend_test(
    fn_tree="tensorflow.random.shuffle",
    dtype_value=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    seed=helpers.ints(min_value=0, max_value=10),
    test_with_out=st.just(False),
)
def test_tensorflow_shuffle(
    frontend,
    fn_tree,
    on_device,
    dtype_value,
    seed,
    test_flags,
    backend_fw,
):
    input_dtypes, values = dtype_value
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        value=values[0],
        seed=seed,
    )


# random_stateless_uniform
@handle_frontend_test(
    fn_tree="tensorflow.random.stateless_uniform",
    shape=helpers.dtype_and_values(
        available_dtypes=("int64", "int32"),
        min_value=1,
        max_value=5,
        min_num_dims=1,
        max_num_dims=1,
        max_dim_size=9,
    ),
    seed=helpers.dtype_and_values(
        available_dtypes=("int64", "int32"), min_value=0, max_value=10, shape=[2]
    ),
    minmaxval=helpers.get_bounds(dtype="int32"),
    dtype=helpers.array_dtypes(
        available_dtypes=("int32", "int64", "float16", "float32", "float64"),
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_stateless_uniform(
    shape,
    seed,
    minmaxval,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    shape_input_dtypes, shape = shape
    seed_input_dtypes, seed = seed

    helpers.test_frontend_function(
        input_dtypes=shape_input_dtypes + seed_input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape[0],
        seed=seed[0],
        minval=int(minmaxval[0]),
        maxval=int(minmaxval[1]),
        dtype=dtype[0],
    )


# random poisson
@handle_frontend_test(
    fn_tree="tensorflow.random.poisson",
    shape=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=5,
    ),
    lam=st.one_of(
        helpers.floats(allow_inf=False, allow_nan=False, min_value=-2, max_value=5),
        helpers.lists(
            x=helpers.floats(
                allow_nan=False, allow_inf=False, min_value=-2, max_value=5
            ),
            min_size=1,
            max_size=10,
        ),
    ),
    dtype=helpers.get_dtypes("float", full=False),
    seed=helpers.ints(min_value=0, max_value=10),
    test_with_out=st.just(False),
)
def test_tensorflow_poisson(
    frontend,
    fn_tree,
    on_device,
    shape,
    lam,
    dtype,
    seed,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        lam=lam,
        dtype=dtype[0],
        seed=seed,
        test_values=False,
    )


# stateless_normal
@handle_frontend_test(
    fn_tree="tensorflow.random.stateless_normal",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    seed=helpers.dtype_and_values(
        available_dtypes=("int64", "int32"), min_value=0, max_value=10, shape=[2]
    ),
    mean=st.floats(allow_nan=False, allow_infinity=False, width=32),
    stddev=st.floats(allow_nan=False, allow_infinity=False, width=32, min_value=0),
    dtype=helpers.get_dtypes("float", full=False),
    test_with_out=st.just(False),
)
def test_tensorflow_stateless_normal(
    frontend,
    fn_tree,
    on_device,
    shape,
    seed,
    mean,
    stddev,
    dtype,
    test_flags,
    backend_fw,
):
    input_dtypes, seed = seed
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        seed=seed[0],
        mean=mean,
        stddev=stddev,
        dtype=dtype[0],
    )


# stateless_poisson
@st.composite
def _shape_lam_dtype(draw):
    dtype = draw(helpers.array_dtypes(available_dtypes=("float32", "float64")))
    common_shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=2,
            max_num_dims=3,
            min_dim_size=1,
            max_dim_size=5,
        )
    )
    _, lam = draw(
        helpers.dtype_and_values(
            available_dtypes=dtype, min_value=0, max_value=10, shape=(common_shape[-1],)
        )
    )
    return common_shape, lam, dtype


@handle_frontend_test(
    fn_tree="tensorflow.random.stateless_poisson",
    shape_lam_dtype=_shape_lam_dtype(),
    seed=helpers.dtype_and_values(
        available_dtypes=("int64", "int32"), min_value=0, max_value=10, shape=[2]
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_stateless_poisson(
    frontend,
    fn_tree,
    on_device,
    shape_lam_dtype,
    seed,
    test_flags,
    backend_fw,
):
    shape, lam, dtype = shape_lam_dtype
    input_dtypes, seed = seed
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        shape=shape,
        seed=seed[0],
        lam=lam[0],
        dtype=dtype[0],
    )


# random gamma
@handle_frontend_test(
    fn_tree="tensorflow.random.gamma",
    dtype=helpers.array_dtypes(
        available_dtypes=("float32", "float64"),
    ),
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=5,
    ),
    alpha=st.floats(
        allow_infinity=False, allow_nan=False, width=32, min_value=1, max_value=3
    ),
    beta=st.floats(
        allow_infinity=False, allow_nan=False, width=32, min_value=1, max_value=3
    ),
    seed=helpers.ints(min_value=0, max_value=10),
    test_with_out=st.just(False),
)
def test_tensorflow_gamma(
    frontend,
    fn_tree,
    on_device,
    shape,
    alpha,
    beta,
    dtype,
    seed,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        alpha=alpha,
        beta=beta,
        dtype=dtype[0],
        seed=seed,
        test_values=False,
    )

@handle_frontend_test(
    fn_tree="tensorflow.random.stateless_categorical",
    logits=st.floats(allow_nan=False, allow_infinity=False, min_value=-5, max_value=5),
    num_samples=helpers.ints(min_value=1, max_value=5),
    seed=helpers.dtype_and_values(
        available_dtypes=("int64", "int32"), min_value=0, max_value=10, shape=[2]
    ),
    test_with_out=st.just(False),
)
def test_tensorflow_stateless_categorical(
    frontend,
    fn_tree,
    on_device,
    logits,
    num_samples,
    seed,
    test_flags,
    backend_fw,
):
    input_dtypes, seed = seed
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        logits=logits,
        num_samples=num_samples,
        seed=seed[0] + seed[1],
    )
