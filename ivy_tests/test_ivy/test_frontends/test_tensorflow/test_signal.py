# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# kaiser_window
@handle_frontend_test(
    fn_tree="tensorflow.signal.kaiser_window",
    window_length=helpers.ints(min_value=1, max_value=100),
    beta=helpers.floats(min_value=0.0, max_value=80),
    test_with_out=st.just(False),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_tensorflow_kaiser_window(
    *,
    window_length,
    beta,
    frontend,
    test_flags,
    fn_tree,
    on_device,
    dtype,
):
    helpers.test_frontend_function(
        input_dtypes=[window_length, beta],
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        window_length=window_length,
        beta=beta,
        dtype=dtype[0],
        rtol=1e-01,
    )


@st.composite
def valid_idct(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            max_value=65280,
            min_value=-65280,
            min_num_dims=1,
            min_dim_size=2,
            shared_dtype=True,
        )
    )
    n = None
    axis = -1
    norm = draw(st.sampled_from([None, "ortho"]))
    type = draw(helpers.ints(min_value=1, max_value=4))
    if norm == "ortho" and type == 1:
        norm = None
    return dtype, x, type, n, axis, norm


# idct
@handle_frontend_test(
    fn_tree="tensorflow.signal.idct",
    dtype_x_and_args=valid_idct(),
    test_with_out=st.just(False),
)
def test_tensorflow_idct(
    *,
    dtype_x_and_args,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtype, x, type, n, axis, norm = dtype_x_and_args
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        type=type,
        n=n,
        axis=axis,
        norm=norm,
        atol=1e-01,
    )
