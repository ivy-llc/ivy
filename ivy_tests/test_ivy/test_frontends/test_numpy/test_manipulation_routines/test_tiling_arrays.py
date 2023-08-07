# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# tile
@handle_frontend_test(
    fn_tree="numpy.tile",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    dtype_and_repeats=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("signed_integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape").map(
            lambda rep: (len(rep),)
        ),
        min_value=0,
        max_value=10,
    ),
    test_with_out=st.just(False),
)
def test_numpy_tile(
    *,
    dtype_and_x,
    dtype_and_repeats,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    repeats_dtype, repeats = dtype_and_repeats
    helpers.test_frontend_function(
        input_dtypes=input_dtype + repeats_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        A=x[0],
        reps=repeats[0],
    )


# repeat
@handle_frontend_test(
    fn_tree="numpy.repeat",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    repeats=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=1),
    test_with_out=st.just(False),
)
def test_numpy_repeat(
    *,
    dtype_and_x,
    repeats,
    axis,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        repeats=repeats,
        axis=axis,
    )
