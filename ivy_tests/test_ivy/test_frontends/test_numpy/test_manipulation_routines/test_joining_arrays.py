# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    input_dtypes, x, casting, dtype = draw(
        np_frontend_helpers.dtypes_values_casting_dtype(
            arr_func=[
                lambda: helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("numeric"),
                    shape=shape,
                    num_arrays=num_arrays,
                    shared_dtype=True,
                )
            ],
        ),
    )
    axis = draw(helpers.get_axis(shape=shape, force_int=True))
    return x, input_dtypes, axis, casting, dtype


# concat
@handle_frontend_test(
    fn_tree="numpy.concatenate",
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
)
def test_numpy_concatenate(
    xs_n_input_dtypes_n_unique_idx,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    xs, input_dtypes, unique_idx, casting, dtype = xs_n_input_dtypes_n_unique_idx
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        atol=1e-01,
        arrays=xs,
        axis=unique_idx,
        casting=casting,
        dtype=dtype,
        out=None,
    )


# stack
@handle_frontend_test(
    fn_tree="numpy.stack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
)
def test_numpy_stack(
    dtype_and_x,
    factor,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    xs = [x[0]]
    for i in range(factor):
        xs += [x[0]]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        arrays=xs,
        axis=0,
    )


# vstack
@handle_frontend_test(
    fn_tree="numpy.vstack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
)
def test_numpy_vstack(
    dtype_and_x,
    factor,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    xs = [x[0]]
    for i in range(factor):
        xs += [x[0]]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tup=xs,
    )


# hstack
@handle_frontend_test(
    fn_tree="numpy.hstack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
)
def test_numpy_hstack(
    dtype_and_x,
    factor,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, x = dtype_and_x
    xs = [
        x[0],
    ]
    for i in range(factor):
        xs += [
            x[0],
        ]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        tup=xs,
    )
