# global
from hypothesis import strategies as st
from hypothesis import composite
from hypothesis.extra.numpy import mutually_broadcastable_shapes

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers


@handle_frontend_test(
    fn_tree="numpy.array",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
)
def test_numpy_array(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        object=a,
        dtype=dtype[0],
    )


# asarray
@handle_frontend_test(
    fn_tree="numpy.asarray",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
)
def test_numpy_asarray(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a,
        dtype=dtype[0],
    )


# choose
@composite
def numpy_choose_dtype_and_values(draw):
    """ Creates arguments for numpy.choose using dtype_and_values. """
    # make sure to set: ivy.set_backend("numpy")
    # generate a mode, and adjust the index array accordingly
    mode = draw(st.sampled_from(['raise', 'wrap', 'clip']))
    #n_choices = ivy.randint(1, 10).item() # number of choices
    n_choices = draw(st.integers(min_value = 1, max_value = 10))
    min_value = None
    max_value = None
    if mode == 'raise':
        min_value = 0
        max_value = n_choices - 1
    # create the index array 'a'
    la_dtype, la = draw(helpers.dtype_and_values(
        available_dtypes = helpers.get_dtypes("integer"),
        min_num_dims = 1,
        min_value = min_value,
        max_value = max_value
    ))
    a_dtype = la_dtype[0]
    a = la[0]
    # create the choices array (same dtype as 'a')
    choices = []
    c_dtypes = []
    c_shapes = draw(mutually_broadcastable_shapes(base_shape=a.shape,
                    num_shapes=n_choices)).input_shapes
    for c_idx in range(n_choices):
        c_shape = c_shapes[c_idx]
        lc_dtype, lc = draw(helpers.dtype_and_values(
            available_dtypes = helpers.get_dtypes("integer"),
            shape = c_shape))
        choices.append(lc[0])
        c_dtypes.append(lc_dtype[0])
    return la_dtype + c_dtypes, la + choices, mode

@handle_frontend_test(
    fn_tree = "numpy.choose",
    dtype_x_mode = numpy_choose_dtype_and_values(),
    #num_positional_args=helpers.num_positional_args(fn_name="choose"),
)
def test_numpy_choose(
    fn_tree,
    dtype_x_mode,
    #num_positional_args,
    test_flags
):
    input_dtypes, x, mode = dtype_x_mode
    
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        a=x[0], 
        choices=x[1:],
        mode=mode,
        out=None,
        frontend="numpy",
        test_flags=test_flags,
        fn_tree=fn_tree
    )


# copy
@handle_frontend_test(
    fn_tree="numpy.copy",
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    test_with_out=st.just(False),
)
def test_numpy_copy(
    dtype_and_a,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        a=a[0],
    )
