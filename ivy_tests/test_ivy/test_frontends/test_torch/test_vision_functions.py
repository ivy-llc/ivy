from hypothesis import assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# pixel_shuffle
@handle_frontend_test(
    fn_tree="torch.nn.functional.pixel_shuffle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
    ),
    factor=helpers.ints(min_value=1),
)
def test_torch_pixel_shuffle(
    *,
    dtype_and_x,
    factor,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume(ivy.shape(x[0])[1] % (factor**2) == 0)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        upscale_factor=factor,
    )


@handle_frontend_test(
    fn_tree="torch.nn.functional.pixel_unshuffle",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        min_num_dims=4,
        max_num_dims=4,
        min_dim_size=1,
    ),
    factor=helpers.ints(min_value=1),
)
def test_torch_pixel_unshuffle(
    *,
    dtype_and_x,
    factor,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume((ivy.shape(x[0])[2] % factor == 0) & (ivy.shape(x[0])[3] % factor == 0))
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
        downscale_factor=factor,
    )


@st.composite
def _pad_generator(draw, shape, mode):
    pad = ()
    m = max(int((len(shape) + 1) / 2), 1)
    for i in range(m):
        if mode != "constant":
            if i < 2:
                max_pad_value = 0
        else:
            max_pad_value = shape[i] - 1
        pad = pad + draw(
            st.tuples(
                st.integers(min_value=0, max_value=max(0, max_pad_value)),
                st.integers(min_value=0, max_value=max(0, max_pad_value)),
            )
        )
    return pad


@st.composite
def _pad_helper(draw):
    mode = draw(
        st.sampled_from(
            [
                "constant",
                "reflect",
                "replicate",
                "circular",
            ]
        )
    )
    min_v = 1
    max_v = 5
    if mode != "constant":
        min_v = 3
        if mode == "reflect":
            max_v = 4
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            ret_shape=True,
            min_num_dims=min_v,
            max_num_dims=max_v,
            min_dim_size=2,
            min_value=-1e05,
            max_value=1e05,
        )
    )
    padding = draw(_pad_generator(shape, mode))
    if mode == "constant":
        value = draw(helpers.ints(min_value=0, max_value=4))
    else:
        value = 0.0
    return dtype, input[0], padding, value, mode


@handle_frontend_test(
    fn_tree="torch.nn.functional.pad",
    dtype_and_input_and_other=_pad_helper(),
)
def test_torch_pad(
    *,
    dtype_and_input_and_other,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, padding, value, mode = dtype_and_input_and_other
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        pad=padding,
        mode=mode,
        value=value,
    )


@st.composite
def _upsample_bilinear_helper(draw):
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=["float32", "float64"],
            ret_shape=True,
            min_num_dims=4,
            max_num_dims=4,
            min_dim_size=1,
            max_dim_size=1000,
            min_value=-1e05,
            max_value=1e05,
        )
    )
    size = None
    scale_factor = None
    is_size_used = draw(st.booleans())
    is_int = draw(st.booleans())
    if is_size_used and is_int:
        size = draw(helpers.ints(min_value=shape[2]))
    elif is_size_used and not is_int:
        size = (
            draw(helpers.ints(min_value=shape[2])),
            draw(helpers.ints(min_value=shape[3])),
        )
    elif not is_size_used and is_int:
        scale_factor = draw(helpers.ints(min_value=1))
    elif not is_size_used and not is_int:
        scale_factor = (
            draw(helpers.ints(min_value=shape[2])),
            draw(helpers.ints(min_value=shape[3])),
        )

    return dtype, input[0], size, scale_factor


@handle_frontend_test(
    fn_tree="torch.nn.functional.upsample_bilinear",
    dtype_and_input_and_other=_upsample_bilinear_helper(),
)
def test_torch_upsample_bilinear(
    *,
    dtype_and_input_and_other,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input, size, scale_factor = dtype_and_input_and_other
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input,
        size=size,
        scale_factor=scale_factor,
    )
