# global
import ivy
import torch
from hypothesis import assume, given, strategies as st
import hypothesis.extra.numpy as hnp

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# Helper functions
@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_length(
                x=st.sampled_from(draw(helpers.get_dtypes("numeric"))), length=1
            ),
            key="dtype",
        )
    )


@st.composite
def _requires_grad(draw):
    dtype = draw(_dtypes())[0]
    if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
        return draw(st.just(False))
    return draw(st.booleans())


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
)
def test_torch_instance_add(
    dtype_and_x,
    alpha,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
            "alpha": alpha,
        },
        frontend="torch",
        class_name="tensor",
        method_name="add",
    )


# new_ones
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
)
def test_torch_instance_new_ones(
    dtype_and_x,
    size,
    dtypes,
    requires_grad,
    device,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=dtypes,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "size": size,
            "dtype": dtypes[0],
            "requires_grad": requires_grad,
            "device": device,
        },
        frontend="torch",
        class_name="tensor",
        method_name="new_ones",
    )


@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
)
def test_torch_instance_reshape(
    dtype_x,
    shape,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "shape": shape,
        },
        frontend="torch",
        class_name="tensor",
        method_name="reshape",
    )


# sin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="sin",
    )


# arcsin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arcsin(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="arcsin",
    )


# atan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_atan(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="atan",
    )


# sin_
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin_(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="sin_",
    )


# cos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cos(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="cos",
    )


# cos_
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cos_(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": list(x[0]) if type(x[0]) == int else x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="cos_",
    )


# sinh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sinh(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="sinh",
    )


# sinh_
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sinh_(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="sinh_",
    )


# view
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
)
def test_torch_instance_view(
    dtype_x,
    shape,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "shape": shape,
        },
        frontend="torch",
        class_name="tensor",
        method_name="view",
    )


@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
)
def test_torch_instance_float(
    dtype_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "memory_format": torch.preserve_format,
        },
        frontend="torch",
        class_name="tensor",
        method_name="float",
    )


# asinh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asinh(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="asinh",
        rtol_=1e-2,
        atol_=1e-2,
    )


# asinh_
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asinh_(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="asinh_",
        rtol_=1e-2,
        atol_=1e-2,
    )


# tan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tan(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="tan",
    )


# asin
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asin(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="asin",
    )


# amax
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
)
def test_torch_instance_amax(
    dtype_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="amax",
    )


# abs
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    )
)
def test_torch_instance_abs(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="abs",
    )


# abs_
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    )
)
def test_torch_instance_abs_(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="abs_",
    )


# amin
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
)
def test_torch_instance_amin(
    dtype_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="amin",
    )


# contiguous
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_contiguous(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "memory_format": torch.contiguous_format,
        },
        frontend="torch",
        class_name="tensor",
        method_name="contiguous",
    )


# log
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="log",
    )


# __add__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_special_add(
    dtype_and_x,
    alpha,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
            "alpha": alpha,
        },
        frontend="torch",
        class_name="tensor",
        method_name="__add__",
    )


# __radd__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_special_radd(
    dtype_and_x,
    alpha,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
            "alpha": alpha,
        },
        frontend="torch",
        class_name="tensor",
        method_name="__radd__",
    )


# __sub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_special_sub(
    dtype_and_x,
    alpha,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
            "alpha": alpha,
        },
        frontend="torch",
        class_name="tensor",
        method_name="__sub__",
    )


# __mul__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_mul(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
        },
        frontend="torch",
        class_name="tensor",
        method_name="__mul__",
    )


# __rmul__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_rmul(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
        },
        frontend="torch",
        class_name="tensor",
        method_name="__rmul__",
    )


# __truediv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    rounding_mode=st.sampled_from([None, "trunc", "floor"]),
)
def test_torch_special_truediv(
    dtype_and_x,
    rounding_mode,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
            "rounding_mode": rounding_mode,
        },
        frontend="torch",
        class_name="tensor",
        method_name="__truediv__",
    )


# to_with_device
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
    copy=st.booleans(),
)
def test_torch_instance_to_with_device(
    dtype_x,
    copy,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "device": ivy.Device("cpu"),
            "dtype": ivy.as_ivy_dtype(input_dtype[0]),
            "non_blocking": False,
            "copy": copy,
            "memory_format": torch.preserve_format,
        },
        frontend="torch",
        class_name="tensor",
        method_name="to",
    )


# to_with_dtype
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
    copy=st.booleans(),
)
def test_torch_instance_to_with_dtype(
    dtype_x,
    copy,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=3,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "dtype": ivy.as_ivy_dtype(input_dtype[0]),
            "non_blocking": False,
            "copy": copy,
            "memory_format": torch.preserve_format,
        },
        frontend="torch",
        class_name="tensor",
        method_name="to",
    )


# arctan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arctan(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="arctan",
    )


# acos
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_acos(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=["float64"] + input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=["float64"] + input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="acos",
    )


# new_tensor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    )
)
def test_torch_instance_new_tensor(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=[input_dtype[0]],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "data": x[1],
            "dtype": input_dtype[1],
        },
        frontend="torch",
        class_name="tensor",
        method_name="new_tensor",
    )


@st.composite
def _array_and_index(
    draw,
    *,
    available_dtypes=helpers.get_dtypes("numeric"),
    min_num_dims=1,
    max_num_dims=3,
    min_dim_size=1,
    max_dim_size=10,
    shape=None,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)

    assert available_dtypes is not None, "Unspecified dtype or available_dtypes."
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    dtype.append("int32")

    if shape is not None:
        if not isinstance(shape, (tuple, list)):
            shape = draw(shape)
    else:
        shape = draw(
            st.shared(
                helpers.get_shape(
                    min_num_dims=min_num_dims,
                    max_num_dims=max_num_dims,
                    min_dim_size=min_dim_size,
                    max_dim_size=max_dim_size,
                ),
                key="shape",
            )
        )

    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=shape,
        )
    )

    index = tuple([draw(helpers.ints(min_value=0, max_value=_ - 1)) for _ in shape])
    index = index if len(index) != 0 else index[0]
    return dtype, [array, index]


# __getitem__
@handle_cmd_line_args
@given(
    dtype_and_x=_array_and_index(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_torch_instance_getitem(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    data = x[0]
    index = x[1]
    helpers.test_frontend_method(
        input_dtypes_init=[input_dtype[0]],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={"data": data},
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={"query": index},
        frontend="torch",
        class_name="tensor",
        method_name="__getitem__",
    )


# view_as
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        num_arrays=2,
    ),
)
def test_torch_instance_view_as(
    dtype_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
        },
        frontend="torch",
        class_name="tensor",
        method_name="view_as",
    )


# unsqueeze
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_unsqueeze(
    dtype_value,
    dim,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "dim": dim,
        },
        frontend="torch",
        class_name="tensor",
        method_name="unsqueeze",
    )


# unsqueeze_
@handle_cmd_line_args
@given(
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_unsqueeze_(
    dtype_value,
    dim,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "dim": dim,
        },
        frontend="torch",
        class_name="tensor",
        method_name="unsqueeze_",
    )


# detach
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
    )
)
def test_torch_instance_detach(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="detach",
    )


# dim
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    )
)
def test_torch_instance_dim(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="dim",
    )


# ndimension
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    )
)
def test_torch_instance_ndimension(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="ndimension",
    )


@st.composite
def _fill_value_and_size(
    draw,
    *,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    if isinstance(min_dim_size, st._internal.SearchStrategy):
        min_dim_size = draw(min_dim_size)
    if isinstance(max_dim_size, st._internal.SearchStrategy):
        max_dim_size = draw(max_dim_size)

    available_dtypes = draw(helpers.get_dtypes("numeric"))
    dtype = draw(
        helpers.array_dtypes(
            num_arrays=1,
            available_dtypes=available_dtypes,
        )
    )
    array = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=(1,),
        )
    )
    dtype.append("int32")
    size = draw(
        st.shared(
            helpers.get_shape(
                min_num_dims=min_num_dims,
                max_num_dims=max_num_dims,
                min_dim_size=min_dim_size,
                max_dim_size=max_dim_size,
            ),
            key="shape",
        )
    )
    fill_value = draw(helpers.ints())

    return dtype, [array, size, fill_value]


# new_full
@handle_cmd_line_args
@given(dtype_and_x=_fill_value_and_size(max_num_dims=3))
def test_torch_instance_new_full(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=[input_dtype[0]],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=2,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "size": x[1],
            "fill_value": x[2],
        },
        frontend="torch",
        class_name="tensor",
        method_name="new_full",
    )


# new_empty (not actually intuitive for testing)
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    size=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=3,
    ),
)
def test_torch_instance_new_empty(dtype_and_x, size, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=[input_dtype[0]],
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x,
        },
        input_dtypes_method=[ivy.int32],
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "size": size,
        },
        frontend="torch",
        class_name="tensor",
        method_name="new_empty",
    )


@st.composite
def _expand_helper(draw):
    shape, _ = draw(hnp.mutually_broadcastable_shapes(num_shapes=2, min_dims=2))
    shape1, shape2 = shape
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid", full=True), shape=shape1
        )
    )
    dtype, x = dtype_x
    return dtype, x, shape1


@handle_cmd_line_args
@given(
    dtype_x_shape=_expand_helper(),
)
def test_torch_instance_expand(
    dtype_x_shape,
    as_variable,
    native_array,
):

    input_dtype, x, shape = dtype_x_shape
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=len(shape),
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={str(i): s for i, s in enumerate(shape)},
        frontend="torch",
        class_name="tensor",
        method_name="expand",
    )


@st.composite
def _unfold_args(draw):
    values_dtype, values, axis, shape = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("float"),
            force_int_axis=True,
            shape=draw(
                helpers.get_shape(
                    allow_none=False,
                    min_num_dims=1,
                    min_dim_size=1,
                )
            ),
            ret_shape=True,
        )
    )
    size = draw(
        st.integers(
            min_value=1,
            max_value=len(shape[axis] - 1),
        )
    )
    step = draw(
        st.integers(
            min_value=1,
            max_value=size,
        )
    )
    return values_dtype, values, axis, size, step


# unfold
@handle_cmd_line_args
@given(
    dtype_values_args=_unfold_args(),
)
def test_torch_instance_unfold(
    dtype_values_args, size, step, as_variable, native_array
):
    input_dtype, x, axis, size, step = dtype_values_args
    print(axis, size, step)
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=3,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "dimension": axis,
            "size": size,
            "step": step,
        },
        frontend="torch",
        class_name="tensor",
        method_name="unfold",
    )


# __mod__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_special_mod(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=1,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
        },
        frontend="torch",
        class_name="tensor",
        method_name="__mod__",
    )


# long
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
    ),
)
def test_torch_instance_long(
    dtype_and_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x,
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="long",
    )


# max
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
)
def test_torch_instance_max(
    dtype_x,
    as_variable,
    native_array,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=input_dtype,
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="max",
    )


# device
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_device(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=1,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        frontend="torch",
        class_name="tensor",
        method_name="device",
    )
