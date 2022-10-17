# global
import torch
from hypothesis import assume, given, strategies as st

# local
from ivy.functional.frontends.torch.Tensor import Tensor
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


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
            "alpha": alpha,
        },
        frontend="torch",
        class_name="tensor",
        method_name="add",
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
        method_name="sin",
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
    assume("bfloat16" not in input_dtype)
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
        method_name="cos",
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
    assume("bfloat16" not in input_dtype)
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
        method_name="tan",
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


# __add__
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        shared_dtype=True,
        num_arrays=2
    )
)
def test_torch_special_add(
    dtype_x,
):
    input_dtype, x = dtype_x
    ret = Tensor(x[0]) + Tensor(x[1])
    ret_gt = torch.tensor(x[0], dtype=input_dtype[0]) + torch.tensor(
        x[1], dtype=input_dtype[1]
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=ret_gt)
    for (u, v) in zip(ret, ret_gt):
        helpers.value_test(
            ret_np_flat=u,
            ret_np_from_gt_flat=v,
            ground_truth_backend="torch",
        )
