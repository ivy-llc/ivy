# global
import ivy
import torch
from hypothesis import assume, strategies as st
import hypothesis.extra.numpy as hnp

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_frontends.test_torch.test_blas_and_lapack_ops import (
    _get_dtype_and_3dbatch_matrices,
    _get_dtype_input_and_matrices,
)
from ivy.functional.frontends.torch import Tensor
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy_tests.test_ivy.helpers import handle_frontend_method, handle_frontend_test


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


@handle_frontend_test(
    fn_tree="torch.argmax",  # dummy fn_tree
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_tensor_property_ivy_array(
    dtype_x,
):
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ret = helpers.flatten_and_to_np(ret=x.ivy_array.data)
    ret_gt = helpers.flatten_and_to_np(ret=data[0])
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        ground_truth_backend="torch",
    )


# add
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.add",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    method_name,
    init_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# new_ones
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.new_ones",
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
    on_device,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "size": size,
            "dtype": dtypes[0],
            "requires_grad": requires_grad,
            "device": on_device,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.reshape",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "shape": shape,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# sin
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.sin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arcsin
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arcsin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arcsin(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# sum
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.sum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sum(
    dtype_and_x,
    as_variable,
    native_array,
    frontend,
    init_name,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=["float64"] + input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=1,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=["float64"] + input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=0,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# atan
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.atan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_atan(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# sin_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.sin_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sin_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# cos
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.cos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cos(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# cos_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.cos_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cos_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    assume("bfloat16" not in input_dtype)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": list(x[0]) if type(x[0]) == int else x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# sinh
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sinh(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# sinh_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.sinh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_sinh_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# cosh
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cosh(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# cosh_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.cosh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_cosh_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# view
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.view",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "shape": shape,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.float",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
)
def test_torch_instance_float(
    dtype_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "memory_format": torch.preserve_format,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# asinh
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.asinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asinh(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    frontend,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
    )


# asinh_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.asinh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asinh_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
        rtol_=1e-2,
        atol_=1e-2,
    )


# tan
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.tan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tan(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# tanh
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tanh(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# tanh_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.tanh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tanh_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# asin
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.asin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_asin(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# amax
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.amax",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
)
def test_torch_instance_amax(
    dtype_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# abs
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_abs(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# abs_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.abs_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_abs_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# amin
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.amin",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
)
def test_torch_instance_amin(
    dtype_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# contiguous
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.contiguous",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_contiguous(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "memory_format": torch.contiguous_format,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# log
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_log(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __add__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__add__",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    frontend,
    method_name,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __long__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__long__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_special_long(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __radd__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__radd__",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __sub__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__sub__",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __mul__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__mul__",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __rmul__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__rmul__",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __truediv__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__truediv__",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
            "rounding_mode": rounding_mode,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# _to_with_device
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.to",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True),
    ),
    copy=st.booleans(),
)
def test_torch_instance_to_with_device(
    dtype_x,
    copy,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "device": ivy.Device("cpu"),
            "dtype": ivy.as_ivy_dtype(input_dtype[0]),
            "non_blocking": False,
            "copy": copy,
            "memory_format": torch.preserve_format,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


@st.composite
def _to_helper(draw):
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
        )
    )
    input_dtype, x = dtype_x
    arg = draw(st.sampled_from(["tensor", "dtype", "device"]))
    if arg == "tensor":
        method_num_positional_args = 1
        method_all_as_kwargs_np = {"other": x[1]}
    elif arg == "dtype":
        method_num_positional_args = 1
        dtype = draw(helpers.get_dtypes("valid", full=False))
        method_all_as_kwargs_np = {"dtype": dtype}
    else:
        method_num_positional_args = 0
        device = draw(st.sampled_from([torch.device("cuda"), torch.device("cpu")]))
        dtype = draw(helpers.get_dtypes("valid", full=False, none=True))
        method_all_as_kwargs_np = {"dtype": dtype, "device": device}
    return input_dtype, x, method_num_positional_args, method_all_as_kwargs_np


# to
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.to",
    args_kwargs=_to_helper(),
)
def test_torch_instance_to(
    args_kwargs,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x, method_num_positional_args, method_all_as_kwargs_np = args_kwargs
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np=method_all_as_kwargs_np,
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arctan
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arctan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arctan(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arctan_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arctan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_arctan_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# acos
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.acos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_acos(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# new_tensor
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.new_tensor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_new_tensor(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "data": x[1],
            "dtype": input_dtype[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
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
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__getitem__",
    dtype_and_x=_array_and_index(available_dtypes=helpers.get_dtypes("numeric")),
)
def test_torch_instance_getitem(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    data = x[0]
    index = x[1]
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={"data": data},
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={"query": index},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# view_as
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.view_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        num_arrays=2,
    ),
)
def test_torch_instance_view_as(
    dtype_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# unsqueeze
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.unsqueeze",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# unsqueeze_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.unsqueeze_",
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
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# detach
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.detach",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
    ),
)
def test_torch_instance_detach(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# dim
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.dim",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_dim(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# ndimension
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.ndimension",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_ndimension(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
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
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.new_full",
    dtype_and_x=_fill_value_and_size(max_num_dims=3),
)
def test_torch_instance_new_full(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "size": x[1],
            "fill_value": x[2],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# new_empty (not actually intuitive for testing)
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.new_empty",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    size=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=3,
    ),
)
def test_torch_instance_new_empty(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    size,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=[ivy.int32],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "size": size,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
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


@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.expand",
    dtype_x_shape=_expand_helper(),
)
def test_torch_instance_expand(
    dtype_x_shape,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):

    input_dtype, x, shape = dtype_x_shape
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={str(i): s for i, s in enumerate(shape)},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
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
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.unfold",
    dtype_values_args=_unfold_args(),
)
def test_torch_instance_unfold(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_values_args,
    size,
    step,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x, axis, size, step = dtype_values_args
    print(axis, size, step)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "dimension": axis,
            "size": size,
            "step": step,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# __mod__
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_special_mod(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# long
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.long",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=1,
    ),
)
def test_torch_instance_long(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# max
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.max",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
)
def test_torch_instance_max(
    dtype_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# device
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.device",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_instance_device(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# is_cuda
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.is_cuda",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtypes=_dtypes(),
    requires_grad=_requires_grad(),
    device=st.booleans(),
)
def test_torch_instance_is_cuda(
    dtype_and_x,
    size,
    dtypes,
    requires_grad,
    device,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    frontend,
    init_name,
    method_name,
):
    input_dtype, x = dtype_and_x
    device = "cpu" if device is False else "gpu:0"
    x = Tensor(data=x[0]).new_ones(
        size=size, dtype=dtypes[0], device=device, requires_grad=requires_grad
    )

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# bitwise_and
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.bitwise_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_instance_bitwise_and(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# add_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.add_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_add_(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arccos_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arccos_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arccos_(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arccos
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arccos",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arccos(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# acos_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.acos_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_acos_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# asin_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.asin_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_asin_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arcsin_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arcsin_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arcsin_(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# atan_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.atan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_atan_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# tan_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.tan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_instance_tan_(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# atanh
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.atanh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_atanh(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# atanh_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.atanh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_atanh_(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arctanh
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arctanh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arctanh(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# arctanh_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.arctanh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_arctanh_(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# pow
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.pow",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_instance_pow(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# pow_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.pow_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_instance_pow_(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_and_x,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# argmax
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.argmax",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        force_int_axis=True,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        min_value=1,
        max_value=5,
        valid_axis=True,
        allow_neg_axes=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_instance_argmax(
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    dtype_input_axis,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    keepdim,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_ceil(
    dtype_and_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# min
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.min",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
    ),
)
def test_torch_instance_min(
    dtype_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    frontend,
    init_name,
    method_name,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


@st.composite
def _get_dtype_and_multiplicative_matrices(draw):
    return draw(
        st.one_of(
            _get_dtype_input_and_matrices(),
            _get_dtype_and_3dbatch_matrices(),
        )
    )


# matmul
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.matmul",
    dtype_indtype_tensor1_tensor2=_get_dtype_and_multiplicative_matrices(),
)
def test_torch_instance_matmul(
    dtype_tensor1_tensor2,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    dtype, tensor1, tensor2 = dtype_tensor1_tensor2
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": tensor1,
        },
        method_input_dtypes=dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={"tensor2": tensor2},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


@st.composite
def _array_idxes_n_dtype(draw, **kwargs):
    num_dims = draw(helpers.ints(min_value=1, max_value=4))
    dtype, x = draw(
        helpers.dtype_and_values(
            **kwargs, min_num_dims=num_dims, max_num_dims=num_dims, shared_dtype=True
        )
    )
    idxes = draw(
        st.lists(
            helpers.ints(min_value=0, max_value=num_dims - 1),
            min_size=num_dims,
            max_size=num_dims,
            unique=True,
        )
    )
    return x, idxes, dtype


# permute
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.permute",
    dtype_values_axis=_array_idxes_n_dtype(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_permute(
    dtype_values_axis,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    init_name,
    method_name,
    frontend,
):
    x, idxes, dtype = dtype_values_axis
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_num_positional_args=method_num_positional_args,
        method_as_variable_flags=as_variable,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "dims": idxes,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# mean
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.mean",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_instance_mean(
    dtype_x,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    frontend,
    init_name,
    method_name,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# transpose
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.transpose",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_transpose(
    dtype_value,
    dim0,
    dim1,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={"dim0": dim0, "dim1": dim1},
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# transpose_
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.transpose_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim0=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dim1=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_transpose_(
    dtype_value,
    dim0,
    dim1,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "dim0": dim0,
            "dim1": dim1,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# flatten
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.flatten",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    start_dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    end_dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_instance_flatten(
    dtype_value,
    start_dim,
    end_dim,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "start_dim": start_dim,
            "end_dim": end_dim,
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )


# cumsum
@handle_frontend_method(
    init_name="tensor",
    method_tree="torch.Tensor.cumsum",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dtypes=_dtypes(),
)
def test_torch_instance_cumsum(
    dtype_value,
    dim,
    dtypes,
    init_num_positional_args: pf.NumPositionalArgFn,
    method_num_positional_args: pf.NumPositionalArgMethod,
    as_variable: pf.AsVariableFlags,
    native_array: pf.NativeArrayFlags,
    init_name,
    method_name,
    frontend,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_as_variable_flags=as_variable,
        init_num_positional_args=init_num_positional_args,
        init_native_array_flags=native_array,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_as_variable_flags=as_variable,
        method_num_positional_args=method_num_positional_args,
        method_native_array_flags=native_array,
        method_all_as_kwargs_np={
            "dim": dim,
            "dtype": dtypes[0],
        },
        frontend=frontend,
        init_name=init_name,
        method_name=method_name,
    )
