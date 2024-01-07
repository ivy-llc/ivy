# global
import numpy as np
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.testing_helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _topk_helper(draw):
    dtype, x, axis = draw(
        helpers.dtype_values_axis(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_num_dims=1,
            force_int_axis=True,
            valid_axis=True,
        )
    )
    k = draw(st.integers(min_value=1, max_value=x[0].shape[axis]))
    return dtype, x, axis, k


# --- Main --- #
# ------------ #


# allclose
@handle_frontend_test(
    fn_tree="torch.allclose",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    equal_nan=st.booleans(),
)
def test_torch_allclose(
    *,
    dtype_and_input,
    equal_nan,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-05,
        atol=1e-08,
        input=input[0],
        other=input[1],
        equal_nan=equal_nan,
    )


# argsort
@handle_frontend_test(
    fn_tree="torch.argsort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    descending=st.booleans(),
)
def test_torch_argsort(
    *,
    dtype_input_axis,
    descending,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dim=axis,
        descending=descending,
    )


# eq
@handle_frontend_test(
    fn_tree="torch.eq",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
)
def test_torch_eq(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    inputs_dtypes, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=inputs_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# equal
@handle_frontend_test(
    fn_tree="torch.equal",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=False),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
)
def test_torch_equal(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    inputs_dtypes, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=inputs_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# fmax
@handle_frontend_test(
    fn_tree="torch.fmax",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_fmax(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# fmin
@handle_frontend_test(
    fn_tree="torch.fmin",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_fmin(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# greater
@handle_frontend_test(
    fn_tree="torch.gt",
    aliases=["torch.greater"],
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
)
def test_torch_greater(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# greater_equal
@handle_frontend_test(
    fn_tree="torch.ge",
    aliases=["torch.greater_equal"],
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
)
def test_torch_greater_equal(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# isclose
@handle_frontend_test(
    fn_tree="torch.isclose",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    equal_nan=st.booleans(),
)
def test_torch_isclose(
    *,
    dtype_and_input,
    equal_nan,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-05,
        atol=1e-08,
        input=input[0],
        other=input[1],
        equal_nan=equal_nan,
    )


# isfinite
@handle_frontend_test(
    fn_tree="torch.isfinite",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_isfinite(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


@handle_frontend_test(
    fn_tree="torch.isin",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
    assume_unique=st.booleans(),
    invert=st.booleans(),
)
def test_torch_isin(
    *,
    dtype_and_inputs,
    assume_unique,
    invert,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        elements=inputs[0],
        test_elements=inputs[1],
        assume_unique=assume_unique,
        invert=invert,
    )


# isinf
@handle_frontend_test(
    fn_tree="torch.isinf",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_isinf(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# isnan
@handle_frontend_test(
    fn_tree="torch.isnan",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=False),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_isnan(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# isneginf
@handle_frontend_test(
    fn_tree="torch.isneginf",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_isneginf(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# isposinf
@handle_frontend_test(
    fn_tree="torch.isposinf",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_isposinf(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# isreal
@handle_frontend_test(
    fn_tree="torch.isreal",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
)
def test_torch_isreal(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
    )


# kthvalue
@handle_frontend_test(
    fn_tree="torch.kthvalue",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ).filter(lambda v: len(np.unique(v[1][0])) == len(np.ravel(v[1][0]))),
    k=st.integers(min_value=1),
    keepdim=st.booleans(),
)
def test_torch_kthvalue(
    *,
    dtype_input_axis,
    k,
    keepdim,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input, dim = dtype_input_axis
    assume(k <= input[0].shape[dim])
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        k=k,
        dim=dim,
        keepdim=keepdim,
    )


# less
@handle_frontend_test(
    fn_tree="torch.less",
    aliases=["torch.lt"],
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_less(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# less_equal
@handle_frontend_test(
    fn_tree="torch.less_equal",
    aliases=["torch.le"],
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_less_equal(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# maximum
@handle_frontend_test(
    fn_tree="torch.maximum",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_maximum(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


@handle_frontend_test(
    fn_tree="torch.minimum",
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_minimum(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# msort
@handle_frontend_test(
    fn_tree="torch.msort",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
)
def test_torch_msort(
    *,
    dtype_and_input,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=x[0],
    )


# not_equal
@handle_frontend_test(
    fn_tree="torch.not_equal",
    aliases=["torch.ne"],
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=False),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_not_equal(
    *,
    dtype_and_inputs,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=inputs[0],
        other=inputs[1],
    )


# sort
@handle_frontend_test(
    fn_tree="torch.sort",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        min_axis=-1,
        max_axis=0,
    ),
    descending=st.booleans(),
    stable=st.booleans(),
)
def test_torch_sort(
    *,
    dtype_input_axis,
    descending,
    stable,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dim=axis,
        descending=descending,
        stable=stable,
    )


# topk
# TODO: add value test after the stable sorting is added to torch
# https://github.com/pytorch/pytorch/issues/88184
@handle_frontend_test(
    fn_tree="torch.topk",
    dtype_x_axis_k=_topk_helper(),
    largest=st.booleans(),
    sorted=st.booleans(),
)
def test_torch_topk(
    dtype_x_axis_k,
    largest,
    sorted,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
):
    input_dtype, input, axis, k = dtype_x_axis_k
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=input[0],
        k=k,
        dim=axis,
        largest=largest,
        sorted=sorted,
        test_values=False,
    )
