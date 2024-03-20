# global
import pytest
from types import SimpleNamespace
import numpy as np

from ivy_tests.test_ivy.test_frontends.test_torch.test_comparison_ops import (
    _topk_helper,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_creation_ops import (
    _as_strided_helper,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_indexing_slicing_joining_mutating_ops import (  # noqa: E501
    _dtype_input_dim_start_length,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_reduction_ops import (
    _get_axis_and_p,
)

import ivy
from hypothesis import strategies as st, given, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.hypothesis_helpers.general_helpers import sizes_
from ivy_tests.test_ivy.test_frontends.test_torch.test_blas_and_lapack_ops import (
    _get_dtype_and_3dbatch_matrices,
    _get_dtype_input_and_matrices,
    _get_dtype_input_and_mat_vec,
)
from ivy.functional.frontends.torch import Tensor
from ivy_tests.test_ivy.helpers import handle_frontend_method, BackendHandler
from ivy_tests.test_ivy.test_functional.test_core.test_searching import (
    _broadcastable_trio,
)
from ivy_tests.test_ivy.test_functional.test_core.test_manipulation import (  # noqa
    _get_splits,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_miscellaneous_ops import (  # noqa
    dtype_value1_value2_axis,
    _get_dtype_value1_value2_cov,
)
from ivy_tests.test_ivy.test_frontends.test_torch.test_linalg import (  # noqa
    _get_dtype_and_matrix,
)
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    _get_castable_dtype,
    _statistical_dtype_values,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_manipulation import (  # noqa
    put_along_axis_helper,
)
from ivy_tests.test_ivy.test_functional.test_experimental.test_core.test_statistical import (  # noqa
    _quantile_helper,
)

try:
    import torch
except ImportError:
    torch = SimpleNamespace()

CLASS_TREE = "ivy.functional.frontends.torch.Tensor"


# --- Helpers --- #
# --------------- #


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


@st.composite
def _arrays_dim_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = 2
    common_shape = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    _dim = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            x=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )

    min_dim = min(unique_dims)
    max_dim = max(unique_dims)
    _idx = draw(
        helpers.array_values(
            shape=min_dim,
            dtype="int64",
            min_value=0,
            max_value=max_dim,
            exclude_min=False,
        )
    )

    xs = []
    available_input_types = draw(helpers.get_dtypes("numeric"))
    input_dtypes = draw(
        helpers.array_dtypes(
            available_dtypes=available_input_types,
            num_arrays=num_arrays,
            shared_dtype=True,
        )
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:_dim] + [ud] + common_shape[_dim:],
                dtype=dt,
                large_abs_safety_factor=2.5,
                small_abs_safety_factor=2.5,
                safety_factor_scale="log",
            )
        )
        xs.append(x)
    return xs, input_dtypes, _dim, _idx


# Helper functions
@st.composite
def _dtypes(draw):
    return draw(
        st.shared(
            helpers.list_of_size(
                x=st.sampled_from(
                    draw(helpers.get_dtypes("numeric", prune_function=False))
                ),
                size=1,
            ),
            key="dtype",
        )
    )


@st.composite
def _expand_helper(draw):
    num_dims = draw(st.integers(min_value=1, max_value=10))
    shape = draw(
        helpers.get_shape(min_num_dims=num_dims, max_num_dims=num_dims).filter(
            lambda x: any(i == 1 for i in x)
        )
    )
    new_shape = draw(
        helpers.get_shape(min_num_dims=num_dims, max_num_dims=num_dims).filter(
            lambda x: all(x[i] == v if v != 1 else True for i, v in enumerate(shape))
        )
    )
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )
    return dtype, x, new_shape


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
    fill_value = draw(helpers.ints()) if "int" in dtype[0] else draw(helpers.floats())

    return dtype, [array, size, fill_value]


@st.composite
def _get_clamp_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    min = draw(st.booleans())
    if min:
        max = draw(st.booleans())
        min = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=0, max_value=25
            )
        )
        max = (
            draw(
                helpers.array_values(
                    dtype=x_dtype[0], shape=shape, min_value=26, max_value=50
                )
            )
            if max
            else None
        )
    else:
        min = None
        max = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=26, max_value=50
            )
        )
    return x_dtype, x, min, max


@st.composite
def _get_clip_min_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )

    min = draw(
        helpers.array_values(dtype=x_dtype[0], shape=shape, min_value=0, max_value=25)
    )

    return x_dtype, x, min


@st.composite
def _get_dtype_and_multiplicative_matrices(draw):
    return draw(
        st.one_of(
            _get_dtype_input_and_matrices(),
            _get_dtype_and_3dbatch_matrices(),
        )
    )


@st.composite
def _get_dtype_input_and_vectors(draw, with_input=False, same_size=False):
    dim_size1 = draw(helpers.ints(min_value=2, max_value=5))
    dim_size2 = dim_size1 if same_size else draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", full=True))
    dtype = [
        draw(st.sampled_from(tuple(set(dtype).difference({"bfloat16", "float16"}))))
    ]
    vec1 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size1,), min_value=2, max_value=5
        )
    )
    vec2 = draw(
        helpers.array_values(
            dtype=dtype[0], shape=(dim_size2,), min_value=2, max_value=5
        )
    )
    if with_input:
        input = draw(
            helpers.array_values(
                dtype=dtype[0], shape=(dim_size1, dim_size2), min_value=2, max_value=5
            )
        )
        return dtype, input, vec1, vec2
    return dtype, vec1, vec2


@st.composite
def _masked_fill_helper(draw):
    cond, xs, dtypes = draw(_broadcastable_trio())
    if ivy.is_uint_dtype(dtypes[0]):
        fill_value = draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtypes[0]):
        fill_value = draw(helpers.ints(min_value=-5, max_value=5))
    else:
        fill_value = draw(helpers.floats(min_value=-5, max_value=5))
    return dtypes[0], xs[0], cond, fill_value


@st.composite
def _repeat_helper(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )

    input_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            shape=shape,
        )
    )

    MAX_NUMPY_DIMS = 32
    repeats = draw(
        st.lists(
            st.integers(min_value=1, max_value=5),
            min_size=len(shape),
            max_size=MAX_NUMPY_DIMS,
        )
    )
    assume(np.prod(repeats) * np.prod(shape) <= 2**28)
    return input_dtype, x, repeats


@st.composite
def _requires_grad_and_dtypes(draw):
    dtypes = draw(_dtypes())
    dtype = dtypes[0]
    if ivy.is_int_dtype(dtype) or ivy.is_uint_dtype(dtype):
        return draw(st.just(False)), dtypes
    return draw(st.booleans()), dtypes


@st.composite
def _to_helper(draw):
    dtype_x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=2,
            large_abs_safety_factor=3,
        )
    )
    input_dtype, x = dtype_x
    arg = draw(st.sampled_from(["tensor", "dtype", "device"]))
    if arg == "tensor":
        method_num_positional_args = 1
        method_all_as_kwargs_np = {"other": x[1]}
    elif arg == "dtype":
        method_num_positional_args = 1
        dtype = draw(helpers.get_dtypes("valid", full=False))[0]
        method_all_as_kwargs_np = {"dtype": dtype}
    else:
        method_num_positional_args = 0
        device = draw(st.just("cpu"))
        dtype = draw(helpers.get_dtypes("valid", full=False, none=True))[0]
        method_all_as_kwargs_np = {"dtype": dtype, "device": device}
    return input_dtype, x, method_num_positional_args, method_all_as_kwargs_np


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
            max_value=max(shape[axis] - 1, 1),
        )
    )
    step = draw(
        st.integers(
            min_value=1,
            max_value=size,
        )
    )
    return values_dtype, values, axis, size, step


# --- Main --- #
# ------------ #


# __add__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__add__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___add__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __and__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__and__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___and__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__array__",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch___array__(
    dtype_and_x,
    dtype,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    if x[0].dtype == "bfloat16":
        return
    dtype[0] = np.dtype(dtype[0])
    ret_gt = torch.tensor(x[0]).__array__(dtype[0])
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        local_importer = ivy_backend.utils.dynamic_import
        function_module = local_importer.import_module("ivy.functional.frontends.torch")
        ret = function_module.tensor(x[0]).__array__(dtype[0])

    helpers.value_test(
        ret_np_flat=ret.ravel(),
        ret_np_from_gt_flat=ret_gt.ravel(),
        ground_truth_backend="torch",
        backend=backend_fw,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__array_wrap__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch___array_wrap__(
    dtype_and_x,
    backend_fw,
    frontend,
):
    input_dtypes, x = dtype_and_x
    if x[1].dtype == "bfloat16":
        return
    if x[0].dtype == "bfloat16":
        ret_gt = torch.tensor(x[0].tolist(), dtype=torch.bfloat16).__array_wrap__(x[1])
    else:
        ret_gt = torch.tensor(x[0]).__array_wrap__(x[1])
    with BackendHandler.update_backend(backend_fw) as ivy_backend:
        local_importer = ivy_backend.utils.dynamic_import
        function_module = local_importer.import_module("ivy.functional.frontends.torch")
        ret = function_module.tensor(x[0]).__array_wrap__(x[1])
        assert isinstance(ret, function_module.Tensor)
    helpers.value_test(
        ret_np_flat=np.array(ret.ivy_array).ravel(),
        ret_np_from_gt_flat=ret_gt.numpy().ravel(),
        ground_truth_backend="torch",
        backend=backend_fw,
    )


# __bool__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__bool__",
    dtype_and_x=helpers.dtype_and_values(
        max_dim_size=1,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch___bool__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __eq__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__eq__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___eq__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__floordiv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_torch___floordiv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1,
    )


# __getitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__getitem__",
    dtype_x_index=helpers.dtype_array_query(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ),
)
def test_torch___getitem__(
    dtype_x_index,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, index = dtype_x_index
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"query": index},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __gt__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__gt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch___gt__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__iand__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch___iand__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __invert__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__invert__",
    dtype_and_x=helpers.dtype_and_values(),
)
def test_torch___invert__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __long__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__long__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___long__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __lt__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__lt__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___lt__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __matmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__matmul__",
    dtype_tensor1_tensor2=_get_dtype_and_multiplicative_matrices(),
)
def test_torch___matmul__(
    dtype_tensor1_tensor2,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    dtype, tensor1, tensor2 = dtype_tensor1_tensor2
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": tensor1,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": tensor2},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __mod__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__mod__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch___mod__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __mul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__mul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___mul__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __ne__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__ne__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___ne__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __neg__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__neg__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___neg__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __or__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__or__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___or__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __pow__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__pow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch___pow__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[1] = ivy.abs(x[1])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "exponent": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __radd__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__radd__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___radd__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __rmul__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__rmul__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___rmul__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __rpow__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__rpow__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=1,
    ),
)
def test_torch___rpow__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[0] = ivy.abs(x[0])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __rsub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__rsub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch___rsub__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __setitem__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__setitem__",
    dtypes_x_index_val=helpers.dtype_array_query_val(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_neg_step=False,
    ).filter(lambda x: x[0][0] == x[0][-1]),
)
def test_torch___setitem__(
    dtypes_x_index_val,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, index, val = dtypes_x_index_val
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[*input_dtype[1:]],
        method_all_as_kwargs_np={"key": index, "value": val},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __sub__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__sub__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___sub__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# __truediv__
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="__truediv__",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch___truediv__(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ),
    requires_grad=st.booleans(),
)
def test_torch__requires_grad(
    dtype_x,
    requires_grad,
    backend_fw,
):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    assert not x._requires_grad
    x.requires_grad_()
    assert x._requires_grad
    x.requires_grad_(requires_grad)
    assert x._requires_grad == requires_grad
    ivy.previous_backend()


# abs
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="abs",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_abs(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# abs_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="abs_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_abs_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_acos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acos_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acos_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_acos_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acosh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_acosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# acosh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="acosh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_acosh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# add
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="add",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
)
def test_torch_add(
    dtype_and_x,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# add_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="add_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
    test_inplace=st.just(True),
)
def test_torch_add_(
    dtype_and_x,
    alpha,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# addbmm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addbmm(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, batch1, batch2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "batch1": batch1,
            "batch2": batch2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addbmm_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addbmm_",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_addbmm_(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, batch1, batch2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "batch1": batch1,
            "batch2": batch2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addcdiv
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcdiv",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
)
def test_torch_addcdiv(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[2], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-03,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcdiv_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
    test_inplace=st.just(True),
)
def test_torch_addcdiv_(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[2], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-03,
    )


# addcmul
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcmul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
)
def test_torch_addcmul(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-02,
    )


# addcmul_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addcmul_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=3,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
        shared_dtype=True,
    ),
    value=st.floats(min_value=-100, max_value=100),
    test_inplace=st.just(True),
)
def test_torch_addcmul_(
    dtype_and_x,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor1": x[1],
            "tensor2": x[2],
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        atol_=1e-02,
    )


# addmm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addmm",
    dtype_and_matrices=_get_dtype_input_and_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addmm(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "mat1": mat1,
            "mat2": mat2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addmm_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addmm_",
    dtype_and_matrices=_get_dtype_input_and_matrices(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_addmm_(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, mat1, mat2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "mat1": mat1,
            "mat2": mat2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addmv
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addmv",
    dtype_and_matrices=_get_dtype_input_and_mat_vec(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addmv(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, mat, vec = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "mat": mat,
            "vec": vec,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addmv_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addmv_",
    dtype_and_matrices=_get_dtype_input_and_mat_vec(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_addmv_(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, mat, vec = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_all_as_kwargs_np={
            "mat": mat,
            "vec": vec,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addr
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addr",
    dtype_and_vecs=_get_dtype_input_and_vectors(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_addr(
    dtype_and_vecs,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    dtype, input, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "vec1": vec1,
            "vec2": vec2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# addr_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="addr_",
    dtype_and_vecs=_get_dtype_input_and_vectors(with_input=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_addr_(
    dtype_and_vecs,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    dtype, input, vec1, vec2 = dtype_and_vecs
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "vec1": vec1,
            "vec2": vec2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="adjoint",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("real_and_complex"),
        min_num_dims=2,
        min_dim_size=2,
    ),
)
def test_torch_adjoint(
    dtype_and_values,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, values = dtype_and_values

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


# all
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="all",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_all(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# amax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="amax",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_amax(
    dtype_x_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# amin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="amin",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_amin(
    dtype_x_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# aminmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="aminmax",
    dtype_input_axis=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_aminmax(
    dtype_input_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# angle
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="angle",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=["float64", "complex64", "complex128"],
    ),
)
def test_torch_angle(
    dtype_and_values,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, values = dtype_and_values

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


# any
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="any",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_any(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# write test for torch instance apply_


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="apply_",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
    ),
    test_inplace=st.just(True),
)
def test_torch_apply_(
    dtype_and_values,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    def func(x):
        return x + 1

    input_dtype, values = dtype_and_values

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "callable": func,
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


# arccos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccos",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_arccos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arccos_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccos_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_arccos_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arccosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccosh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_arccosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arccosh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arccosh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_arccosh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arcsin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arcsin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_arcsin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arcsin_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arcsin_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_arcsin_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arcsinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arcsinh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_arcsinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arcsinh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arcsinh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_arcsinh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_arctan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_arctan2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan2_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan2_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_arctan2_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctan_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_arctan_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctanh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_arctanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# arctanh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="arctanh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_arctanh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argmax",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        force_int_axis=True,
        valid_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_argmax(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argmin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argmin",
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
def test_torch_argmin(
    dtype_input_axis,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argsort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argsort",
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
    descending=st.booleans(),
)
def test_torch_argsort(
    dtype_input_axis,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "descending": descending,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# argwhere
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="argwhere",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_argwhere(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="as_strided",
    dtype_x_and_other=_as_strided_helper(),
)
def test_torch_as_strided(
    dtype_x_and_other,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, size, stride, offset = dtype_x_and_other
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "size": size,
            "stride": stride,
            "storage_offset": offset,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# asin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_asin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asin_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asin_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_asin_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# asinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_asinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


# asinh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="asinh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_asinh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        rtol_=1e-2,
        atol_=1e-2,
        on_device=on_device,
    )


# atan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_atan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atan2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_atan2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atan2_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan2_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_atan2_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atan_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_atan_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atanh",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_atanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# atanh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="atanh_",
    dtype_and_x=helpers.dtype_and_values(
        min_value=-1.0,
        max_value=1.0,
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_atanh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", prune_function=False),
        num_arrays=3,
        min_value=-1e3,
        max_value=1e3,
    ).filter(lambda x: all(dt == "float32" for dt in x[0])),
)
def test_torch_backward(
    dtype_x,
    backend_fw,
):
    ivy.set_backend(backend_fw)
    if ivy.current_backend_str() == "numpy":
        ivy.warnings.warn("Gradient calculation unavailable for numpy backend")
        return
    if ivy.current_backend_str() == "paddle":
        ivy.warnings.warn("torch.Tensor.backward() unavailable for paddle backend")
        return
    _, values = dtype_x
    x = Tensor(values[0], requires_grad=True)
    y = Tensor(values[1], requires_grad=True)
    z = Tensor(values[2], requires_grad=True)
    a = x + y.pow(2)
    b = z * a
    c = b.sum()
    c.backward()
    x_torch = torch.tensor(values[0], requires_grad=True, dtype=torch.float32)
    y_torch = torch.tensor(values[1], requires_grad=True, dtype=torch.float32)
    z_torch = torch.tensor(values[2], requires_grad=True, dtype=torch.float32)
    a_torch = x_torch + y_torch.pow(2)
    b_torch = z_torch * a_torch
    c_torch = b_torch.sum()
    c_torch.backward()
    helpers.assertions.value_test(
        ret_np_flat=helpers.flatten_and_to_np(
            ret=x._grads.ivy_array, backend=backend_fw
        ),
        ret_np_from_gt_flat=helpers.flatten_and_to_np(
            ret=ivy.to_ivy(x_torch.grad.numpy()), backend=backend_fw
        ),
        rtol=1e-3,
        atol=1e-3,
        backend="torch",
    )
    helpers.assertions.value_test(
        ret_np_flat=helpers.flatten_and_to_np(
            ret=y._grads.ivy_array, backend=backend_fw
        ),
        ret_np_from_gt_flat=helpers.flatten_and_to_np(
            ret=ivy.to_ivy(y_torch.grad.numpy()), backend=backend_fw
        ),
        rtol=1e-3,
        atol=1e-3,
        backend="torch",
    )
    helpers.assertions.value_test(
        ret_np_flat=helpers.flatten_and_to_np(
            ret=z._grads.ivy_array, backend=backend_fw
        ),
        ret_np_from_gt_flat=helpers.flatten_and_to_np(
            ret=ivy.to_ivy(z_torch.grad.numpy()), backend=backend_fw
        ),
        rtol=1e-3,
        atol=1e-3,
        backend="torch",
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="baddbmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True, input_3d=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
)
def test_torch_baddbmm(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, batch1, batch2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "batch1": batch1,
            "batch2": batch2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="baddbmm_",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True, input_3d=True),
    beta=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    alpha=st.floats(
        min_value=-5,
        max_value=5,
        allow_nan=False,
        allow_subnormal=False,
        allow_infinity=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_baddbmm_(
    dtype_and_matrices,
    beta,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, batch1, batch2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "batch1": batch1,
            "batch2": batch2,
            "beta": beta,
            "alpha": alpha,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# bernoulli
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bernoulli",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_with_out=st.just(True),
)
def test_torch_bernoulli(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "input": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"generator": x[1], "out": x[2]},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_bitwise_and(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_and_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_and_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_bitwise_and_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_left_shift
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_left_shift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_bitwise_left_shift(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
)
def test_torch_bitwise_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        method_all_as_kwargs_np={},
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_not_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_not_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_bitwise_not_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        method_all_as_kwargs_np={},
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_or
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_bitwise_or(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_or_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_or_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_bitwise_or_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise right shift
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_right_shift",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_bitwise_right_shift(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype width produce backend-defined behavior
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1), dtype=input_dtype[1]
    )
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_right_shift_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_right_shift_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_bitwise_right_shift_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    # negative shifts will throw an exception
    # shifts >= dtype width produce backend-defined behavior
    x[1] = np.asarray(
        np.clip(x[1], 0, np.iinfo(input_dtype[1]).bits - 1), dtype=input_dtype[1]
    )
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_xor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_xor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
    ),
)
def test_torch_bitwise_xor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# bitwise_xor_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bitwise_xor_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.one_of(st.just(("bool",)), helpers.get_dtypes("integer")),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_bitwise_xor_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bmm",
    dtype_and_matrices=_get_dtype_and_3dbatch_matrices(with_input=True, input_3d=True),
)
def test_torch_bmm(
    dtype_and_matrices,
    backend_fw,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, _, x, mat2 = dtype_and_matrices
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"mat2": mat2},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
    )


# bool
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="bool",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_torch_bool(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# byte
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="byte",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_byte(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ceil
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ceil",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_ceil(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ceil_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ceil_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_ceil_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# char
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="char",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-128,
        max_value=127,
    ),
)
def test_torch_char(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cholesky",
    dtype_and_x=_get_dtype_and_matrix(square=True),
    upper=st.booleans(),
)
def test_torch_cholesky(
    dtype_and_x,
    upper,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    x = x[0]
    # make symmetric positive-definite
    x = np.matmul(x.swapaxes(-1, -2), x) + np.identity(x.shape[-1]) * 1e-3

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "upper": upper,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
    )


# chunk
@pytest.mark.skip("Testing takes a lot of time")
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="chunk",
    dtype_x_dim=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
        force_int_axis=True,
        valid_axis=True,
    ),
    chunks=st.integers(
        min_value=1,
        max_value=5,
    ),
)
def test_torch_chunk(
    dtype_x_dim,
    chunks,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, dim = dtype_x_dim
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "chunks": chunks,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clamp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clamp",
    dtype_and_x_min_max=_get_clamp_inputs(),
)
def test_torch_clamp(
    dtype_and_x_min_max,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, min, max = dtype_and_x_min_max
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clamp_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clamp_",
    dtype_and_x_min_max=_get_clamp_inputs(),
    test_inplace=st.just(True),
)
def test_torch_clamp_(
    dtype_and_x_min_max,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, min, max = dtype_and_x_min_max
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clamp_min",
    input_and_ranges=_get_clip_min_inputs(),
)
def test_torch_clamp_min(
    input_and_ranges,
    frontend_method_data,
    init_flags,
    backend_fw,
    frontend,
    on_device,
    method_flags,
):
    x_dtype, x, min = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=x_dtype,
        method_all_as_kwargs_np={
            "min": min,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clamp_min_",
    input_and_ranges=_get_clip_min_inputs(),
    test_inplace=st.just(True),
)
def test_torch_clamp_min_(
    input_and_ranges,
    frontend_method_data,
    init_flags,
    backend_fw,
    frontend,
    on_device,
    method_flags,
):
    x_dtype, x, min = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=x_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=x_dtype,
        method_all_as_kwargs_np={
            "min": min,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clip
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clip",
    input_and_ranges=_get_clamp_inputs(),
)
def test_torch_clip(
    input_and_ranges,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clip_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clip_",
    input_and_ranges=_get_clamp_inputs(),
)
def test_torch_clip_(
    input_and_ranges,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, min, max = input_and_ranges
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"min": min, "max": max},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# clone
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="clone",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
    ),
)
def test_torch_clone(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="conj",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_complex")
    ),
)
def test_torch_conj(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# contiguous
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="contiguous",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_contiguous(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# copy_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="copy_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_copy_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# copysign
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="copysign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        num_arrays=2,
    ),
)
def test_torch_copysign(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# copysign_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="copysign_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_copysign_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cos
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cos",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_cos(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cos_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cos_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_cos_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": list(x[0]) if isinstance(x[0], int) else x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cosh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cosh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_cosh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cosh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cosh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_cosh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
    )


# count_nonzero
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="count_nonzero",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
)
def test_torch_count_nonzero(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim": dim},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cov
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cov",
    dtype_and_x=_get_dtype_value1_value2_cov(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        max_dim_size=5,
        min_value=1,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torch_cov(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, correction, fweights, aweights = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=["float64", "int64", "float64"],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["int64", "float64"],
        method_all_as_kwargs_np={
            "correction": correction,
            "fweights": fweights,
            "aweights": aweights,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
    )


# cross
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cross",
    dtype_input_other_dim=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=3,
        max_dim_size=3,
        min_value=-1e10,
        max_value=1e10,
        abs_smallest_val=0.01,
        large_abs_safety_factor=2,
        safety_factor_scale="log",
    ),
)
def test_torch_cross(
    dtype_input_other_dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    dtype, input, other, dim = dtype_input_other_dim
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "other": other,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ).filter(
        lambda x: "bfloat16" not in x[0]
        and "uint16" not in x[0]
        and "uint32" not in x[0]
        and "uint64" not in x[0]
    ),
)
def test_torch_cuda(dtype_x, backend_fw):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0], device="gpu:0")
    device = "gpu:0"
    ivy.utils.assertions.check_equal(x.cuda, device, as_array=False)
    ivy.previous_backend()


# cummax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cummax",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=False,
        force_int=True,
    ),
)
def test_torch_cummax(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim": dim},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cumprod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cumprod",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dtypes=_dtypes(),
)
def test_torch_cumprod(
    dtype_value,
    dim,
    dtypes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "dim": dim,
            "dtype": dtypes[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cumsum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cumsum",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    dtypes=_dtypes(),
)
def test_torch_cumsum(
    dtype_value,
    dim,
    dtypes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "dim": dim,
            "dtype": dtypes[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# cumsum_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="cumsum_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    test_inplace=st.just(True),
)
def test_torch_cumsum_(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "dtype": input_dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# det
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="det",
    dtype_and_x=_get_dtype_and_matrix(square=True, batch=True),
)
def test_torch_det(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# detach
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="detach",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_detach(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# detach_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="detach_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_inplace=st.just(True),
)
def test_torch_detach_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_device(
    dtype_x,
    backend_fw,
):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.device, ivy.dev(ivy.array(data[0])), as_array=False
    )
    ivy.previous_backend()


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="diag",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=2), key="shape"),
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_diag(
    dtype_and_values,
    diagonal,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, values = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "diagonal": diagonal,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="diagonal",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dims_and_offset=helpers.dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
)
def test_torch_diagonal(
    dtype_and_values,
    dims_and_offset,
    frontend,
    frontend_method_data,
    backend_fw,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, value = dtype_and_values
    dim1, dim2, offset = dims_and_offset
    input = value[0]
    num_dims = len(np.shape(input))
    assume(dim1 != dim2)
    if dim1 < 0:
        assume(dim1 + num_dims != dim2)
    if dim2 < 0:
        assume(dim1 != dim2 + num_dims)
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        init_all_as_kwargs_np={"data": input},
        method_input_dtypes=[input_dtype[0]],
        method_all_as_kwargs_np={
            "offset": offset,
            "dim1": dim1,
            "dim2": dim2,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        backend_to_test=backend_fw,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# diff
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="diff",
    dtype_n_x_n_axis=helpers.dtype_values_axis(
        available_dtypes=st.shared(helpers.get_dtypes("valid"), key="dtype"),
        min_num_dims=1,
        min_value=-1e09,
        max_value=1e09,
        valid_axis=True,
        force_int_axis=True,
    ),
    n=st.integers(min_value=0, max_value=5),
    dtype_prepend=helpers.dtype_and_values(
        available_dtypes=st.shared(helpers.get_dtypes("numeric"), key="dtype"),
        min_num_dims=1,
        max_num_dims=1,
    ),
    dtype_append=helpers.dtype_and_values(
        available_dtypes=st.shared(helpers.get_dtypes("numeric"), key="dtype"),
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_torch_diff(
    dtype_n_x_n_axis,
    n,
    dtype_prepend,
    dtype_append,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_n_x_n_axis
    _, prepend = dtype_prepend
    _, append = dtype_append
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "n": n,
            "dim": axis,
            "prepend": prepend[0],
            "append": append[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# dim
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="dim",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_dim(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# div
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="div",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    rounding_mode=st.sampled_from(["floor", "trunc"]) | st.none(),
)
def test_torch_div(
    dtype_and_x,
    rounding_mode,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "rounding_mode": rounding_mode,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# div_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="div_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    rounding_mode=st.sampled_from(["floor", "trunc"]) | st.none(),
    test_inplace=st.just(True),
)
def test_torch_div_(
    dtype_and_x,
    rounding_mode,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "rounding_mode": rounding_mode,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# divide
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="divide",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_divide(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="dot",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shape=(1,),
    ),
)
def test_torch_dot(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "tensor": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="double",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_double(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# dsplit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="dsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=3), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=3,
        axis=2,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_dsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={"indices_or_sections": indices_or_sections},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_dtype(dtype_x, backend_fw):
    ivy.set_backend(backend_fw)
    dtype, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.dtype, dtype[0], as_array=False)
    ivy.previous_backend()


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="eq_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_eq_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_equal(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-04,
        rtol_=1e-04,
        on_device=on_device,
    )


# erf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="erf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_erf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# erf_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="erf_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_inplace=st.just(True),
)
def test_torch_erf_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# erfinv_ tests
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="erfinv_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1,
        max_value=1,
        abs_smallest_val=1e-05,
    ),
)
def test_torch_erfinv(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# erfinv_ tests
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="erfinv_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1,
        max_value=1,
        abs_smallest_val=1e-05,
    ),
    test_inplace=st.just(True),
)
def test_torch_erfinv_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# exp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="exp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_exp(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# exp_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="exp_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    test_inplace=st.just(True),
)
def test_torch_exp_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expand",
    dtype_x_shape=_expand_helper(),
    unpack_shape=st.booleans(),
)
def test_torch_expand(
    dtype_x_shape,
    unpack_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, shape = dtype_x_shape

    if backend_fw == "paddle":
        assume(
            input_dtype[0] in ["int32", "int64", "float32", "float64", "bool"]
            and len(shape) < 7
        )

    if unpack_shape:
        method_flags.num_positional_args = len(shape) + 1
        size = {}
        i = 0
        for x_ in shape:
            size[f"x{i}"] = x_
            i += 1
    else:
        size = {
            "size": shape,
        }
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=size,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# expand_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expand_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    ),
)
def test_torch_expand_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# expm1
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expm1",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_expm1(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# expm1_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="expm1_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_expm1_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fill_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fill_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    value=helpers.floats(min_value=1, max_value=10),
    test_inplace=st.just(True),
)
def test_torch_fill_(
    dtype_and_x,
    value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "value": value,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fix
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fix",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_fix(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fix_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fix_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    test_inplace=st.just(True),
)
def test_torch_fix_(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# flatten
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="flatten",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    axes=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        min_size=2,
        max_size=2,
        unique=False,
        force_tuple=True,
    ),
)
def test_torch_flatten(
    dtype_value,
    axes,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "start_dim": axes[0],
            "end_dim": axes[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# flip
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="flip",
    dtype_values_axis=_array_idxes_n_dtype(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_flip(
    dtype_values_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    x, idxes, dtype = dtype_values_axis
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "dims": idxes,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fliplr
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fliplr",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
    ),
)
def test_torch_fliplr(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="float",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_float(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# floor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="floor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_floor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="floor_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_floor_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_fmax(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fmin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fmin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_fmin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# fmod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fmod",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
    ),
)
def test_torch_fmod(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# fmod_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="fmod_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
    ),
    test_inplace=st.just(True),
)
def test_torch_fmod_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"other": x[1]},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# frac
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="frac",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes(kind="valid"),
        num_arrays=1,
        max_value=1e6,
        min_value=-1e6,
    ),
)
def test_torch_frac(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="gather",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        indices_same_dims=True,
    ),
)
def test_torch_gather(
    params_indices_others,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# gcd
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="gcd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_torch_gcd(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(
        lambda x: "bfloat16" not in x[0]
        and "uint16" not in x[0]
        and "uint32" not in x[0]
        and "uint64" not in x[0]
    ),
)
def test_torch_get_device(
    dtype_x,
    backend_fw,
):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(x.get_device, -1, as_array=False)
    x = Tensor(data[0], "gpu:0")
    ivy.utils.assertions.check_equal(x.get_device, 0, as_array=False)
    x = Tensor(data[0], "tpu:3")
    ivy.utils.assertions.check_equal(x.get_device, 3, as_array=False)
    ivy.previous_backend()


def test_torch_grad(backend_fw):
    ivy.set_backend(backend_fw)
    x = Tensor(ivy.array([1.0, 2.0, 3.0]))
    grads = ivy.array([1.0, 2.0, 3.0])
    x._grads = grads
    assert ivy.array_equal(x.grad, grads)
    ivy.previous_backend()


def test_torch_grad_fn(backend_fw):
    ivy.set_backend(backend_fw)
    x = Tensor(ivy.array([3.0]), requires_grad=True)
    ivy.utils.assertions.check_equal(x.grad_fn, None, as_array=False)
    y = x.pow(2)
    ivy.utils.assertions.check_equal(y.grad_fn, "PowBackward", as_array=False)
    ivy.utils.assertions.check_equal(
        y.grad_fn.next_functions[0], "AccumulateGrad", as_array=False
    )
    z = y.detach()
    ivy.utils.assertions.check_equal(z.grad_fn, None, as_array=False)
    ivy.previous_backend()


# greater
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_greater(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_greater_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_greater_equal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# greater_equal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="greater_equal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_greater_equal_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# half
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="half",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_half(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="heaviside",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
    ),
)
def test_torch_heaviside(
    dtype_and_values,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, values = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "values": values[1],
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


# hsplit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="hsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1,
        axis=1,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_hsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={"indices_or_sections": indices_or_sections},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex", prune_function=False)
    ),
)
def test_torch_imag(dtype_x, backend_fw):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.imag, ivy.imag(data[0]))
    ivy.previous_backend()


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_add",
    xs_dtypes_dim_idx=_arrays_dim_idx_n_dtypes(),
    alpha=st.integers(min_value=1, max_value=2),
)
def test_torch_index_add(
    *,
    xs_dtypes_dim_idx,
    alpha,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    frontend,
    backend_fw,
):
    xs, input_dtypes, axis, indices = xs_dtypes_dim_idx
    if xs[0].shape[axis] < xs[1].shape[axis]:
        source, input = xs
    else:
        input, source = xs
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=["int64", input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "source": source,
            "alpha": alpha,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        rtol_=1e-03,
    )


# index_add
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_add_",
    xs_dtypes_dim_idx=_arrays_dim_idx_n_dtypes(),
    alpha=st.integers(min_value=1, max_value=2),
    test_inplace=st.just(True),
)
def test_torch_index_add_(
    *,
    xs_dtypes_dim_idx,
    alpha,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    frontend,
    backend_fw,
):
    xs, input_dtypes, axis, indices = xs_dtypes_dim_idx
    if xs[0].shape[axis] < xs[1].shape[axis]:
        source, input = xs
    else:
        input, source = xs
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=["int64", input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "source": source,
            "alpha": alpha,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        rtol_=1e-03,
    )


# index_fill
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_fill",
    dtype_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        first_dimension_only=True,
        indices_same_dims=False,
    ),
    value=st.floats(min_value=-100, max_value=100),
)
def test_torch_index_fill(
    dtype_indices_axis,
    value,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, axis, _ = dtype_indices_axis
    if indices.ndim != 1:
        indices = ivy.flatten(indices)
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=[input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "value": value,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# todo: remove dtype specifications
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_put",
    x_and_indices=helpers.array_indices_axis(
        array_dtypes=st.just(("float32",)),
        indices_dtypes=st.just(("int64",)),
    ),
    values=helpers.dtype_and_values(
        available_dtypes=st.just(("float32",)), max_num_dims=1, max_dim_size=1
    ),
    accumulate=st.booleans(),
)
def test_torch_index_put(
    x_and_indices,
    values,
    accumulate,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, indices, *_ = x_and_indices
    values_dtype, values = values
    init_dtypes = [input_dtype[0]]
    method_dtypes = [input_dtype[1], values_dtype[0]]
    helpers.test_frontend_method(
        init_input_dtypes=init_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=method_dtypes,
        method_all_as_kwargs_np={
            "indices": (indices,),
            "values": values[0],
            "accumulate": accumulate,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_put_",
    x_and_indices=helpers.array_indices_axis(
        array_dtypes=st.just(("float32",)),
        indices_dtypes=st.just(("int64",)),
    ),
    values=helpers.dtype_and_values(
        available_dtypes=st.just(("float32",)), max_num_dims=1, max_dim_size=1
    ),
    accumulate=st.booleans(),
    test_inplace=st.just(True),
)
def test_torch_index_put_(
    x_and_indices,
    values,
    accumulate,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, indices, *_ = x_and_indices
    values_dtype, values = values
    init_dtypes = [input_dtype[0]]
    method_dtypes = [input_dtype[1], values_dtype[0]]
    helpers.test_frontend_method(
        init_input_dtypes=init_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=method_dtypes,
        method_all_as_kwargs_np={
            "indices": (indices,),
            "values": values[0],
            "accumulate": accumulate,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# index_select
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="index_select",
    params_indices_others=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("valid"),
        indices_dtypes=["int64"],
        max_num_dims=1,
        indices_same_dims=True,
    ),
)
def test_torch_index_select(
    params_indices_others,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, input, indices, axis, batch_dims = params_indices_others
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": input,
        },
        method_input_dtypes=[input_dtypes[1]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# int
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="int",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_torch_int(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# inverse
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="inverse",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
    ).filter(lambda s: s[1][0].shape[-1] == s[1][0].shape[-2]),
)
def test_torch_inverse(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# is_complex
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="is_complex",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_is_complex(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_is_cuda(
    dtype_x,
    backend_fw,
):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.is_cuda, "gpu" in ivy.dev(ivy.array(data[0])), as_array=False
    )
    ivy.previous_backend()


# is_floating_point
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="is_floating_point",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_is_floating_point(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    requires_grad=st.booleans(),
)
def test_torch_is_leaf(requires_grad, backend_fw):
    ivy.set_backend(backend_fw)
    x = Tensor(ivy.array([3.0]), requires_grad=requires_grad)
    ivy.utils.assertions.check_equal(x.is_leaf, True, as_array=False)
    y = x.pow(2)
    ivy.utils.assertions.check_equal(y.is_leaf, not requires_grad, as_array=False)
    z = y.detach()
    ivy.utils.assertions.check_equal(z.is_leaf, True, as_array=False)
    ivy.previous_backend()


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_is_meta(
    dtype_x,
    backend_fw,
):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.is_meta, "meta" in ivy.dev(ivy.array(data[0])), as_array=False
    )
    ivy.previous_backend()


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_is_quantized(
    dtype_x,
    backend_fw,
):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(
        x.is_quantized, "q" in ivy.dtype(ivy.array(data[0])), as_array=False
    )
    ivy.previous_backend()


# isfinite
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="isfinite",
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
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x = dtype_and_input
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# isinf
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="isinf",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_isinf(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# isnan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="isnan",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_isnan(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# isreal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="isreal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_isreal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_ivy_array(
    dtype_x,
    backend_fw,
):
    _, data = dtype_x
    ivy.set_backend(backend_fw)
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ret = helpers.flatten_and_to_np(ret=x.ivy_array.data, backend=backend_fw)
    ret_gt = helpers.flatten_and_to_np(ret=data[0], backend=backend_fw)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_gt,
        backend="torch",
    )


# lcm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="lcm",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        shared_dtype=True,
    ),
)
def test_torch_lcm(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# lcm_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="lcm_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
        shared_dtype=True,
    ),
    test_inplace=st.just(True),
)
def test_torch_lcm_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# less
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_less(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_less_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_less_equal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# less_equal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="less_equal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_less_equal_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_log(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log10
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log10",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_log10(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log10_ tests
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log10_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_log10_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log1p",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        max_value=1e37,
    ),
)
def test_torch_log1p(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# log1p_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log1p_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        max_value=1e37,
    ),
    test_inplace=st.just(True),
)
def test_torch_log1p_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# log2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_log2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log2_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log2_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_log2_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# log_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_log_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="log_softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtypes=helpers.get_dtypes("float", none=False, full=False),
)
def test_torch_log_softmax(
    *,
    dtype_x_and_axis,
    dtypes,
    on_device,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
):
    input_dtype, x, axis = dtype_x_and_axis

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "dtype": dtypes[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logaddexp
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logaddexp",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
        shared_dtype=True,
    ),
)
def test_torch_logaddexp(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logdet
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logdet",
    dtype_and_x=_get_dtype_and_matrix(square=True, batch=True),
)
def test_torch_logdet(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    dtype, x = dtype_and_x
    x = np.matmul(x.T, x) + np.identity(x.shape[0])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_and
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_and",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_logical_and(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_not
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_not",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=1
    ),
)
def test_torch_logical_not(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_not_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_not_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=1,
        large_abs_safety_factor=12,
    ),
    test_inplace=st.just(True),
)
def test_torch_logical_not_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_or
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_or",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_logical_or(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logical_xor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logical_xor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_logical_xor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# logit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logit",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        min_num_dims=1,
        min_dim_size=1,
    ),
)
def test_torch_logit(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# long
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="long",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
    ),
)
def test_torch_long(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# masked_fill
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="masked_fill",
    x_mask_val=_masked_fill_helper(),
)
def test_torch_masked_fill(
    x_mask_val,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    dtype, x, mask, val = x_mask_val
    helpers.test_frontend_method(
        init_input_dtypes=[dtype],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["bool", dtype],
        method_all_as_kwargs_np={
            "mask": mask,
            "value": val,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# matmul
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="matmul",
    dtype_tensor1_tensor2=_get_dtype_and_multiplicative_matrices(),
)
def test_torch_matmul(
    dtype_tensor1_tensor2,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    dtype, tensor1, tensor2 = dtype_tensor1_tensor2
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": tensor1,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={"other": tensor2},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# matrix_power
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="matrix_power",
    dtype_x=_get_dtype_and_matrix(square=True, invertible=True),
    n=helpers.ints(min_value=2, max_value=5),
)
def test_torch_matrix_power(
    dtype_x,
    n,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "n": n,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# max
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="max",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_max(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# maximum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="maximum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_maximum(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mean
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mean",
    dtype_and_x=_statistical_dtype_values(
        function="mean",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdims=st.booleans(),
)
def test_torch_mean(
    dtype_and_x,
    keepdims,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdims,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# median
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="median",
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    keepdim=st.booleans(),
)
def test_torch_median(
    dtype_input_axis,
    keepdim,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_input_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# min
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="min",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_min(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# minimum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="minimum",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_minimum(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mm",
    dtype_xy=_get_dtype_input_and_matrices(),
)
def test_torch_mm(
    dtype_xy,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    dtype, x, y = dtype_xy
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "mat2": y,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="movedim",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
    ),
    source=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
    destination=helpers.get_axis(
        allow_none=False,
        unique=True,
        shape=st.shared(
            helpers.get_shape(
                min_num_dims=1,
                max_num_dims=3,
                min_dim_size=1,
                max_dim_size=3,
            ),
            key="a_s_d",
        ),
        min_size=1,
        force_int=True,
    ),
)
def test_torch_movedim(
    dtype_and_input,
    source,
    destination,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, value = dtype_and_input
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": value[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "source": source,
            "destination": destination,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# msort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="msort",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=["float32", "float64", "int32", "int64"],
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_msort(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mul
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mul",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_mul(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# mul_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="mul_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    test_inplace=st.just(True),
)
def test_torch_mul_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# multiply
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="multiply",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_multiply(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# multiply_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="multiply_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_multiply_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# nanmean
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="nanmean",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_nanmean(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# nansum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="nansum",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
    ),
)
def test_torch_nansum(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="narrow",
    dtype_input_dim_start_length=_dtype_input_dim_start_length(),
)
def test_torch_narrow(
    dtype_input_dim_start_length,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    (input_dtype, x, dim, start, length) = dtype_input_dim_start_length
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "start": start,
            "length": length,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_ndim(dtype_x, backend_fw):
    ivy.set_backend(backend_fw)
    dtype, data, shape = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(x.ndim, data[0].ndim, as_array=False)
    ivy.previous_backend()


# ndimension
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ndimension",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_ndimension(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# ne
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ne",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_ne(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ne_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_ne_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# neg
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="neg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_neg(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# neg_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="neg_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_neg_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# negative
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="negative",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_negative(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new",
    dtype_and_x=helpers.dtype_and_values(),
)
def test_torch_new_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_empty (not actually intuitive for testing)
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_empty",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    size=helpers.get_shape(
        min_num_dims=1,
        max_num_dims=3,
    ),
)
def test_torch_new_empty(
    dtype_and_x,
    size,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=[ivy.int32],
        method_all_as_kwargs_np={
            "size": size,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_full
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_full",
    dtype_and_x=_fill_value_and_size(max_num_dims=3),
)
def test_torch_new_full(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={
            "size": x[1],
            "fill_value": x[2],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_ones
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_ones",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    requires_grad_and_dtypes=_requires_grad_and_dtypes(),
)
def test_torch_new_ones(
    dtype_and_x,
    size,
    requires_grad_and_dtypes,
    on_device,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    requires_grad, dtypes = requires_grad_and_dtypes
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "size": size,
            "dtype": dtypes[0],
            "requires_grad": requires_grad,
            "device": on_device,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_tensor
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_tensor",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_torch_new_tensor(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[input_dtype[1]],
        method_all_as_kwargs_np={
            "data": x[1],
            "dtype": input_dtype[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# new_zeros
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="new_zeros",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    size=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    requires_grad_and_dtypes=_requires_grad_and_dtypes(),
)
def test_torch_new_zeros(
    dtype_and_x,
    size,
    requires_grad_and_dtypes,
    on_device,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    requires_grad, dtypes = requires_grad_and_dtypes
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtypes,
        method_all_as_kwargs_np={
            "size": size,
            "dtype": dtypes[0],
            "requires_grad": requires_grad,
            "device": on_device,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# nonzero
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="nonzero",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_nonzero(
    dtype_and_values,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# norm
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="norm",
    p_dtype_x_axis=_get_axis_and_p(),
    keepdim=st.booleans(),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_norm(
    p_dtype_x_axis,
    keepdim,
    dtype,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    p, values = p_dtype_x_axis
    input_dtype, x, axis = values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "p": p,
            "dim": axis,
            "keepdim": keepdim,
            "dtype": dtype[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# normal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="normal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    mean=helpers.floats(min_value=-1, max_value=1),
    std=helpers.floats(min_value=0, max_value=1),
)
def test_torch_normal_(
    dtype_and_x,
    mean,
    std,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    dtype, x = dtype_and_x

    def call():
        return helpers.test_frontend_method(
            init_input_dtypes=dtype,
            backend_to_test=backend_fw,
            init_all_as_kwargs_np={"data": x[0]},
            method_input_dtypes=dtype,
            method_all_as_kwargs_np={
                "mean": mean,
                "std": std,
            },
            frontend_method_data=frontend_method_data,
            init_flags=init_flags,
            method_flags=method_flags,
            frontend=frontend,
            on_device=on_device,
            test_values=False,
        )

    ret = call()

    if not ivy.exists(ret):
        return

    ret_np, ret_from_np = ret
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=backend_fw)
    ret_from_np = helpers.flatten_and_to_np(ret=ret_from_np, backend=backend_fw)
    for u, v in zip(ret_np, ret_from_np):
        assert u.dtype == v.dtype
        assert u.shape == v.shape


# not_equal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="not_equal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_not_equal(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="not_equal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_not_equal_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# numpy
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="numpy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_numpy(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    ret, frontend_ret = helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
    )
    # manual testing required as function return is numpy frontend
    helpers.value_test(
        ret_np_flat=helpers.flatten_and_to_np(ret=ret, backend=backend_fw),
        ret_np_from_gt_flat=frontend_ret[0],
        ground_truth_backend="torch",
        backend=backend_fw,
    )


# permute
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="permute",
    dtype_values_axis=_array_idxes_n_dtype(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_permute(
    dtype_values_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    x, idxes, dtype = dtype_values_axis
    unpack_dims = True
    if unpack_dims:
        method_flags.num_positional_args = len(idxes) + 1
        dims = {}
        i = 0
        for x_ in idxes:
            dims[f"x{i}"] = x_
            i += 1
    else:
        dims = {
            "dims": tuple(idxes),
        }
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np=dims,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# pow
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="pow",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_pow(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[1] = ivy.abs(x[1])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "exponent": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# pow_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="pow_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_pow_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    dtype = input_dtype[0]
    if "int" in dtype:
        x[1] = ivy.abs(x[1])
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "exponent": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# prod
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="prod",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
        large_abs_safety_factor=10,
        small_abs_safety_factor=10,
        safety_factor_scale="log",
    ),
    dtype=helpers.get_dtypes("float", none=True, full=False),
    keepdims=st.booleans(),
)
def test_torch_prod(
    dtype_x_axis,
    dtype,
    keepdims,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    if ivy.current_backend_str() == "torch":
        init_flags.as_variable = [False]
        method_flags.as_variable = [False]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "keepdim": keepdims,
            "dtype": dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="quantile",
    dtype_and_x=_quantile_helper().filter(lambda x: "bfloat16" not in x[0]),
    keepdims=st.booleans(),
)
def test_torch_quantile(
    dtype_and_x,
    keepdims,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis, interpolation, q = dtype_and_x
    if type(axis) is tuple:
        axis = axis[0]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "q": q,
            "dim": axis,
            "keepdim": keepdims,
            "interpolation": interpolation[0],
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# rad2deg
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="rad2deg",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_rad2deg(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# random_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="random_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float_and_integer"),
        min_value=1,
        max_value=5,
        min_num_dims=1,
        max_num_dims=5,
    ),
    to=helpers.ints(min_value=1, max_value=100),
    test_inplace=st.just(True),
)
def test_torch_random_(
    dtype_and_x,
    to,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        method_input_dtypes=input_dtype,
        frontend_method_data=frontend_method_data,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_all_as_kwargs_np={
            "to": to,
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
    )


# ravel
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="ravel",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_ravel(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("complex", prune_function=False)
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_real(dtype_x, backend_fw):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0])
    x.ivy_array = data[0]
    ivy.utils.assertions.check_equal(x.real, ivy.real(data[0]))
    ivy.previous_backend()


# reciprocal
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reciprocal",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=1,
    ),
)
def test_torch_reciprocal(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reciprocal_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reciprocal_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=1,
    ),
    test_inplace=st.just(True),
)
def test_torch_reciprocal_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# relu
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_relu(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# remainder
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="remainder",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        shared_dtype=True,
        num_arrays=2,
    ),
)
def test_torch_remainder(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# remainder_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="remainder_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-1e04,
        max_value=1e04,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        shared_dtype=True,
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_remainder_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# repeat
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="repeat",
    dtype_x_repeats=_repeat_helper(),
    unpack_repeat=st.booleans(),
)
def test_torch_repeat(
    dtype_x_repeats,
    unpack_repeat,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, repeats = dtype_x_repeats

    if backend_fw == "paddle":
        # paddle only supports size of the shape of repeats
        # to be less than or equal to 6
        assume(len(repeats) <= 6)

    repeat = {
        "repeats": repeats,
    }
    if unpack_repeat:
        method_flags.num_positional_args = len(repeat["repeats"]) + 1
        for i, x_ in enumerate(repeat["repeats"]):
            repeat[f"x{i}"] = x_
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=repeat,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
    ),
    requires_grad=st.booleans(),
)
def test_torch_requires_grad(dtype_x, requires_grad, backend_fw):
    ivy.set_backend(backend_fw)
    _, data = dtype_x
    x = Tensor(data[0], requires_grad=requires_grad)
    ivy.utils.assertions.check_equal(x.requires_grad, requires_grad, as_array=False)
    x.requires_grad = not requires_grad
    ivy.utils.assertions.check_equal(x.requires_grad, not requires_grad, as_array=False)
    ivy.previous_backend()


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reshape",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(), key="value_shape")
    ),
    unpack_shape=st.booleans(),
)
def test_torch_reshape(
    dtype_x,
    shape,
    unpack_shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    shape = {
        "shape": shape,
    }
    if unpack_shape:
        method_flags.num_positional_args = len(shape["shape"]) + 1
        i = 0
        for x_ in shape["shape"]:
            shape[f"x{i}"] = x_
            i += 1
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=shape,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# reshape_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="reshape_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=2
    ),
)
def test_torch_reshape_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# round
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="round",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    decimals=st.integers(min_value=0, max_value=5),
)
def test_torch_round(
    dtype_and_x,
    decimals,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "decimals": decimals,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# round_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="round_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    decimals=st.integers(min_value=0, max_value=5),
    test_inplace=st.just(True),
)
def test_torch_round_(
    dtype_and_x,
    decimals,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "decimals": decimals,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# rsqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="rsqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_rsqrt(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# rsqrt_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="rsqrt_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_rsqrt_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# scatter
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="scatter",
    args=put_along_axis_helper(),
)
def test_torch_scatter(
    args,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, values, axis = args
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["int64", input_dtypes[0]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "src": values,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# scatter_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="scatter_",
    args=put_along_axis_helper(),
    reduce=st.sampled_from(["add", "multiply"]),
)
def test_torch_scatter_(
    args,
    reduce,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, values, axis = args
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x},
        method_input_dtypes=["int64", input_dtypes[0]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "src": values,
            "reduce": reduce,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# scatter_add
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="scatter_add",
    args=put_along_axis_helper(),
)
def test_torch_scatter_add(
    args,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, values, axis = args
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["int64", input_dtypes[0]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "src": values,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# scatter_add_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="scatter_add_",
    args=put_along_axis_helper(),
)
def test_torch_scatter_add_(
    args,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, values, axis = args
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["int64", input_dtypes[0]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "src": values,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# scatter_reduce
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="scatter_reduce",
    args=put_along_axis_helper(),
    mode=st.sampled_from(["sum", "prod", "amin", "amax"]),
)
def test_torch_scatter_reduce(
    args,
    mode,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, values, axis = args
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["int64", input_dtypes[0]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "src": values,
            "reduce": mode,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# scatter_reduce_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="scatter_reduce_",
    args=put_along_axis_helper(),
    mode=st.sampled_from(["sum", "prod", "amin", "amax"]),
)
def test_torch_scatter_reduce_(
    args,
    mode,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtypes, x, indices, values, axis = args
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=["int64", input_dtypes[0]],
        method_all_as_kwargs_np={
            "dim": axis,
            "index": indices,
            "src": values,
            "reduce": mode,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", prune_function=False),
        ret_shape=True,
    ).filter(lambda x: "bfloat16" not in x[0]),
)
def test_torch_shape(dtype_x, backend_fw):
    ivy.set_backend(backend_fw)
    dtype, data, shape = dtype_x
    x = Tensor(data[0])
    ivy.utils.assertions.check_equal(
        x.ivy_array.shape, ivy.Shape(shape), as_array=False
    )
    ivy.previous_backend()


# short
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="short",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_short(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sigmoid
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sigmoid",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_sigmoid(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sigmoid
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sigmoid_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_sigmoid_(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sign
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sign",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_sign(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sign_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sign_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_inplace=st.just(True),
)
def test_torch_sign_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=[input_dtype],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        frontend=frontend,
    )


# sin
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sin",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_sin(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sin_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sin_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_sin_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sinc",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_sinc(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# sinc_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sinc_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    test_inplace=st.just(True),
)
def test_torch_sinc_(
    *,
    dtype_and_x,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# sinh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sinh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_sinh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sinh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sinh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_sinh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# size
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="size",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_int=True,
    ),
)
def test_torch_size(
    dtype_and_x,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# softmax
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_softmax(
    dtype_x_and_axis,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "dtype": dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sort
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sort",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    descending=st.booleans(),
)
def test_torch_sort(
    dtype_value,
    dim,
    descending,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "descending": descending,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# split
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    split_size=_get_splits(allow_none=False, min_num_dims=1, allow_array_indices=False),
    dim=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
)
def test_torch_split(
    dtype_value,
    split_size,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "split_size": split_size,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sqrt
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sqrt",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_torch_sqrt(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sqrt_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sqrt_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    test_inplace=st.just(True),
)
def test_torch_sqrt_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# square
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="square",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_square(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# square_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="square_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        allow_inf=False,
        max_value=1e04,
        min_value=-1e04,
    ),
)
def test_torch_square_(
    dtype_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# squeeze
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="squeeze",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_squeeze(
    dtype_value_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# squeeze_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="squeeze_",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    test_inplace=st.just(True),
)
def test_torch_squeeze_(
    dtype_value_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# std
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="std",
    dtype_and_x=_statistical_dtype_values(function="std"),
)
def test_torch_std(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, _, _ = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="stride",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_torch_stride(
    dtype_value_axis,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim": axis},
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# sub
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sub",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
    alpha=st.floats(min_value=-1e04, max_value=1e04, allow_infinity=False),
)
def test_torch_sub(
    dtype_and_x,
    alpha,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
            "alpha": alpha,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        atol_=1e-02,
        on_device=on_device,
    )


# subtract_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="subtract_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
    test_inplace=st.just(True),
)
def test_torch_subtract_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtype[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# sum
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="sum",
    dtype_x_dim=_get_castable_dtype(
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdim=st.booleans(),
)
def test_torch_sum(
    dtype_x_dim,
    keepdim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, dim, castable_dtype = dtype_x_dim
    if method_flags.as_variable:
        castable_dtype = input_dtype
    input_dtype = [input_dtype]
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
            "keepdim": keepdim,
            "dtype": castable_dtype,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="svd",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: (x, x)),
    ),
    some=st.booleans(),
    compute_uv=st.booleans(),
)
def test_torch_svd(
    dtype_and_x,
    some,
    compute_uv,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    x = np.asarray(x[0], dtype=input_dtype[0])

    ret, frontend_ret = helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "some": some,
            "compute_uv": compute_uv,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
        test_values=False,
    )
    with helpers.update_backend(backend_fw) as ivy_backend:
        ret = [ivy_backend.to_numpy(x) for x in ret]
    frontend_ret = [np.asarray(x) for x in frontend_ret]

    u, s, vh = ret
    frontend_u, frontend_s, frontend_vh = frontend_ret

    if compute_uv:
        helpers.assert_all_close(
            ret_np=frontend_u @ np.diag(frontend_s) @ frontend_vh.T,
            ret_from_gt_np=u @ np.diag(s) @ vh,
            rtol=1e-2,
            atol=1e-2,
            backend=backend_fw,
            ground_truth_backend=frontend,
        )
    else:
        helpers.assert_all_close(
            ret_np=frontend_s,
            ret_from_gt_np=s,
            rtol=1e-2,
            atol=1e-2,
            backend=backend_fw,
            ground_truth_backend=frontend,
        )


# t
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="t",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=helpers.get_shape(min_num_dims=2, max_num_dims=2),
    ),
)
def test_torch_t(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="take_along_dim",
    dtype_indices_axis=helpers.array_indices_axis(
        array_dtypes=helpers.get_dtypes("numeric"),
        indices_dtypes=["int64"],
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
        indices_same_dims=True,
    ),
)
def test_torch_take_along_dim(
    dtype_indices_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, value, indices, axis, _ = dtype_indices_axis
    helpers.test_frontend_method(
        init_input_dtypes=[input_dtypes[0]],
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": value,
        },
        method_input_dtypes=[input_dtypes[1]],
        method_all_as_kwargs_np={
            "indices": indices,
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tan
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tan",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_tan(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tan_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tan_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_tan_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tanh
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tanh",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
)
def test_torch_tanh(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tanh_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tanh_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_tanh_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# corrcoef
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="corrcoef",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_torch_tensor_corrcoef(
    dtype_and_x,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# erfc_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="erfc_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_torch_tensor_erfc_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
    )


# logaddexp2
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="logaddexp2",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
        shared_dtype=True,
    ),
)
def test_torch_tensor_logaddexp2(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# negative_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="negative_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_tensor_negative_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# positive
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="positive",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1e04,
        max_value=1e04,
        allow_inf=False,
    ),
)
def test_torch_tensor_positive(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tensor_split
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tensor_split",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=1, allow_none=False, allow_array_indices=False
    ),
    dim=st.shared(
        helpers.get_axis(
            shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape"),
            force_int=True,
        ),
        key="target_axis",
    ),
    method_num_positional_args=st.just(1),
)
def test_torch_tensor_split(
    dtype_value,
    indices_or_sections,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={
            "indices_or_sections": indices_or_sections,
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tile",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    reps=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=False,
    ),
)
def test_torch_tile(
    dtype_and_values,
    reps,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, values = dtype_and_values
    if isinstance(reps, tuple):
        method_flags.num_positional_args = len(reps)
    else:
        method_flags.num_positional_args = 1
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": values[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "reps": reps,
        },
        init_flags=init_flags,
        method_flags=method_flags,
        frontend_method_data=frontend_method_data,
        frontend=frontend,
        on_device=on_device,
    )


# to
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="to",
    args_kwargs=_to_helper(),
)
def test_torch_to(
    args_kwargs,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, method_num_positional_args, method_all_as_kwargs_np = args_kwargs
    method_flags.num_positional_args = method_num_positional_args
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np=method_all_as_kwargs_np,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# topk
# TODO: add value test after the stable sorting is added to torch
# https://github.com/pytorch/pytorch/issues/88184
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="topk",
    dtype_x_axis_k=_topk_helper(),
    largest=st.booleans(),
    sorted=st.booleans(),
)
def test_torch_topk(
    dtype_x_axis_k,
    largest,
    sorted,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, input, axis, k = dtype_x_axis_k
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": input[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "k": k,
            "dim": axis,
            "largest": largest,
            "sorted": sorted,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
        test_values=False,
    )


# transpose
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="transpose",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
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
def test_torch_transpose(
    dtype_value,
    dim0,
    dim1,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"dim0": dim0, "dim1": dim1},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# transpose_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="transpose_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
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
def test_torch_transpose_(
    dtype_value,
    dim0,
    dim1,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim0": dim0,
            "dim1": dim1,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tril
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tril",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
)
def test_torch_tril(
    dtype_and_values,
    diagonal,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "diagonal": diagonal,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# tril_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="tril_",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
    test_inplace=st.just(True),
)
def test_torch_tril_(
    dtype_and_values,
    diagonal,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_values
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "diagonal": diagonal,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# triu
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="triu",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    diagonal=st.integers(
        min_value=-4,
        max_value=4,
    ),
)
def test_torch_triu(
    dtype_x,
    diagonal,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"diagonal": diagonal},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# triu_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="triu_",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
    diagonal=st.integers(
        min_value=-4,
        max_value=4,
    ),
)
def test_torch_triu_(
    dtype_x,
    diagonal,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={"diagonal": diagonal},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# true_divide_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="true_divide_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
    test_inplace=st.just(True),
)
def test_torch_true_divide_(
    dtype_and_x,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    assume(not np.any(np.isclose(x[1], 0)))

    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# trunc
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="trunc",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
)
def test_torch_trunc(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# trunc_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="trunc_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    test_inplace=st.just(True),
)
def test_torch_trunc_(
    dtype_value,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# type
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="type",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_type(
    dtype_and_x,
    dtype,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dtype": dtype[0],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# type_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="type_as",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
    ),
)
def test_torch_type_as(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unbind
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unbind",
    dtype_value_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
)
def test_torch_unbind(
    dtype_value_axis,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtypes, x, axis = dtype_value_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtypes,
        method_all_as_kwargs_np={
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unflatten",
    shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=1,
        shape_key="shape",
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_int=True,
    ),
)
def test_torch_unflatten(
    *,
    dtype_and_values,
    on_device,
    frontend,
    backend_fw,
    shape,
    axis,
    frontend_method_data,
    init_flags,
    method_flags,
):
    dtype, x = dtype_and_values
    sizes = sizes_(shape, axis)
    helpers.test_frontend_method(
        init_input_dtypes=dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "sizes": sizes,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unfold
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unfold",
    dtype_values_args=_unfold_args(),
)
def test_torch_unfold(
    dtype_values_args,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x, axis, size, step = dtype_values_args
    print(axis, size, step)
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x,
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dimension": axis,
            "size": size,
            "step": step,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# uniform_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="uniform_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=1,
        max_value=5,
        min_num_dims=1,
        max_num_dims=5,
    ),
    from_=helpers.floats(min_value=-1000, max_value=0),
    to=helpers.floats(min_value=1, max_value=1000),
    test_inplace=st.just(True),
)
def test_torch_uniform_(
    dtype_and_x,
    from_,
    to,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    method_flags.num_positional_args = 3
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "from_": from_,
            "to": to,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
    )


# unique
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unique",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        force_int_axis=True,
    ),
    sorted=st.booleans(),
    return_inverse=st.booleans(),
    return_counts=st.booleans(),
)
def test_torch_unique(
    dtype_x_axis,
    sorted,
    return_inverse,
    return_counts,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "sorted": sorted,
            "return_inverse": return_inverse,
            "return_counts": return_counts,
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unique_consecutive
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unique_consecutive",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,
        min_dim_size=2,
        force_int_axis=True,
        valid_axis=True,
    ),
    return_inverse=st.booleans(),
    return_counts=st.booleans(),
)
def test_torch_unique_consecutive(
    dtype_x_axis,
    return_inverse,
    return_counts,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "return_inverse": return_inverse,
            "return_counts": return_counts,
            "dim": axis,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unsqueeze
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unsqueeze",
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
def test_torch_unsqueeze(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# unsqueeze_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="unsqueeze_",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="shape"),
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        allow_neg=True,
        force_int=True,
    ),
    test_inplace=st.just(True),
)
def test_torch_unsqueeze_(
    dtype_value,
    dim,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": dim,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="var",
    dtype_and_x=_statistical_dtype_values(
        function="var",
        min_value=-1e04,
        max_value=1e04,
    ),
    keepdim=st.booleans(),
)
def test_torch_var(
    dtype_and_x,
    keepdim,
    frontend,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
    backend_fw,
):
    input_dtype, x, axis, correction = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={"data": x[0]},
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "dim": axis,
            "correction": int(correction),
            "keepdim": keepdim,
        },
        frontend=frontend,
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        on_device=on_device,
    )


# view
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="view",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
    ),
    shape=helpers.reshape_shapes(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="value_shape")
    ),
)
def test_torch_view(
    dtype_x,
    shape,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "size": shape,
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# view_as
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="view_as",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(), key="value_shape"),
        num_arrays=2,
    ),
)
def test_torch_view_as(
    dtype_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# vsplit
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="vsplit",
    dtype_value=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="value_shape"),
    ),
    indices_or_sections=_get_splits(
        min_num_dims=2,
        axis=0,
        allow_none=False,
        allow_array_indices=False,
        is_mod_split=True,
    ),
)
def test_torch_vsplit(
    dtype_value,
    indices_or_sections,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_value
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=[],
        method_all_as_kwargs_np={"indices_or_sections": indices_or_sections},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# where
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="where",
    broadcastables=_broadcastable_trio(),
)
def test_torch_where(
    broadcastables,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    cond, xs, dtypes = broadcastables
    helpers.test_frontend_method(
        init_input_dtypes=dtypes,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": xs[0],
        },
        method_input_dtypes=["bool", dtypes[1]],
        method_all_as_kwargs_np={
            "condition": cond,
            "other": xs[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )


# xlogy
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="xlogy",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
        shared_dtype=True,
    ),
)
def test_torch_xlogy(
    dtype_and_x,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# xlogy_
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="xlogy_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        min_value=-100,
        max_value=100,
        shared_dtype=True,
    ),
    test_inplace=st.just(True),
)
def test_torch_xlogy_(
    dtype_and_x,
    frontend,
    backend_fw,
    frontend_method_data,
    init_flags,
    method_flags,
    on_device,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={
            "other": x[1],
        },
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        backend_to_test=backend_fw,
        on_device=on_device,
    )


# zero_ tests
@handle_frontend_method(
    class_tree=CLASS_TREE,
    init_tree="torch.tensor",
    method_name="zero_",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
    ),
    test_inplace=st.just(True),
)
def test_torch_zero_(
    dtype_and_x,
    frontend_method_data,
    init_flags,
    method_flags,
    frontend,
    on_device,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        init_input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        init_all_as_kwargs_np={
            "data": x[0],
        },
        method_input_dtypes=input_dtype,
        method_all_as_kwargs_np={},
        frontend_method_data=frontend_method_data,
        init_flags=init_flags,
        method_flags=method_flags,
        frontend=frontend,
        on_device=on_device,
    )
