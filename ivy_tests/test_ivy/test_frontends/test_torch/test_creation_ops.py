# global
from hypothesis import strategies as st, assume
import math
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.helpers.globals as test_globals
from ivy_tests.test_ivy.helpers import handle_frontend_test, BackendHandler


# Helper functions


@st.composite
def _fill_value(draw):
    with_array = draw(st.sampled_from([True, False]))
    dtype = draw(st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"))[0]
    with BackendHandler.update_backend(test_globals.CURRENT_BACKEND) as ivy_backend:
        if ivy_backend.is_uint_dtype(dtype):
            ret = draw(helpers.ints(min_value=0, max_value=5))
        elif ivy_backend.is_int_dtype(dtype):
            ret = draw(helpers.ints(min_value=-5, max_value=5))
        else:
            ret = draw(helpers.floats(min_value=-5, max_value=5))
        if with_array:
            return np.array(ret, dtype=dtype)
        else:
            return ret


@st.composite
def _start_stop_step(draw):
    start = draw(helpers.ints(min_value=0, max_value=50))
    stop = draw(helpers.ints(min_value=0, max_value=50))
    if start < stop:
        step = draw(helpers.ints(min_value=1, max_value=50))
    else:
        step = draw(helpers.ints(min_value=-50, max_value=-1))
    return start, stop, step


# full
@handle_frontend_test(
    fn_tree="torch.full",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtype=st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"),
)
def test_torch_full(
    *,
    shape,
    fill_value,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        size=shape,
        fill_value=fill_value,
        dtype=dtype[0],
        device=on_device,
    )


# ones_like
@handle_frontend_test(
    fn_tree="torch.ones_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_ones_like(
    *,
    dtype_and_x,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dtype=dtype[0],
        device=on_device,
    )


# ones
@handle_frontend_test(
    fn_tree="torch.ones",
    size=helpers.ints(min_value=1, max_value=3),
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_ones(
    *,
    shape,
    size,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dims = {}
    size = (size,)
    if shape is None:
        i = 0
        for x_ in size:
            dims[f"x{i}"] = x_
            i += 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **dims,
        size=shape,
        dtype=dtype[0],
        device=on_device,
    )


# zeros
@handle_frontend_test(
    fn_tree="torch.zeros",
    size=helpers.ints(min_value=1, max_value=3),
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_zeros(
    *,
    size,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dims = {}
    size = (size,)
    if shape is None:
        i = 0
        for x_ in size:
            dims[f"x{i}"] = x_
            i += 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        **dims,
        size=shape,
        dtype=dtype[0],
        device=on_device,
    )


# zeros_like
@handle_frontend_test(
    fn_tree="torch.zeros_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_zeros_like(
    *,
    dtype_and_x,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dtype=dtype[0],
        device=on_device,
    )


# empty
@handle_frontend_test(
    fn_tree="torch.empty",
    size=helpers.ints(min_value=1, max_value=3),
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_empty(
    *,
    size,
    shape,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dims = {}
    size = (size,)
    if shape is None:
        i = 0
        for x_ in size:
            dims[f"x{i}"] = x_
            i += 1
    helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        **dims,
        size=shape,
        dtype=dtype[0],
        test_values=False,
        device=on_device,
    )


# arange
@handle_frontend_test(
    fn_tree="torch.arange",
    start_stop_step=_start_stop_step(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_arange(
    *,
    start_stop_step,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    start, stop, step = start_stop_step
    helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        step=step,
        out=None,
        dtype=dtype[0],
        device=on_device,
    )


# range
@handle_frontend_test(
    fn_tree="torch.range",
    start_stop_step=_start_stop_step(),
    dtype=helpers.get_dtypes("float", full=False),
    number_positional_args=st.just(3),
)
def test_torch_range(
    *,
    start_stop_step,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    start, stop, step = start_stop_step
    helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        step=step,
        dtype=dtype[0],
        device=on_device,
    )


# linspace
@handle_frontend_test(
    fn_tree="torch.linspace",
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_linspace(
    *,
    start,
    stop,
    num,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        steps=num,
        dtype=dtype[0],
        device=on_device,
        rtol=1e-01,
    )


# logspace
@handle_frontend_test(
    fn_tree="torch.logspace",
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_logspace(
    *,
    start,
    stop,
    num,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        steps=num,
        dtype=dtype[0],
        device=on_device,
        rtol=1e-01,
    )


# empty_like
@handle_frontend_test(
    fn_tree="torch.empty_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_empty_like(
    *,
    dtype_and_x,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=inputs[0],
        dtype=dtype[0],
        device=on_device,
        test_values=False,
    )


# full_like
@handle_frontend_test(
    fn_tree="torch.full_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.shared(
            helpers.get_dtypes("numeric", full=False), key="dtype"
        )
    ),
    fill_value=_fill_value(),
    dtype=st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"),
)
def test_torch_full_like(
    *,
    dtype_and_x,
    fill_value,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, inputs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        input=inputs[0],
        fill_value=fill_value,
        dtype=dtype[0],
        device=on_device,
        test_values=False,
    )


# as_tensor
@handle_frontend_test(
    fn_tree="torch.as_tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_as_tensor(
    *,
    dtype_and_x,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        data=input[0],
        dtype=dtype[0],
        device=on_device,
    )


# from_numpy
@handle_frontend_test(
    fn_tree="torch.from_numpy",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_from_numpy(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test=backend_fw,
        on_device=on_device,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        data=input[0],
    )


# tensor
@handle_frontend_test(
    fn_tree="torch.tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_tensor(
    *,
    dtype_and_x,
    dtype,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        data=input[0],
        dtype=dtype[0],
        device=on_device,
    )


@st.composite
def _heaviside_helper(draw):
    input_dtype, data = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("float"),
        )
    )
    _, values = draw(
        helpers.dtype_and_values(
            available_dtypes=input_dtype,
            shape=helpers.get_shape(
                min_num_dims=1,
                max_num_dims=1,
                min_dim_size=1,
                max_dim_size=1,
            ),
        )
    )
    return input_dtype, data, values


@st.composite
def _as_strided_helper(draw):
    x_dtype, x, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_num_dims=1,
            ret_shape=True,
        )
    )
    ndim = len(shape)
    numel = x[0].size
    offset = draw(st.integers(min_value=0, max_value=numel - 1))
    numel = numel - offset
    size = draw(
        helpers.get_shape(
            min_num_dims=ndim,
            max_num_dims=ndim,
        ).filter(lambda s: math.prod(s) <= numel)
    )
    stride = draw(
        helpers.get_shape(
            min_num_dims=ndim,
            max_num_dims=ndim,
        ).filter(lambda s: all(numel // s_i >= size[i] for i, s_i in enumerate(s)))
    )
    return x_dtype, x, size, stride, offset


# as_strided
@handle_frontend_test(
    fn_tree="torch.as_strided",
    dtype_x_and_other=_as_strided_helper(),
)
def test_torch_as_strided(
    *,
    dtype_x_and_other,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    x_dtype, x, size, stride, offset = dtype_x_and_other
    try:
        helpers.test_frontend_function(
            input_dtypes=x_dtype,
            backend_to_test=backend_fw,
            frontend=frontend,
            test_flags=test_flags,
            fn_tree=fn_tree,
            on_device=on_device,
            input=x[0],
            size=size,
            stride=stride,
            storage_offset=offset,
        )
    except Exception as e:
        if hasattr(e, "message") and "out of bounds for storage of size" in e.message:
            assume(False)
        else:
            raise e


# heaviside
@handle_frontend_test(
    fn_tree="torch.heaviside",
    dtype_and_input=_heaviside_helper(),
)
def test_torch_heaviside(
    *,
    dtype_and_input,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
    frontend,
):
    input_dtype, data, values = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        frontend=frontend,
        fn_tree=fn_tree,
        input=data[0],
        values=values[0],
        on_device=on_device,
    )


# asarray
@handle_frontend_test(
    fn_tree="torch.asarray",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_asarray(
    *,
    dtype_and_x,
    dtype,
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
        obj=x[0],
        dtype=dtype[0],
        device=on_device,
    )


# from_dlpack
@handle_frontend_test(
    fn_tree="torch.from_dlpack",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_torch_from_dlpack(
    *,
    dtype_and_x,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        ext_tensor=x[0],
        backend_to_test=backend_fw,
        input_dtypes=input_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
    )


@st.composite
def _get_dtype_buffer_count_offset(draw):
    dtype, value = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
        )
    )
    value = np.array(value)
    length = value.size
    value = value.tobytes()

    offset = draw(helpers.ints(min_value=0, max_value=length - 1))
    count = draw(helpers.ints(min_value=-(2**30), max_value=length - offset))
    if count == 0:
        count = -1
    offset = offset * np.dtype(dtype[0]).itemsize

    return dtype, value, count, offset


@handle_frontend_test(
    fn_tree="torch.frombuffer",
    dtype_buffer_count_offset=_get_dtype_buffer_count_offset(),
)
def test_torch_frombuffer(
    dtype_buffer_count_offset,
    test_flags,
    frontend,
    backend_fw,
    fn_tree,
    on_device,
):
    input_dtype, buffer, count, offset = dtype_buffer_count_offset
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        buffer=buffer,
        dtype=input_dtype[0],
        count=count,
        offset=offset,
    )
