# global
from numpy import mgrid as np_mgrid, ogrid as np_ogrid
from hypothesis import strategies as st

import ivy

# local
from ivy.functional.frontends.numpy import mgrid, ogrid
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test, handle_frontend_method


# helpers
@st.composite
def _get_range_for_grid(draw):
    start = draw(st.booleans())
    step = draw(st.booleans())
    if start:
        start = draw(helpers.ints(min_value=-25, max_value=25))
        stop = draw(st.booleans())
        if stop:
            stop = draw(helpers.ints(min_value=30, max_value=100))
        else:
            stop = None
    else:
        start = None
        stop = draw(helpers.ints(min_value=30, max_value=100))
    if step:
        step = draw(helpers.ints(min_value=1, max_value=5))
        return start, stop, step
    return start, stop, None


@st.composite
def _get_dtype_and_range(draw):
    dim = draw(helpers.ints(min_value=2, max_value=5))
    dtype = draw(helpers.get_dtypes("float", index=1, full=False))
    start = draw(
        helpers.array_values(dtype=dtype[0], shape=(dim,), min_value=-50, max_value=0)
    )
    stop = draw(
        helpers.array_values(dtype=dtype[0], shape=(dim,), min_value=1, max_value=50)
    )
    return dtype * 2, start, stop


# arange
@handle_frontend_test(
    fn_tree="numpy.arange",
    start=helpers.ints(min_value=-50, max_value=0),
    stop=helpers.ints(min_value=1, max_value=50),
    step=helpers.ints(min_value=1, max_value=5),
    dtype=helpers.get_dtypes("float"),
    test_with_out=st.just(False),
)
def test_numpy_arange(
    start,
    stop,
    step,
    dtype,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.as_ivy_dtype("int8")],
        frontend=frontend,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        stop=stop,
        step=step,
        dtype=dtype[0],
    )


# linspace
@handle_frontend_test(
    fn_tree="numpy.linspace",
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=2, max_value=5),
    axis=helpers.ints(min_value=-1, max_value=0),
    test_with_out=st.just(False),
)
def test_numpy_linspace(
    dtype_start_stop,
    num,
    axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        stop=stop,
        num=num,
        endpoint=True,
        retstep=False,
        dtype=input_dtypes[0],
        axis=axis,
    )


# logspace
@handle_frontend_test(
    fn_tree="numpy.logspace",
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=5, max_value=50),
    base=helpers.ints(min_value=2, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=0),
    test_with_out=st.just(False),
)
def test_numpy_logspace(
    dtype_start_stop,
    num,
    base,
    axis,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-01,
        start=start,
        stop=stop,
        num=num,
        endpoint=True,
        base=base,
        dtype=input_dtypes[0],
        axis=axis,
    )


# meshgrid
@handle_frontend_test(
    fn_tree="numpy.meshgrid",
    dtype_and_arrays=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=st.integers(min_value=1, max_value=4),
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=1,
        shared_dtype=True,
    ),
    copy=st.booleans(),
    sparse=st.booleans(),
    indexing=st.sampled_from(["xy", "ij"]),
    test_with_out=st.just(False),
)
def test_numpy_meshgrid(
    *,
    dtype_and_arrays,
    copy,
    sparse,
    indexing,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, arrays = dtype_and_arrays
    kw = {}
    i = 0
    for x_ in arrays:
        kw[f"x{i}"] = x_
        i += 1
    test_flags.num_positional_args = len(arrays)
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        test_values=False,
        **kw,
        copy=copy,
        sparse=sparse,
        indexing=indexing,
    )


# mgrid
@handle_frontend_method(
    class_tree="ivy.functional.frontends.numpy.mgrid",
    init_tree="numpy.mgrid",
    method_name="__getitem__",
    range=_get_range_for_grid(),
)
def test_numpy_mgrid(
    range,
    backend_fw,
    frontend,
):
    start, stop, step = range
    ret = mgrid[start:stop:step]
    ret_np = np_mgrid[start:stop:step]
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=frontend)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_np,
        rtol=1e-03,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


# ogrid
@handle_frontend_method(
    class_tree="ivy.functional.frontends.numpy.ogrid",
    init_tree="numpy.ogrid",
    method_name="__getitem__",
    range=_get_range_for_grid(),
)
def test_numpy_ogrid(range, backend_fw, frontend):
    start, stop, step = range
    ret = ogrid[start:stop:step]
    ret_np = np_ogrid[start:stop:step]
    ret = helpers.flatten_and_to_np(ret=ret, backend=backend_fw)
    ret_np = helpers.flatten_and_to_np(ret=ret_np, backend=frontend)
    helpers.value_test(
        ret_np_flat=ret,
        ret_np_from_gt_flat=ret_np,
        rtol=1e-03,
        backend=backend_fw,
        ground_truth_backend=frontend,
    )


@handle_frontend_test(
    fn_tree="numpy.geomspace",
    dtype_start_stop=_get_dtype_and_range(),
    num=helpers.ints(min_value=5, max_value=50),
    endpoint=st.booleans(),
    test_with_out=st.just(False),
)
def test_numpy_geomspace(
    dtype_start_stop,
    num,
    endpoint,
    frontend,
    test_flags,
    fn_tree,
    backend_fw,
    on_device,
):
    input_dtypes, start, stop = dtype_start_stop
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-1,
        start=start,
        stop=stop,
        num=num,
        endpoint=endpoint,
        dtype=input_dtypes[0],
    )
