# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy.functional.frontends.numpy as np_frontend


@st.composite
def _helper_r_(draw):
    elems_in_last_dim = draw(st.integers(min_value=2, max_value=8))
    num_of_elems = draw(st.integers(min_value=1, max_value=4))
    dim = draw(st.one_of(st.just(1), st.integers(2, 4)))
    first_elem_str = draw(st.booleans())
    ret = []
    if first_elem_str:
        to_mat = draw(st.booleans())
        if to_mat:
            elem = draw(st.sampled_from(["c", "r"]))
            if dim > 2:
                dim = 2
        else:
            num = draw(st.integers(1, 3))
            elem = ""
            if num == 1:
                elem += str(draw(st.integers(-1, dim - 1)))
            elif num >= 2:
                ndmin = draw(st.integers(1, 6))
                elem += str(draw(st.integers(-1, ndmin - 1)))
                elem += "," + str(ndmin)

            if num == 3:
                elem += "," + str(draw(st.integers(-1, ndmin - 1)))
        ret.append(elem)
    if "ndmin" not in locals():
        ndmin = False

    if dim == 1:
        while num_of_elems > 0:
            num_of_elems -= 1
            elem_type = draw(st.sampled_from(["array", "slice"]))

            if elem_type == "array":
                if not ndmin:
                    shape = (draw(st.integers(1, 5)),)
                else:
                    shape = (elems_in_last_dim,)
                elem = draw(
                    helpers.array_values(
                        dtype=helpers.get_dtypes("valid"),
                        shape=shape,
                    )
                )
                if len(elem) == 1 and draw(st.booleans()):
                    elem = elem[0]
            else:
                start = draw(st.integers(min_value=-100, max_value=100))
                step = draw(st.integers(1, 3))
                if not ndmin:
                    stop = draw(st.integers(start + 1, start + 10))
                else:
                    stop = start + 1 + (elems_in_last_dim - 1) * step

                elem = slice(start, stop, step)
            ret.append(elem)
    else:
        elem_shape = draw(helpers.get_shape(min_num_dims=dim, max_num_dims=dim))
        input_dtypes, x, casting, dtype = draw(
            np_frontend_helpers.dtypes_values_casting_dtype(
                arr_func=[
                    lambda: helpers.dtype_and_values(
                        available_dtypes=helpers.get_dtypes("numeric"),
                        shape=elem_shape,
                        num_arrays=num_of_elems,
                        shared_dtype=True,
                    )
                ],
            ),
        )
        ret += x
    return ret, elems_in_last_dim, dim


@st.composite
def _helper_c_(draw):
    dim = draw(st.integers(1, 3))
    num_of_elems = draw(st.integers(1, 5))
    elem_shape = draw(helpers.get_shape(min_num_dims=dim, max_num_dims=dim))
    ret = []
    if dim == 1:
        start = draw(st.integers(min_value=-100, max_value=100))
        step = draw(st.integers(1, 3))
        stop = start + 1 + (tuple(elem_shape)[0] - 1) * step
        elem = slice(start, stop, step)
        ret.append(elem)
    input_dtypes, x, casting, dtype = draw(
        np_frontend_helpers.dtypes_values_casting_dtype(
            arr_func=[
                lambda: helpers.dtype_and_values(
                    available_dtypes=helpers.get_dtypes("numeric"),
                    shape=elem_shape,
                    num_arrays=num_of_elems,
                    shared_dtype=True,
                )
            ],
        ),
    )
    return x + ret


@handle_frontend_test(
    fn_tree="numpy.fill_diagonal",
    dtype_x_axis=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        min_dim_size=2,
        max_num_dims=2,
    ),
    val=helpers.floats(min_value=-10, max_value=10),
    wrap=helpers.get_dtypes(kind="bool"),
    test_with_out=st.just(False),
)
def test_numpy_fill_diagonal(
    dtype_x_axis,
    wrap,
    val,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    input_dtype, x = dtype_x_axis
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        on_device=on_device,
        backend_to_test=backend_fw,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        a=x[0],
        val=val,
        wrap=wrap,
    )


@handle_frontend_test(fn_tree="numpy.add", inputs=_helper_r_())  # dummy fn_tree
def test_numpy_r_(inputs):
    inputs, elems_in_last_dim, dim = inputs
    ret_gt = np.r_.__getitem__(tuple(inputs))
    ret = np_frontend.r_.__getitem__(tuple(inputs))
    if isinstance(inputs[0], str) and inputs[0] in ["r", "c"]:
        ret = ret._data
    else:
        ret = ret.ivy_array
    assert np.allclose(ret, ret_gt)


@handle_frontend_test(fn_tree="numpy.add", inputs=_helper_c_())  # dummy fn_tree
def test_numpy_c_(inputs):
    ret_gt = np.c_.__getitem__(tuple(inputs))
    ret = np_frontend.c_.__getitem__(tuple(inputs))
    if isinstance(inputs[0], str) and inputs[0] in ["r", "c"]:
        ret = ret._data
    else:
        ret = ret.ivy_array
    assert np.allclose(ret, ret_gt)
