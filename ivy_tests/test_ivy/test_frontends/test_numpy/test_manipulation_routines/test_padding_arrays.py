# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# --- Helpers --- #
# --------------- #


@st.composite
def _pad_helper(draw):
    mode = draw(
        st.sampled_from(
            [
                "constant",
                "edge",
                "linear_ramp",
                "maximum",
                "mean",
                "median",
                "minimum",
                "reflect",
                "symmetric",
                "wrap",
            ]
        )
    )
    if mode in ["median", "mean"]:
        dtypes = "float"
    else:
        dtypes = "numeric"
    dtype, input, shape = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes(dtypes),
            ret_shape=True,
            min_num_dims=1,
            min_value=-100,
            max_value=100,
        ).filter(
            lambda x: x[0][0] not in ["float16", "bfloat16", "complex64", "complex128"]
        ),
    )
    ndim = len(shape)
    pad_width = draw(_st_tuples_or_int(ndim, min_val=0))
    kwargs = {}
    if mode in ["reflect", "symmetric"]:
        kwargs["reflect_type"] = draw(st.sampled_from(["even", "odd"]))
    if mode in ["maximum", "mean", "median", "minimum"]:
        kwargs["stat_length"] = draw(_st_tuples_or_int(ndim, min_val=2))
    if mode in ["linear_ramp"]:
        kwargs["end_values"] = draw(_st_tuples_or_int(ndim))
    if mode == "constant":
        kwargs["constant_values"] = draw(_st_tuples_or_int(ndim))
    return dtype, input[0], pad_width, kwargs, mode


def _st_tuples_or_int(n_pairs, min_val=0):
    return st.one_of(
        st_tuples(
            st.tuples(
                st.integers(min_value=min_val, max_value=4),
                st.integers(min_value=min_val, max_value=4),
            ),
            min_size=n_pairs,
            max_size=n_pairs,
        ),
        helpers.ints(min_value=min_val, max_value=4),
    )


# --- Main --- #
# ------------ #


def st_tuples(elements, *, min_size=0, max_size=None, unique_by=None, unique=False):
    return st.lists(
        elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=unique_by,
        unique=unique,
    ).map(tuple)


# pad
@handle_frontend_test(
    fn_tree="numpy.pad",
    args=_pad_helper(),
    test_with_out=st.just(False),
)
def test_numpy_pad(
    *,
    args,
    fn_tree,
    backend_fw,
    on_device,
    test_flags,
    frontend,
):
    dtype, x, pad_width, kwargs, mode = args
    helpers.test_frontend_function(
        input_dtypes=dtype,
        backend_to_test="numpy",
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        array=x,
        pad_width=pad_width,
        mode=mode,
        **kwargs,
    )
