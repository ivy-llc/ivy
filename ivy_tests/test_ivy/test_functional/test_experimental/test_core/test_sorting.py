# global
from hypothesis import strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# --- Helpers --- #
# --------------- #


@st.composite
def _invert_permutation_helper(draw, for_frontend_test=False):
    perm = draw(
        st.permutations(list(range(draw(st.integers(min_value=3, max_value=10)))))
    )
    if for_frontend_test or draw(st.booleans()):
        perm = np.array(perm)
    dtype = draw(
        st.sampled_from(["int32", "int64"] if not for_frontend_test else ["int64"])
    )
    return dtype, perm


# --- Main --- #
# ------------ #


# invert_permutation
@handle_test(
    fn_tree="functional.ivy.experimental.invert_permutation",
    dtype_and_perm=_invert_permutation_helper(),
    test_instance_method=st.just(False),
    test_with_out=st.just(False),
    test_gradients=st.just(False),
    ground_truth_backend="numpy",
)
def test_invert_permutation(dtype_and_perm, test_flags, backend_fw, fn_name, on_device):
    dtype, perm = dtype_and_perm
    helpers.test_function(
        input_dtypes=[dtype],
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        x=perm,
    )


# lexsort
@handle_test(
    fn_tree="functional.ivy.experimental.lexsort",
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    ground_truth_backend="numpy",
    test_with_out=st.just(False),
    test_gradients=st.just(False),
)
def test_lexsort(
    dtype_x_axis,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=input_dtype,
        test_flags=test_flags,
        on_device=on_device,
        backend_to_test=backend_fw,
        fn_name=fn_name,
        keys=x[0],
        axis=axis,
    )
