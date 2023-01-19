# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_dtype_and_matrix,
)


# eigvalsh
@handle_frontend_test(
    fn_tree="numpy.linalg.eigvalsh",
    x=_get_dtype_and_matrix(symmetric=True),
    UPLO=st.sampled_from(["L", "U"]),
)
def test_numpy_eigvalsh(
    x,
    UPLO,
    frontend,
    test_flags,
    fn_tree,
    on_device,
):
    input_dtypes, xs = x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        a=xs,
        UPLO=UPLO,
    )
