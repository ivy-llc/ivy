# global
from hypothesis import settings

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# is_native_dtype
@handle_test(
    fn_tree="functional.ivy.experimental.is_native_dtype",
    input_dtype=helpers.get_dtypes("valid", full=False),
)
@settings(max_examples=200)
def test_is_native_dtype(
    input_dtype,
):
    input_dtype = input_dtype[0]
    if isinstance(input_dtype, str):
        assert ivy.is_native_dtype(input_dtype) is False

    assert ivy.is_native_dtype(ivy.as_native_dtype(input_dtype)) is True
