from hypothesis import strategies as st

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.one_hot",

    # YOU SHOULD ONLY MODIFY THIS #
    helper_function_output=helpers.dtype_and_values(
        num_arrays=2,
        # ...
    ),
)
def get_helper_function_output(
        helper_function_output,
):
    print(helper_function_output)
