# global
from hypothesis import strategies as st
import math

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test



# TODO: uncomment after frontend is not required
#  to be set as backend in test_frontend_function

 
@handle_frontend_test(
    fn_tree="mindspore.numpy.array",
        input_dtypes=helpers.get_dtypes("float"),
            x=st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1),
            min_size=2,
            max_size=9,
            ),
)
def test_mindspore_nn(
    x,
    frontend,
    test_flags,
    fn_tree
):
    x=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1),
        min_size=2,
        max_size=9,
    )

    helpers.test_frontend_function(
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        test_values=False,
        x=x[0]
    )