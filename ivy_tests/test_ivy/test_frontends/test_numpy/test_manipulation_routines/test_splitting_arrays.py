# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.hypothesis_helpers.array_helpers import array_indices_axis
from ivy_tests.test_ivy.helpers import handle_frontend_test


# split
@handle_frontend_test(
    class_tree="ivy.functional.frontends.numpy.split()"
    init_tree="numpy.array",
    method_name="__split__",
)
def test_numpy_split(
    as_variable,
    dtype_and_x,
    native_array,
    num_positional_args,
):
    indices_or_sections, ary, axis = array_indices_axis
    helpers.test_frontend_function(
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        frontend="numpy",
        fn_tree="split",
        native_array_flags=native_array,
        ary=ary[0],
        axis=axis,
        indices_or_sections=indices_or_sections[0],
    )
