# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.hypothesis_helpers.array_helpers import array_indices_axis
from ivy_tests.test_ivy.helpers import handle_frontend_method


# split
@handle_frontend_method(
    class_tree="numpy.ndarray",
    init_tree="numpy.array",
    method_name="__split__",
    indices_or_sections_ary_axis=array_indices_axis
)
def test_numpy_split(
    indices_or_sections_ary_axis,
    as_variable,
    dtype_and_x,
    native_array,
    num_positional_args,
):
    indices_or_sections, ary, axis=indices_or_sections_ary_axis
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
