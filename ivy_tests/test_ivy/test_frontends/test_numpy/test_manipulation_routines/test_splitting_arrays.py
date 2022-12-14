# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers.hypothesis_helpers.array_helpers import array_indices_axis, array_dtypes
from ivy_tests.test_ivy.helpers import handle_frontend_test


# split
@handle_frontend_test(
    fn_tree="numpy.split",
    indices_or_sections_ary_axis=array_indices_axis(array_dtypes(helpers.ints(min_value=1, max_value=5)))
)
def test_numpy_split(
    fn_tree,
    indices_or_sections_ary_axis,
    as_variable,
    native_array,
    num_positional_args,
):
    indices_or_sections, ary, axis = indices_or_sections_ary_axis
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
