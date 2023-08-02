from hypothesis import strategies as st
from ivy_tests.test_ivy.helpers import globals as test_globals


@st.composite
def num_positional_args_from_dict(draw, backends_dict):
    parameter_info = backends_dict[test_globals.CURRENT_BACKEND]
    return draw(
        st.integers(
            min_value=parameter_info.positional_only,
            max_value=(parameter_info.total - parameter_info.keyword_only),
        )
    )
