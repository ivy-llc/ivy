# local
from hypothesis import strategies as st

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args, given


# random
@handle_cmd_line_args
@given(
    input_dtypes=helpers.get_dtypes("integer", full=False),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.random"
    ),
    size=helpers.get_shape(allow_none=True),
)
def test_numpy_random(input_dtypes, num_positional_args, size, fw, native_array):
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="random.random",
        test_values=False,
        size=size,
    )


# multinomial
@st.composite
def _pop_size_num_samples_replace_n_probs(draw):
    prob_dtype = draw(helpers.get_dtypes("float", full=False))
    batch_size = draw(helpers.ints(min_value=1, max_value=5))
    population_size = draw(helpers.ints(min_value=1, max_value=20))
    replace = draw(st.booleans())
    if replace:
        num_samples = draw(helpers.ints(min_value=1, max_value=20))
    else:
        num_samples = draw(helpers.ints(min_value=1, max_value=population_size))
    probs = draw(
        helpers.array_values(
            dtype=prob_dtype[0],
            shape=[num_samples],
            min_value=0,
            max_value=1.0 / num_samples,
            exclude_min=True,
            safety_factor_scale="linear",
        )
    )
    return prob_dtype, batch_size, population_size, num_samples, replace, probs


@handle_cmd_line_args
@given(
    data=_pop_size_num_samples_replace_n_probs(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.multinomial"
    )
)
def test_numpy_multinomial(
    data,
    num_positional_args,
    fw,
):
    prob_dtype, batch_size, population_size, num_samples, replace, probs = data
    helpers.test_frontend_function(
        input_dtypes=prob_dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=False,
        test_values=False,
        fw=fw,
        frontend="numpy",
        fn_tree="random.multinomial",
        n=population_size,
        pvals=probs,
        size=batch_size,
    )
