import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from hypothesis import strategies as st


@handle_frontend_test(
    fn_tree="tensorflow.random.uniform",
    input_dtype=helpers.get_dtypes("float"),
    minval=st.integers(),
    maxval=st.integers(),
    shape=st.lists(st.tuples(st.integers(), st.integers()),
                   unique_by=(lambda x: x[0], lambda x: x[1])),
    # shape=helpers.get_shape(allow_none=False)
    seed=st.integers(min_value=0)
)
def test_tensorflow_uniform(
        *,
        input_dtype,
        num_positional_args,
        as_variable,
        native_array,
        frontend,
        fn_tree,
        on_device,
        minval,
        maxval,
        shape,
        seed
):
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        minval=minval,
        maxval=maxval,
        shape=shape,
        seed=seed,
    )
