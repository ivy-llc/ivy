from hypothesis import strategies as st

import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="sklearn.model_selection.train_test_split",
    arrays_and_dtypes=helpers.dtype_and_values(
        num_arrays=helpers.ints(min_value=2, max_value=4),
        shape=helpers.lists(
            x=helpers.ints(min_value=2, max_value=5),
            min_size=2,
            max_size=3,
        )),
    shuffle=st.booleans(),
)
def test_sklearn_test_train_split(
    arrays_and_dtypes,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
    shuffle,
):
    dtypes, values = arrays_and_dtypes
    kw = {}
    for i, x_ in enumerate(values):
        kw["x{}".format(i)] = x_
    test_flags.num_positional_args = len(values)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
        **kw,
        shuffle=shuffle,
    )
