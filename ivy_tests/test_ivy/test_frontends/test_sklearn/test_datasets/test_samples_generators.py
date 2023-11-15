import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="sklearn.datasets.make_circles",
    n_samples=helpers.ints(min_value=1, max_value=10),
)
def test_sklearn_make_circles(
    n_samples,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        n_samples=n_samples,
        input_dtypes=["int32"],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
    )


@handle_frontend_test(
    fn_tree="sklearn.datasets.make_moons",
    n_samples=helpers.ints(min_value=1, max_value=5),
)
def test_sklearn_make_moons(
    n_samples,
    on_device,
    fn_tree,
    frontend,
    test_flags,
    backend_fw,
):
    helpers.test_frontend_function(
        n_samples=n_samples,
        input_dtypes=["int32"],
        backend_to_test=backend_fw,
        test_flags=test_flags,
        fn_tree=fn_tree,
        frontend=frontend,
        on_device=on_device,
        test_values=False,
    )
