# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.jax._src.tree_util import tree_leaves, tree_map


# tree_leaves
@handle_frontend_test(
    fn_tree="jax._src.tree_util.tree_leaves",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_tree_leaves(
    *,
    dtype_and_x,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    data = {
        "a": ivy.array([1, 2, 3]),
        "b": {"c": ivy.array([4, 5, 6]), "d": ivy.array([7, 8, 9])},
    }

    # Calling implementation of tree_leaves
    leaves = tree_leaves(data)

    # Define the expected result
    expected = [ivy.array([1, 2, 3]), ivy.array([4, 5, 6]), ivy.array([7, 8, 9])]

    # Check that the result is correct
    assert ivy.all(
        ivy.equal(expected, leaves)
    ), f"Expected {expected}, but got {leaves}"


# Define a function to recursively check that the tree has been squared correctly
def assert_squared(x, y):
    if isinstance(x, ivy.Array):
        assert ivy.all(ivy.equal(y, ivy.square(x)))
    else:
        assert ivy.all(ivy.equal(y, x**2))


def tree_multimap(fn, tree, *rest):
    if not rest:
        return tree
    if isinstance(tree, dict):
        return {k: tree_multimap(fn, tree[k], *[r[k] for r in rest]) for k in tree}
    else:
        return fn(tree, *[r for r in rest])


# tree_map
@handle_frontend_test(
    fn_tree="jax._src.tree_util.tree_map",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
)
def test_jax_tree_map(
    *,
    dtype_and_x,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    # Define a sample tree
    tree = {
        "a": ivy.array([1, 2, 3]),
        "b": {"c": ivy.array([4, 5]), "d": ivy.array([6])},
    }

    # Define a function to square each leaf node
    def square(x):
        if isinstance(x, ivy.Array):
            return ivy.square(x)
        else:
            return x**2

    # Apply the square function to the tree using tree_map
    result = tree_map(square, tree)

    # Recursively check that the resulting tree has been squared correctly
    tree_multimap(assert_squared, tree, result)
