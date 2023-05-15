# local
import ivy
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.jax._src.tree_util import tree_map
import hypothesis.strategies as st


# Define a function to square each leaf node
def square(x):
    if isinstance(x, ivy.Array):
        return ivy.square(x)
    else:
        return x**2


def leaf_strategy():
    return st.lists(st.integers(1, 10)).map(ivy.array)


def tree_strategy(max_depth=2):
    if max_depth == 0:
        return leaf_strategy()
    else:
        return st.dictionaries(
            keys=st.one_of(
                *[
                    st.text(
                        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                        min_size=1,
                        max_size=1,
                    ).filter(lambda x: x not in used_keys)
                    for used_keys in [set()]
                ]
            ),
            values=st.one_of(leaf_strategy(), tree_strategy(max_depth - 1)),
            min_size=1,
            max_size=10,
        )


@st.composite
def tree_dict_strategy(draw):
    return draw(tree_strategy())


# tree_map
@handle_frontend_test(
    fn_tree="jax._src.tree_util.tree_map",
    tree=tree_dict_strategy(),
)
def test_jax_tree_map(
    *,
    tree,
    test_flags,
    fn_tree,
    frontend,
    on_device,
):
    # Apply the square function to the tree using tree_map
    result = tree_map(square, tree)

    # compute the expected result
    expected = ivy.square(ivy.Container(tree))

    assert ivy.equal(ivy.Container(result), expected)
