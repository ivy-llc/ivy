# local
import ivy
from ivy.functional.ivy.gradients import _variable
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
from ivy.functional.frontends.jax._src.tree_util import tree_leaves, tree_map
