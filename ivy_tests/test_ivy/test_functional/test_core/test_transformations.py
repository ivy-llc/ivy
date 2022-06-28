from hypothesis import given, strategies as st

import ivy_tests.test_ivy.helpers as helpers
import ivy


def _fn1(x, y):
    return ivy.matmul(x, y)

def _fn2(x, y):
    return ivy.dot(x, y)

def _fn3(x, y):
    ivy.add(x, y)


@given(func=st.sampled_from([_fn1, _fn2, _fn3]))
def test_vmap(func, fw):

    vmapped_func = ivy.vmap(func)
    num_args = helpers.num_positional_args(vmapped_func)



    ivy.set_backend("jax")
    jax_vmapped_fn = ivy.vmap(func)
    assert callable(jax_vmapped_fn)
    # resuts comparisons
    ivy.unset_backend()


@given(s = st.sampled_from(['1', '2', '3']))
def test(s):
    print(s)