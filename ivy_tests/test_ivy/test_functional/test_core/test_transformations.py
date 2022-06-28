from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
from random import choice
import numpy as np
import ivy


def _fn1(x, y):
    return ivy.matmul(x, y)

def _fn2(x, y):
    return ivy.vdot(x, y)

def _fn3(x, y):
    ivy.add(x, y)


@given(func=st.sampled_from([_fn1, _fn2, _fn3]),
       shape1=hnp.array_shapes(min_dims=2, max_dims=3, min_side=3, max_side=4),
       shape2=hnp.array_shapes(min_dims=2, max_dims=3, min_side=3, max_side=4))
def test_vmap(func,shape1, shape2, fw):
    num_pos_args = func.__code__.co_argcount
    arrays = [np.random.normal(size = choice((shape1, shape2))) for _ in range(num_pos_args)]
    print(arrays[1].shape, func.__name__)
    vmapped_func = ivy.vmap(func)
    assert callable(vmapped_func)

    try:
        other_fw_res = vmapped_func(*arrays)
    except Exception:
        other_fw_res = None

    ivy.set_backend("jax")
    jax_vmapped_func = ivy.vmap(func)
    assert callable(jax_vmapped_func)

    try:
        jax_res = jax_vmapped_func(*arrays)
    except Exception:
        jax_res =  None

    ivy.unset_backend()
    # resuts comparisons
    # try:
    #     print(fw)
    #     print(func.__name__)
    #     print(other_fw_res, jax_res)
    #     print("end1")
    # except Exception as error:
    #    pass






