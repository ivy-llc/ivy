from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp
from random import choice
import numpy as np
import ivy
import ivy_tests.test_ivy.helpers as helpers


def _fn1(x, y):
    return ivy.matmul(x, y)


def _fn2(x, y):
    return ivy.vecdot(x, y)


def _fn3(x, y):
    ivy.add(x, y)


# @given(func=st.sampled_from([_fn1, _fn2, _fn3]),
#        shape1=st.shared(hnp.array_shapes(min_dims=2, max_dims=5, min_side=2, max_side=5),key="shape1"),
#        shape2=hnp.array_shapes(min_dims=2, max_dims=5, min_side=2, max_side=5),
#        in_axes= st.shared(helpers.get_num_axis(shape1, num=2), key="shape1"),
#        in_axes_as_cont=st.booleans())
# def test_vmap(func, shape1, shape2, in_axes_as_cont, fw):
#     num_pos_args = func.__code__.co_argcount
#     array1 = np.random.randint(0, 10, shape1)
#     array2 = np.random.randint(0, 10, shape2)
#     arrays = [ivy.to_native(ivy.array(array1)), ivy.to_native(ivy.array(array2))]
#     # in_axes = data.draw(helpers.valid_axes(len(array1.shape)))
#
#     if in_axes_as_cont:
#         dim_list = list()
#         for xi in arrays:
#             dim_list.append(len(xi.shape))
#         in_axes = np.random.randint(1, high=dim_list, size=num_pos_args).tolist()
#         vmapped_func = ivy.vmap(func, in_axes=list(in_axes), out_axes=0)
#     else:
#         vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)
#
#     assert callable(vmapped_func)
#
#     try:
#         fw_res = vmapped_func(*arrays)
#     except Exception as error:
#         print("fw Error:", error)
#         fw_res = None
#
#     ivy.set_backend("jax")
#     arrays = [ivy.array(array1), ivy.array(array2)]
#     if in_axes_as_cont:
#         dim_list = list()
#         for xi in arrays:
#             dim_list.append(len(xi.shape))
#         in_axes = np.random.randint(1, high=dim_list, size=num_pos_args).tolist()
#         jax_vmapped_func = ivy.vmap(func, in_axes=(in_axes), out_axes=0)
#     else:
#         jax_vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)
#
#     assert callable(jax_vmapped_func)
#
#     try:
#         jax_res = jax_vmapped_func(*arrays)
#     except Exception as error:
#         print("jax Error:", error)
#         jax_res = None
#
#     ivy.unset_backend()
#
#     if fw_res is not None and jax_res is not None:
#         assert ivy.array_equal(fw_res, jax_res), f"Results are not equal. fw: {fw_res}, Jax: {jax_res}"
#
#     elif fw_res is None and jax_res is None:
#         pass
#     else:
#         print("shape1:", shape1, "shape2:", shape2)
#         print("fw_res:", fw_res)
#         print("jax_res:", jax_res)
#         assert False, "One of the results is None while other isn't"
#     #del jax_res, fw_res


@given(func=st.sampled_from([_fn1, _fn2, _fn3]),
       arrays_and_axes=helpers.arrays_and_axes(allow_none=False,
                                               min_num_dims=2,
                                               max_num_dims=5,
                                               min_dim_size=2,
                                               max_dim_size=10,
                                               num=2),
       in_axes_as_cont=st.booleans())
def test_vmap2(func, arrays_and_axes, in_axes_as_cont, fw):
    num_pos_args = func.__code__.co_argcount
    generated_arrays, in_axes = arrays_and_axes
    arrays = [ivy.native_array(array) for array in generated_arrays]

    if in_axes_as_cont:
        vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(vmapped_func)

    try:
        fw_res = vmapped_func(*arrays)
    except Exception as error:
        #print("fw Error:", error)
        fw_res = None

    ivy.set_backend("jax")
    arrays = [ivy.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        jax_vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        jax_vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(jax_vmapped_func)

    try:
        jax_res = jax_vmapped_func(*arrays)
    except Exception as error:
        #print("jax Error:", error)
        jax_res = None

    ivy.unset_backend()

    if fw_res is not None and jax_res is not None:
        assert ivy.array_equal(fw_res, jax_res), f"Results are not equal. fw: {fw_res}, Jax: {jax_res}"
        print(" A HIT")

    elif fw_res is None and jax_res is None:
        pass
    else:
        # print("shape1:", shapes[0], "shape2:", shapes[0])
        # print("fw_res:", fw_res)
        # print("jax_res:", jax_res)
        assert False, "One of the results is None while other isn't"






