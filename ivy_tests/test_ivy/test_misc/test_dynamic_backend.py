import ivy
import pytest
import types

try:
    import tensorflow as tf
except ImportError:
    tf = types.SimpleNamespace()
try:
    import torch
except ImportError:
    torch = types.SimpleNamespace()
try:
    import jax
except ImportError:
    jax = types.SimpleNamespace()

import numpy as np


backends = ["numpy", "torch", "tensorflow", "jax"]
backend_combinations  = [(a,b) for a in backends  for b in backends if a != b]

@pytest.mark.parametrize("middle_backend,end_backend", backend_combinations)
def test_dynamic_backend_all_combos( middle_backend, end_backend):

    # create an ivy array, container and native container
    a = ivy.array([1, 2, 3])
    b = ivy.array([4, 5, 6])
    ivy_cont = ivy.Container({"w": a, "b": b})
    nativ_cont = ivy.Container({"w": tf.Variable([1, 2, 3]), "b": tf.Variable([4, 5, 6])})

    # set dynamic_backend to false for all objects
    ivy_cont.dynamic_backend = False
    nativ_cont.dynamic_backend = False
    a.dynamic_backend = False
    b.dynamic_backend = False

    # set the middle backend
    ivy.set_backend(middle_backend)

    # set dynamic_backend to true for all objects
    ivy_cont.dynamic_backend = True
    nativ_cont.dynamic_backend = True
    a.dynamic_backend = True
    b.dynamic_backend = True

    # set the final backend
    ivy.set_backend(end_backend)

    # add the necessary asserts to check if the data of the objects are in the correct format

    if end_backend == "numpy":
        assert isinstance(a.data, np.ndarray)
    elif end_backend == "torch":
        assert isinstance(a.data, torch.Tensor)
    elif end_backend == "jax":
        assert isinstance(a.data, jax.interpreters.xla.DeviceArray)
    elif end_backend == "tensorflow":
        assert isinstance(a.data, tf.Tensor)


    if end_backend == "numpy":
        assert isinstance(ivy_cont['b'].data, np.ndarray)
    elif end_backend == "torch":
        assert isinstance(ivy_cont['b'].data, torch.Tensor)
    elif end_backend == "jax":
        assert isinstance(ivy_cont['b'].data, jax.interpreters.xla.DeviceArray)
    elif end_backend == "tensorflow":
        assert isinstance(ivy_cont['b'].data, tf.Tensor)


    if end_backend == "numpy":
        assert isinstance(nativ_cont['b'].data, np.ndarray)
    elif end_backend == "jax":
        assert isinstance(nativ_cont['b'].data, jax.interpreters.xla.DeviceArray)

    if middle_backend not in ("jax", "numpy"): # these frameworks don't support native variables
        if end_backend == "torch":
            assert isinstance(nativ_cont['b'].data, torch.Tensor) and nativ_cont['b'].data.requires_grad == True
        if end_backend == "tensorflow":
            assert isinstance(nativ_cont['b'].data, tf.Variable)

    else:
        if end_backend == "torch":
            assert isinstance(nativ_cont['b'].data, torch.Tensor)
        if end_backend == "tensorflow":
            assert isinstance(nativ_cont['b'].data, tf.Tensor)

    ivy.clear_backend_stack()

def test_dynamic_backend_setter():

    a = ivy.array([1,2,3])
    type_a = type(a.data)
    a.dynamic_backend = False

    ivy.set_backend("tensorflow")
    assert type(a.data) == type_a

    a.dynamic_backend = True
    assert isinstance(a.data, tf.Tensor)

    ivy.set_backend("torch")
    assert isinstance(a.data, torch.Tensor)

    ivy.clear_backend_stack()

def test_variables():

    ivy.set_backend("tensorflow")

    a = tf.Variable(0)
    b = tf.Variable(1)

    dyn_cont = ivy.Container({"w": a, "b": b})
    stat_cont = ivy.Container({"w": a, "b": b})
    stat_cont.dynamic_backend = False

    ivy.set_backend("torch")
    assert isinstance(dyn_cont["w"].data, torch.Tensor) and \
           dyn_cont["w"].data.requires_grad == True

    assert isinstance(stat_cont["w"], tf.Variable)

    ivy.clear_backend_stack()

def test_dynamic_backend_context_manager():

    with ivy.dynamic_backend_as(True):
        a = ivy.array([0., 1.])
        b = ivy.array([2., 3.])

    with ivy.dynamic_backend_as(False):
        c = ivy.array([4., 5.])
        d = ivy.array([6., 7.])


    assert a.dynamic_backend == True
    assert b.dynamic_backend == True
    assert c.dynamic_backend == False
    assert d.dynamic_backend == False