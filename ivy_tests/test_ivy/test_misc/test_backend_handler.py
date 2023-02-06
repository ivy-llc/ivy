# global
import pytest
import importlib
import types


try:
    import tensorflow as tf
except ImportError:
    tf = types.SimpleNamespace()
    tf.constant = lambda x: x
try:
    import torch
except ImportError:
    torch = types.SimpleNamespace()
    torch.tensor = lambda x: x
try:
    import jax.numpy as jnp
    import jax
except ImportError:
    jnp = types.SimpleNamespace()
    jnp.array = lambda x: x
    jax = types.SimpleNamespace()
import numpy as np

# local
import ivy
from ivy.backend_handler import _backend_dict

from ivy_tests.test_ivy.helpers.available_frameworks \
    import available_frameworks

available_frameworks_with_none = available_frameworks()[:]
available_frameworks_with_none.append(None)

available_array_types_class = [
    ("numpy", "<class 'numpy.ndarray'>"),
]

available_array_types_input = [
    ("numpy", np.array(3.0)),
]

if "tensorflow" in available_frameworks():
    available_array_types_input.append(("tensorflow", tf.constant([3.0])))
    available_array_types_class.append(
        ("tensorflow", "<class 'tensorflow.python.framework.ops.EagerTensor'>")
    )

if "jax" in available_frameworks():
    available_array_types_input.append(("jax", jnp.array(3.0)))
    available_array_types_class.append(
        ("jax", "<class 'jaxlib.xla_extension.DeviceArray'>")
    )


if "torch" in available_frameworks():
    available_array_types_input.append(("torch", torch.tensor([3.0])))
    available_array_types_class.append(("torch", "<class 'torch.Tensor'>"))


@pytest.mark.parametrize(
    (
        "backend",
        "array_type",
    ),
    available_array_types_class,
)
def test_set_backend(backend, array_type):
    # recording data before backend change
    stack_before = []
    func_address_before = id(ivy.sum)
    stack_before.extend(ivy.backend_stack)

    ivy.set_backend(backend)
    stack_after = ivy.backend_stack
    # check that the function id has changed as inverse=True.
    ivy.assertions.check_equal(func_address_before, id(ivy.sum), inverse=True)
    # using ivy assertions to ensure the desired backend is set
    ivy.assertions.check_less(len(stack_before), len(stack_after))
    ivy.assertions.check_equal(ivy.current_backend_str(), backend)
    backend = importlib.import_module(_backend_dict[backend])
    ivy.assertions.check_equal(stack_after[-1], backend)
    x = ivy.array([1, 2, 3])
    ivy.assertions.check_equal(str(type(ivy.to_native(x))), array_type)


@pytest.mark.parametrize(("backend"), available_frameworks())
def test_unset_backend(backend):

    if not ivy.backend_stack:
        assert ivy.unset_backend() is None

    ivy.set_backend(backend)
    stack_before_unset = []
    func_address_before_unset = id(ivy.sum)
    stack_before_unset.extend(ivy.backend_stack)

    unset_backend = ivy.unset_backend()
    stack_after_unset = ivy.backend_stack
    # check that the function id has changed as inverse=True.
    ivy.assertions.check_equal(
        func_address_before_unset,
        id(ivy.sum),
        inverse=True
    )
    ivy.assertions.check_equal(
        unset_backend, importlib.import_module(_backend_dict[backend])
    )
    ivy.assertions.check_greater(
        len(stack_before_unset),
        len(stack_after_unset)
    )

    # checking a previously set backend is still set
    ivy.set_backend(backend)
    ivy.set_backend("numpy")
    ivy.unset_backend()
    ivy.assertions.check_equal(ivy.current_backend_str(), backend)


def test_clear_backend_stack():
    for backend_str in available_frameworks():
        ivy.set_backend(backend_str)

    ivy.clear_backend_stack()
    ivy.assertions.check_equal(ivy.backend_stack, [])


@pytest.mark.parametrize(
    ("backend", "array_type"),
    available_array_types_input,
)
def test_current_backend(backend, array_type):
    # test backend inference from arguments when stack clear
    ivy.clear_backend_stack()
    assert ivy.current_backend(array_type) is importlib.import_module(
        _backend_dict[backend]
    )

    # global_backend > argument's backend.
    if "torch" in available_frameworks():
        ivy.set_backend("torch")
        ivy.assertions.check_equal(
            ivy.current_backend(array_type),
            importlib.import_module(_backend_dict["torch"]),
        )
    else:
        ivy.set_backend("numpy")
        ivy.assertions.check_equal(
            ivy.current_backend(array_type),
            importlib.import_module(_backend_dict["numpy"]),
        )


@pytest.mark.parametrize(("excluded"), available_frameworks_with_none)
def test_choose_random_backend(excluded):
    backend = ivy.choose_random_backend(excluded=excluded)
    if excluded is None:
        assert backend in list(_backend_dict.keys())
    else:
        backends_list = list(_backend_dict.keys())
        backends_list.remove(excluded)
        assert backend in backends_list


@pytest.mark.parametrize("backend", available_frameworks())
def test_get_backend(backend):
    imported_backend = importlib.import_module(_backend_dict[backend])

    # checking whether the updating of __dict__ works
    assert "pi" not in imported_backend.__dict__
    ivy.get_backend(backend)
    assert "pi" in imported_backend.__dict__

    # checking whether the backend is returned correctly
    ivy.assertions.check_equal(ivy.get_backend(backend), imported_backend)


# Dynamic Backend

backends = ["numpy", "torch", "tensorflow", "jax"]
backend_combinations = [(a, b) for a in backends for b in backends if a != b]


@pytest.mark.parametrize("middle_backend,end_backend", backend_combinations)
def test_dynamic_backend_all_combos(middle_backend, end_backend):

    # create an ivy array, container and native container
    a = ivy.array([1, 2, 3])
    b = ivy.array([4, 5, 6])
    ivy_cont = ivy.Container({"w": a, "b": b})
    nativ_cont = ivy.Container(
        {"w": tf.Variable([1, 2, 3]), "b": tf.Variable([4, 5, 6])}
    )

    # clear the backend stack after initialization of inputs
    ivy.clear_backend_stack()

    # set dynamic_backend to false for all objects
    ivy_cont.dynamic_backend = False
    nativ_cont.dynamic_backend = False
    a.dynamic_backend = False
    b.dynamic_backend = False

    # set the middle backend
    ivy.set_backend(middle_backend, dynamic=True)

    # set dynamic_backend to true for all objects
    ivy_cont.dynamic_backend = True
    nativ_cont.dynamic_backend = True
    a.dynamic_backend = True
    b.dynamic_backend = True

    # set the final backend
    ivy.set_backend(end_backend, dynamic=True)

    # add the necessary asserts to check if the data
    # of the objects are in the correct format

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
        assert isinstance(
            nativ_cont['b'].data,
            jax.interpreters.xla
        )

    if middle_backend not in ("jax", "numpy"):
        # these frameworks don't support native variables
        if end_backend == "torch":
            assert isinstance(nativ_cont['b'].data, torch.Tensor) \
                   and nativ_cont['b'].data.requires_grad is True
        if end_backend == "tensorflow":
            assert isinstance(nativ_cont['b'].data, tf.Variable)

    else:
        if end_backend == "torch":
            assert isinstance(nativ_cont['b'].data, torch.Tensor)
        if end_backend == "tensorflow":
            assert isinstance(nativ_cont['b'].data, tf.Tensor)


def test_dynamic_backend_setter():

    a = ivy.array([1, 2, 3])
    type_a = type(a.data)
    a.dynamic_backend = False

    # clear the backend stack after initialization of inputs
    ivy.clear_backend_stack()

    ivy.set_backend("tensorflow", dynamic=True)
    assert type(a.data) == type_a

    a.dynamic_backend = True
    assert isinstance(a.data, tf.Tensor)

    ivy.set_backend("torch", dynamic=True)
    assert isinstance(a.data, torch.Tensor)


def test_variables():

    # clear the backend stack
    ivy.clear_backend_stack()

    ivy.set_backend("tensorflow", dynamic=True)

    a = tf.Variable(0)
    b = tf.Variable(1)

    dyn_cont = ivy.Container({"w": a, "b": b})
    stat_cont = ivy.Container({"w": a, "b": b})
    stat_cont.dynamic_backend = False

    ivy.set_backend("torch", dynamic=True)
    assert isinstance(dyn_cont["w"].data, torch.Tensor) and \
           dyn_cont["w"].data.requires_grad is True

    assert isinstance(stat_cont["w"], tf.Variable)


def test_dynamic_backend_context_manager():

    with ivy.dynamic_backend_as(True):
        a = ivy.array([0., 1.])
        b = ivy.array([2., 3.])

    with ivy.dynamic_backend_as(False):
        c = ivy.array([4., 5.])
        d = ivy.array([6., 7.])

    assert a.dynamic_backend is True
    assert b.dynamic_backend is True
    assert c.dynamic_backend is False
    assert d.dynamic_backend is False
