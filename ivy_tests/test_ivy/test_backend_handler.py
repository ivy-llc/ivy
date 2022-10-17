# global
import pytest
import importlib
# local
import ivy
from ivy.backend_handler import _backend_dict


@pytest.mark.parametrize(("backend", "array_type",),
                         [("numpy", "<class 'numpy.ndarray'>"),
                          ("tensorflow",
                           "<class 'tensorflow.python.framework.ops.EagerTensor'>"),
                          ("torch", "<class 'torch.Tensor'>"),
                          ("jax", "<class 'jaxlib.xla_extension.DeviceArray'>")])
def test_set_backend(backend, array_type):
    # recording data before backend change
    stack_before = []
    func_address_before = id(ivy.sum)
    stack_before.extend(ivy.backend_stack)

    ivy.set_backend(backend)
    stack_after = ivy.backend_stack
    # check that the function id has changed as inverse=True.
    ivy.assertions.check_equal(func_address_before,
                               id(ivy.sum),
                               inverse=True)
    # using ivy assertions to ensure the desired backend is set
    ivy.assertions.check_less(len(stack_before), len(stack_after))
    ivy.assertions.check_equal(ivy.current_backend_str(), backend)
    backend = importlib.import_module(_backend_dict[backend])
    ivy.assertions.check_equal(stack_after[-1], backend)
    x = ivy.array([1, 2, 3])
    ivy.assertions.check_equal(str(type(ivy.to_native(x))), array_type)


@pytest.mark.parametrize(("backend"), [
    'numpy', 'jax', 'tensorflow', 'torch'
])
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
    ivy.assertions.check_equal(func_address_before_unset,
                               id(ivy.sum),
                               inverse=True)
    ivy.assertions.check_equal(unset_backend,
                               importlib.import_module(_backend_dict[backend]))
    ivy.assertions.check_greater(len(stack_before_unset), len(stack_after_unset))

    # checking a previously set backend is still set
    ivy.set_backend(backend)
    ivy.set_backend('torch')
    ivy.unset_backend()
    ivy.assertions.check_equal(ivy.current_backend_str(), backend)
