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


