
#global
import pytest

#local
import ivy
from ivy_tests.test_ivy import helpers


@pytest.mark.parametrize(
    "shape", [(), (1, 2, 3), tuple([1]*10)])
@pytest.mark.parametrize(
    "dtype", ['float32'])
@pytest.mark.parametrize(
    "tensor_fn", [ivy.array, helpers.var_fn])
def test_zeros(shape, dtype, tensor_fn, dev, call):
    # smoke test
    ret = ivy.zeros(shape, dtype, dev)
    # docstring test
    helpers.assert_docstring_examples_run(ivy.zeros)