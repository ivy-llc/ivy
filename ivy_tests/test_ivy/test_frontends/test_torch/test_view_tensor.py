# global
import pytest
import ivy

# local
from ivy.functional.frontends.torch import Tensor
from ivy_tests.test_ivy.helpers.assertions import assert_all_close


class Database:
    def __init__(self):
        self.base = Tensor([[-1, -2], [2, 1]])
        self.view_base = self.base.view((2, 2))
        self.view_view = self.view_base.view((4, 1))

    def reset(self):
        self.base = Tensor([[-1, -2], [2, 1]])


ivy.set_numpy_backend()
cache = Database()


@pytest.mark.parametrize(["test_input"], [[1], [2], [-3]])
def test_view_tensor_add(test_input):
    assert_all_close(
        ret_np=cache.view_base.add(test_input).ivy_array,
        ret_from_gt_np=cache.base.add(test_input).ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.add(test_input).ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.add(test_input).ivy_array, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[0], [-1], [-2]])
def test_view_tensor_add_(test_input):
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(
        ret_np=view_base.add_(test_input).ivy_array, ret_from_gt_np=base.ivy_array
    )
    assert_all_close(
        ret_np=view_view.add_(test_input).ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_asin():
    assert_all_close(
        ret_np=cache.view_base.asin().ivy_array,
        ret_from_gt_np=cache.base.asin().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.asin().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.asin().ivy_array, (4, 1)),
    )


def test_view_tensor_asin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.asin_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.asin_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_sin():
    assert_all_close(
        ret_np=cache.view_base.sin().ivy_array,
        ret_from_gt_np=cache.base.sin().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.sin().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.sin().ivy_array, (4, 1)),
    )


def test_view_tensor_sin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.sin_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.sin_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_sinh():
    assert_all_close(
        ret_np=cache.view_base.sinh().ivy_array,
        ret_from_gt_np=cache.base.sinh().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.sinh().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.sinh().ivy_array, (4, 1)),
    )


def test_view_tensor_sinh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.sinh_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.sinh_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_cos():
    assert_all_close(
        ret_np=cache.view_base.cos().ivy_array,
        ret_from_gt_np=cache.base.cos().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.cos().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.cos().ivy_array, (4, 1)),
    )


def test_view_tensor_cos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.cos_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.cos_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_cosh():
    assert_all_close(
        ret_np=cache.view_base.cosh().ivy_array,
        ret_from_gt_np=cache.base.cosh().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.cosh().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.cosh().ivy_array, (4, 1)),
    )


def test_view_tensor_cosh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.cosh_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.cosh_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_arcsin():
    assert_all_close(
        ret_np=cache.view_base.arcsin().ivy_array,
        ret_from_gt_np=cache.base.arcsin().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.arcsin().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.arcsin().ivy_array, (4, 1)),
    )


def test_view_tensor_arcsin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(
        ret_np=view_base.arcsin_().ivy_array, ret_from_gt_np=base.ivy_array
    )
    assert_all_close(
        ret_np=view_view.arcsin_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_atan():
    assert_all_close(
        ret_np=cache.view_base.atan().ivy_array,
        ret_from_gt_np=cache.base.atan().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.atan().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.atan().ivy_array, (4, 1)),
    )


def test_view_tensor_atan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.atan_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.atan_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_asinh():
    assert_all_close(
        ret_np=cache.view_base.asinh().ivy_array,
        ret_from_gt_np=cache.base.asinh().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.asinh().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.asinh().ivy_array, (4, 1)),
    )


def test_view_tensor_asinh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.asinh_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.asinh_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_tan():
    assert_all_close(
        ret_np=cache.view_base.tan().ivy_array,
        ret_from_gt_np=cache.base.tan().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.tan().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.tan().ivy_array, (4, 1)),
    )


def test_view_tensor_tan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.tan_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.tan_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_tanh():
    assert_all_close(
        ret_np=cache.view_base.tanh().ivy_array,
        ret_from_gt_np=cache.base.tanh().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.tanh().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.tanh().ivy_array, (4, 1)),
    )


def test_view_tensor_tanh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.tanh_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.tanh_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_atanh():
    assert_all_close(
        ret_np=cache.view_base.atanh().ivy_array,
        ret_from_gt_np=cache.base.atanh().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.atanh().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.atanh().ivy_array, (4, 1)),
    )


def test_view_tensor_atanh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.atanh_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.atanh_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_log():
    assert_all_close(
        ret_np=cache.view_base.log().ivy_array,
        ret_from_gt_np=cache.base.log().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.log().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.log().ivy_array, (4, 1)),
    )


def test_view_tensor_abs():
    assert_all_close(
        ret_np=cache.view_base.abs().ivy_array,
        ret_from_gt_np=cache.base.abs().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.abs().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.abs().ivy_array, (4, 1)),
    )


def test_view_tensor_abs_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.abs_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.abs_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_arctan():
    assert_all_close(
        ret_np=cache.view_base.arctan().ivy_array,
        ret_from_gt_np=cache.base.arctan().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.arctan().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.arctan().ivy_array, (4, 1)),
    )


def test_view_tensor_arctan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(
        ret_np=view_base.arctan_().ivy_array, ret_from_gt_np=base.ivy_array
    )
    assert_all_close(
        ret_np=view_view.arctan_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_acos():
    assert_all_close(
        ret_np=cache.view_base.acos().ivy_array,
        ret_from_gt_np=cache.base.acos().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.acos().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.acos().ivy_array, (4, 1)),
    )


def test_view_tensor_acos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.acos_().ivy_array, ret_from_gt_np=base.ivy_array)
    assert_all_close(
        ret_np=view_view.acos_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_arccos():
    assert_all_close(
        ret_np=cache.view_base.arccos().ivy_array,
        ret_from_gt_np=cache.base.arccos().ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.arccos().ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.arccos().ivy_array, (4, 1)),
    )


def test_view_tensor_arccos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(
        ret_np=view_base.arccos_().ivy_array, ret_from_gt_np=base.ivy_array
    )
    assert_all_close(
        ret_np=view_view.arccos_().ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[0], [1], [2]])
def test_view_tensor_pow(test_input):
    assert_all_close(
        ret_np=cache.view_base.pow(test_input).ivy_array,
        ret_from_gt_np=cache.base.pow(test_input).ivy_array,
    )
    assert_all_close(
        ret_np=cache.view_view.pow(test_input).ivy_array,
        ret_from_gt_np=ivy.reshape(cache.base.pow(test_input).ivy_array, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[2], [3], [4]])
def test_view_tensor_pow_(test_input):
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(
        ret_np=view_base.pow_(test_input).ivy_array, ret_from_gt_np=base.ivy_array
    )
    assert_all_close(
        ret_np=view_view.pow_(test_input).ivy_array,
        ret_from_gt_np=ivy.reshape(base.ivy_array, (4, 1)),
    )


def test_view_tensor_size():
    assert_all_close(ret_np=cache.view_base.size(), ret_from_gt_np=ivy.array([2, 2]))
    assert_all_close(ret_np=cache.view_view.size(), ret_from_gt_np=ivy.array([4, 1]))
