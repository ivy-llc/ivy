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
        ret_np=cache.view_base.add(test_input).ivyArray,
        ret_from_gt_np=cache.base.add(test_input).ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.add(test_input).ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.add(test_input).ivyArray, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[0], [-1], [-2]])
def test_view_tensor_add_(test_input):
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(
        ret_np=view_base.add_(test_input).ivyArray, ret_from_gt_np=base.ivyArray
    )
    assert_all_close(
        ret_np=view_view.add_(test_input).ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_asin():
    assert_all_close(
        ret_np=cache.view_base.asin().ivyArray,
        ret_from_gt_np=cache.base.asin().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.asin().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.asin().ivyArray, (4, 1)),
    )


def test_view_tensor_asin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.asin_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.asin_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_sin():
    assert_all_close(
        ret_np=cache.view_base.sin().ivyArray, ret_from_gt_np=cache.base.sin().ivyArray
    )
    assert_all_close(
        ret_np=cache.view_view.sin().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.sin().ivyArray, (4, 1)),
    )


def test_view_tensor_sin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.sin_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.sin_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_sinh():
    assert_all_close(
        ret_np=cache.view_base.sinh().ivyArray,
        ret_from_gt_np=cache.base.sinh().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.sinh().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.sinh().ivyArray, (4, 1)),
    )


def test_view_tensor_sinh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.sinh_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.sinh_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_cos():
    assert_all_close(
        ret_np=cache.view_base.cos().ivyArray, ret_from_gt_np=cache.base.cos().ivyArray
    )
    assert_all_close(
        ret_np=cache.view_view.cos().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.cos().ivyArray, (4, 1)),
    )


def test_view_tensor_cos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.cos_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.cos_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_cosh():
    assert_all_close(
        ret_np=cache.view_base.cosh().ivyArray,
        ret_from_gt_np=cache.base.cosh().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.cosh().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.cosh().ivyArray, (4, 1)),
    )


def test_view_tensor_cosh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.cosh_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.cosh_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_arcsin():
    assert_all_close(
        ret_np=cache.view_base.arcsin().ivyArray,
        ret_from_gt_np=cache.base.arcsin().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.arcsin().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.arcsin().ivyArray, (4, 1)),
    )


def test_view_tensor_arcsin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.arcsin_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.arcsin_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_atan():
    assert_all_close(
        ret_np=cache.view_base.atan().ivyArray,
        ret_from_gt_np=cache.base.atan().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.atan().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.atan().ivyArray, (4, 1)),
    )


def test_view_tensor_atan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.atan_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.atan_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_asinh():
    assert_all_close(
        ret_np=cache.view_base.asinh().ivyArray,
        ret_from_gt_np=cache.base.asinh().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.asinh().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.asinh().ivyArray, (4, 1)),
    )


def test_view_tensor_asinh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.asinh_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.asinh_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_tan():
    assert_all_close(
        ret_np=cache.view_base.tan().ivyArray, ret_from_gt_np=cache.base.tan().ivyArray
    )
    assert_all_close(
        ret_np=cache.view_view.tan().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.tan().ivyArray, (4, 1)),
    )


def test_view_tensor_tan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.tan_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.tan_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_tanh():
    assert_all_close(
        ret_np=cache.view_base.tanh().ivyArray,
        ret_from_gt_np=cache.base.tanh().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.tanh().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.tanh().ivyArray, (4, 1)),
    )


def test_view_tensor_tanh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.tanh_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.tanh_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_atanh():
    assert_all_close(
        ret_np=cache.view_base.atanh().ivyArray,
        ret_from_gt_np=cache.base.atanh().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.atanh().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.atanh().ivyArray, (4, 1)),
    )


def test_view_tensor_atanh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.atanh_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.atanh_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_log():
    assert_all_close(
        ret_np=cache.view_base.log().ivyArray, ret_from_gt_np=cache.base.log().ivyArray
    )
    assert_all_close(
        ret_np=cache.view_view.log().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.log().ivyArray, (4, 1)),
    )


def test_view_tensor_abs():
    assert_all_close(
        ret_np=cache.view_base.abs().ivyArray, ret_from_gt_np=cache.base.abs().ivyArray
    )
    assert_all_close(
        ret_np=cache.view_view.abs().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.abs().ivyArray, (4, 1)),
    )


def test_view_tensor_abs_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.abs_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.abs_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_arctan():
    assert_all_close(
        ret_np=cache.view_base.arctan().ivyArray,
        ret_from_gt_np=cache.base.arctan().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.arctan().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.arctan().ivyArray, (4, 1)),
    )


def test_view_tensor_arctan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.arctan_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.arctan_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_acos():
    assert_all_close(
        ret_np=cache.view_base.acos().ivyArray,
        ret_from_gt_np=cache.base.acos().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.acos().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.acos().ivyArray, (4, 1)),
    )


def test_view_tensor_acos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.acos_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.acos_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_arccos():
    assert_all_close(
        ret_np=cache.view_base.arccos().ivyArray,
        ret_from_gt_np=cache.base.arccos().ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.arccos().ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.arccos().ivyArray, (4, 1)),
    )


def test_view_tensor_arccos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.arccos_().ivyArray, ret_from_gt_np=base.ivyArray)
    assert_all_close(
        ret_np=view_view.arccos_().ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[0], [1], [2]])
def test_view_tensor_pow(test_input):
    assert_all_close(
        ret_np=cache.view_base.pow(test_input).ivyArray,
        ret_from_gt_np=cache.base.pow(test_input).ivyArray,
    )
    assert_all_close(
        ret_np=cache.view_view.pow(test_input).ivyArray,
        ret_from_gt_np=ivy.reshape(cache.base.pow(test_input).ivyArray, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[2], [3], [4]])
def test_view_tensor_pow_(test_input):
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(
        ret_np=view_base.pow_(test_input).ivyArray, ret_from_gt_np=base.ivyArray
    )
    assert_all_close(
        ret_np=view_view.pow_(test_input).ivyArray,
        ret_from_gt_np=ivy.reshape(base.ivyArray, (4, 1)),
    )


def test_view_tensor_size():
    assert_all_close(ret_np=cache.view_base.size(), ret_from_gt_np=ivy.array([2, 2]))
    assert_all_close(ret_np=cache.view_view.size(), ret_from_gt_np=ivy.array([4, 1]))
