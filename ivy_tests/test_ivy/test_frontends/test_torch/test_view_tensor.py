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
        ret_np=cache.view_base.add(test_input).data,
        ret_from_gt_np=cache.base.add(test_input).data,
    )
    assert_all_close(
        ret_np=cache.view_view.add(test_input).data,
        ret_from_gt_np=ivy.reshape(cache.base.add(test_input).data, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[0], [-1], [-2]])
def test_view_tensor_add_(test_input):
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.add_(test_input).data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.add_(test_input).data,
        ret_from_gt_np=ivy.reshape(base.data, (4, 1)),
    )


def test_view_tensor_asin():
    assert_all_close(
        ret_np=cache.view_base.asin().data, ret_from_gt_np=cache.base.asin().data
    )
    assert_all_close(
        ret_np=cache.view_view.asin().data,
        ret_from_gt_np=ivy.reshape(cache.base.asin().data, (4, 1)),
    )


def test_view_tensor_asin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.asin_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.asin_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_sin():
    assert_all_close(
        ret_np=cache.view_base.sin().data, ret_from_gt_np=cache.base.sin().data
    )
    assert_all_close(
        ret_np=cache.view_view.sin().data,
        ret_from_gt_np=ivy.reshape(cache.base.sin().data, (4, 1)),
    )


def test_view_tensor_sin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.sin_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.sin_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_sinh():
    assert_all_close(
        ret_np=cache.view_base.sinh().data, ret_from_gt_np=cache.base.sinh().data
    )
    assert_all_close(
        ret_np=cache.view_view.sinh().data,
        ret_from_gt_np=ivy.reshape(cache.base.sinh().data, (4, 1)),
    )


def test_view_tensor_sinh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.sinh_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.sinh_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_cos():
    assert_all_close(
        ret_np=cache.view_base.cos().data, ret_from_gt_np=cache.base.cos().data
    )
    assert_all_close(
        ret_np=cache.view_view.cos().data,
        ret_from_gt_np=ivy.reshape(cache.base.cos().data, (4, 1)),
    )


def test_view_tensor_cos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.cos_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.cos_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_cosh():
    assert_all_close(
        ret_np=cache.view_base.cosh().data, ret_from_gt_np=cache.base.cosh().data
    )
    assert_all_close(
        ret_np=cache.view_view.cosh().data,
        ret_from_gt_np=ivy.reshape(cache.base.cosh().data, (4, 1)),
    )


def test_view_tensor_cosh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.cosh_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.cosh_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_arcsin():
    assert_all_close(
        ret_np=cache.view_base.arcsin().data, ret_from_gt_np=cache.base.arcsin().data
    )
    assert_all_close(
        ret_np=cache.view_view.arcsin().data,
        ret_from_gt_np=ivy.reshape(cache.base.arcsin().data, (4, 1)),
    )


def test_view_tensor_arcsin_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.arcsin_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.arcsin_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_atan():
    assert_all_close(
        ret_np=cache.view_base.atan().data, ret_from_gt_np=cache.base.atan().data
    )
    assert_all_close(
        ret_np=cache.view_view.atan().data,
        ret_from_gt_np=ivy.reshape(cache.base.atan().data, (4, 1)),
    )


def test_view_tensor_atan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.atan_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.atan_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_asinh():
    assert_all_close(
        ret_np=cache.view_base.asinh().data, ret_from_gt_np=cache.base.asinh().data
    )
    assert_all_close(
        ret_np=cache.view_view.asinh().data,
        ret_from_gt_np=ivy.reshape(cache.base.asinh().data, (4, 1)),
    )


def test_view_tensor_asinh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.asinh_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.asinh_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_tan():
    assert_all_close(
        ret_np=cache.view_base.tan().data, ret_from_gt_np=cache.base.tan().data
    )
    assert_all_close(
        ret_np=cache.view_view.tan().data,
        ret_from_gt_np=ivy.reshape(cache.base.tan().data, (4, 1)),
    )


def test_view_tensor_tan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.tan_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.tan_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_tanh():
    assert_all_close(
        ret_np=cache.view_base.tanh().data, ret_from_gt_np=cache.base.tanh().data
    )
    assert_all_close(
        ret_np=cache.view_view.tanh().data,
        ret_from_gt_np=ivy.reshape(cache.base.tanh().data, (4, 1)),
    )


def test_view_tensor_tanh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.tanh_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.tanh_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_atanh():
    assert_all_close(
        ret_np=cache.view_base.atanh().data, ret_from_gt_np=cache.base.atanh().data
    )
    assert_all_close(
        ret_np=cache.view_view.atanh().data,
        ret_from_gt_np=ivy.reshape(cache.base.atanh().data, (4, 1)),
    )


def test_view_tensor_atanh_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.atanh_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.atanh_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_log():
    assert_all_close(
        ret_np=cache.view_base.log().data, ret_from_gt_np=cache.base.log().data
    )
    assert_all_close(
        ret_np=cache.view_view.log().data,
        ret_from_gt_np=ivy.reshape(cache.base.log().data, (4, 1)),
    )


def test_view_tensor_abs():
    assert_all_close(
        ret_np=cache.view_base.abs().data, ret_from_gt_np=cache.base.abs().data
    )
    assert_all_close(
        ret_np=cache.view_view.abs().data,
        ret_from_gt_np=ivy.reshape(cache.base.abs().data, (4, 1)),
    )


def test_view_tensor_abs_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.abs_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.abs_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_arctan():
    assert_all_close(
        ret_np=cache.view_base.arctan().data, ret_from_gt_np=cache.base.arctan().data
    )
    assert_all_close(
        ret_np=cache.view_view.arctan().data,
        ret_from_gt_np=ivy.reshape(cache.base.arctan().data, (4, 1)),
    )


def test_view_tensor_arctan_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.arctan_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.arctan_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_acos():
    assert_all_close(
        ret_np=cache.view_base.acos().data, ret_from_gt_np=cache.base.acos().data
    )
    assert_all_close(
        ret_np=cache.view_view.acos().data,
        ret_from_gt_np=ivy.reshape(cache.base.acos().data, (4, 1)),
    )


def test_view_tensor_acos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.acos_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.acos_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


def test_view_tensor_arccos():
    assert_all_close(
        ret_np=cache.view_base.arccos().data, ret_from_gt_np=cache.base.arccos().data
    )
    assert_all_close(
        ret_np=cache.view_view.arccos().data,
        ret_from_gt_np=ivy.reshape(cache.base.arccos().data, (4, 1)),
    )


def test_view_tensor_arccos_():
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.arccos_().data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.arccos_().data, ret_from_gt_np=ivy.reshape(base.data, (4, 1))
    )


@pytest.mark.parametrize(["test_input"], [[0], [1], [2]])
def test_view_tensor_pow(test_input):
    assert_all_close(
        ret_np=cache.view_base.pow(test_input).data,
        ret_from_gt_np=cache.base.pow(test_input).data,
    )
    assert_all_close(
        ret_np=cache.view_view.pow(test_input).data,
        ret_from_gt_np=ivy.reshape(cache.base.pow(test_input).data, (4, 1)),
    )


@pytest.mark.parametrize(["test_input"], [[2], [3], [4]])
def test_view_tensor_pow_(test_input):
    base = Tensor([[-1, -2], [2, 1]])
    view_base = base.view((2, 2))
    view_view = view_base.view((4, 1))
    assert_all_close(ret_np=view_base.pow_(test_input).data, ret_from_gt_np=base.data)
    assert_all_close(
        ret_np=view_view.pow_(test_input).data,
        ret_from_gt_np=ivy.reshape(base.data, (4, 1)),
    )


def test_view_tensor_size():
    assert_all_close(ret_np=cache.view_base.size(), ret_from_gt_np=ivy.array([2, 2]))
    assert_all_close(ret_np=cache.view_view.size(), ret_from_gt_np=ivy.array([4, 1]))
