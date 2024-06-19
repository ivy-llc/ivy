from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_bbox_generator(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        torch.tensor([5, 3]),
        torch.tensor([7, 4]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6, 8]),
        torch.tensor([10, 12]),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.bbox.bbox_generator,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_bbox_generator3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([0, 3]),
        torch.tensor([1, 4]),
        torch.tensor([2, 5]),
        torch.tensor([10, 40]),
        torch.tensor([20, 50]),
        torch.tensor([30, 60]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([5, 8]),
        torch.tensor([6, 7]),
        torch.tensor([9, 11]),
        torch.tensor([15, 20]),
        torch.tensor([25, 30]),
        torch.tensor([35, 45]),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.bbox.bbox_generator3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_bbox_to_mask(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[[1., 1.], [3., 1.], [3., 2.], [1., 2.]]]),
        5,
        5,
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[[2., 2.], [4., 2.], [4., 3.], [2., 3.]]]),
        6,
        6,
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.bbox.bbox_to_mask,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_bbox_to_mask3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[
            [1., 1., 1.], [2., 1., 1.], [2., 2., 1.], [1., 2., 1.],
            [1., 1., 2.], [2., 1., 2.], [2., 2., 2.], [1., 2., 2.]
        ]]),
        (4, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[
            [2., 2., 2.], [3., 2., 2.], [3., 3., 2.], [2., 3., 2.],
            [2., 2., 3.], [3., 2., 3.], [3., 3., 3.], [2., 3., 3.]
        ]]),
        (5, 6, 6),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.bbox.bbox_to_mask3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_infer_bbox_shape(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[
            [1., 1.], [2., 1.], [2., 2.], [1., 2.]
        ]]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[
            [2., 2.], [4., 2.], [4., 3.], [2., 3.]
        ]]),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.bbox.infer_bbox_shape,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_infer_bbox_shape3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[
            [0, 1, 2], [10, 1, 2], [10, 21, 2], [0, 21, 2],
            [0, 1, 32], [10, 1, 32], [10, 21, 32], [0, 21, 32]
        ]]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[
            [3, 4, 5], [43, 4, 5], [43, 54, 5], [3, 54, 5],
            [3, 4, 65], [43, 4, 65], [43, 54, 65], [3, 54, 65]
        ]]),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.bbox.infer_bbox_shape3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_nms(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([
            [10., 10., 20., 20.],
            [15., 5., 15., 25.],
            [100., 100., 200., 200.],
            [100., 100., 200., 200.]
        ]),
        torch.tensor([0.9, 0.8, 0.7, 0.9]),
        0.8,
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([
            [5., 5., 15., 15.],
            [10., 0., 20., 20.],
            [90., 90., 180., 180.],
            [90., 90., 180., 180.]
        ]),
        torch.tensor([0.85, 0.75, 0.65, 0.85]),
        0.7,
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.bbox.nms,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_transform_bbox(target_framework, mode, backend_compile):
    trace_args = (
        torch.eye(3).unsqueeze(0),
        torch.tensor([[
            [0., 0.], [2., 0.], [2., 2.], [0., 2.]
        ]]),
    )
    trace_kwargs = {'mode': 'xyxy'}
    test_args = (
        torch.eye(3).unsqueeze(0),
        torch.tensor([[
            [1., 1.], [3., 1.], [3., 3.], [1., 3.]
        ]]),
    )
    test_kwargs = {'mode': 'xyxy'}
    _test_function(
        kornia.geometry.bbox.transform_bbox,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
