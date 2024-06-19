from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_draw_line(target_framework, mode, backend_compile):
    trace_args = (
        torch.zeros((1, 8, 8)),
        torch.tensor([6, 4]),
        torch.tensor([1, 4]),
        torch.tensor([255]),
    )
    trace_kwargs = {}
    test_args = (
        torch.zeros((1, 8, 8)),
        torch.tensor([0, 2]),
        torch.tensor([5, 1]),
        torch.tensor([255]),
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.draw_line,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_draw_rectangle(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3, 10, 12),
        torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]]]),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 3, 10, 12),
        torch.tensor([[[0, 0, 4, 4]], [[4, 4, 10, 10]], [[2, 2, 6, 6]]]),
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.draw_rectangle,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_draw_convex_polygon(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 12, 16),
        torch.tensor([[[4, 4], [12, 4], [12, 8], [4, 8]]]),
        torch.tensor([0.5, 0.5, 0.5]),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 3, 12, 16),
        torch.tensor([[[4, 4], [12, 4], [12, 8], [4, 8]], [[3, 3], [10, 3], [10, 7], [3, 7]]]),
        torch.tensor([0.5, 0.5, 0.5]),
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.draw_convex_polygon,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_create_meshgrid(target_framework, mode, backend_compile):
    trace_args = (2, 2)
    trace_kwargs = {}
    test_args = (4, 4)
    test_kwargs = {}
    _test_function(
        kornia.utils.create_meshgrid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_create_meshgrid3d(target_framework, mode, backend_compile):
    trace_args = (2, 2, 2)
    trace_kwargs = {}
    test_args = (4, 4, 4)
    test_kwargs = {}
    _test_function(
        kornia.utils.create_meshgrid3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_one_hot(target_framework, mode, backend_compile):
    trace_args = (
        torch.LongTensor([[[0, 1], [2, 0]]]),
        3,
        torch.device('cpu'),
        torch.int64,
    )
    trace_kwargs = {}
    test_args = (
        torch.LongTensor([[[1, 2], [0, 1]]]),
        5,
        torch.device('cpu'),
        torch.int64,
    )
    test_kwargs = {}
    _test_function(
        kornia.utils.one_hot,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
