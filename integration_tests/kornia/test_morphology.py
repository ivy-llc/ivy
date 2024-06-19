from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_dilation(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.morphology.dilation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_erosion(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.morphology.erosion,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_opening(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.morphology.opening,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_closing(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.morphology.closing,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_gradient(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.morphology.gradient,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_top_hat(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.morphology.top_hat,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_bottom_hat(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.morphology.bottom_hat,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
