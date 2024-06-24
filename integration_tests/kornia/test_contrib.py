from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_compute_padding(target_framework, mode, backend_compile):
    trace_args = (
        (4, 3),
        (3, 3),
    )
    trace_kwargs = {}
    test_args = (
        (8, 5),
        (4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.contrib.compute_padding,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_extract_tensor_patches(target_framework, mode, backend_compile):
    trace_args = (
        torch.arange(16).view(1, 1, 4, 4),
    )
    trace_kwargs = {
        'window_size': (2, 2),
        'stride': (2, 2),
    }
    test_args = (
        torch.flip(torch.arange(32), (0,)).view(2, 1, 4, 4),
    )
    test_kwargs = {
        'window_size': (2, 2),
        'stride': (2, 2),
    }
    _test_function(
        kornia.contrib.extract_tensor_patches,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_combine_tensor_patches(target_framework, mode, backend_compile):
    trace_args = (
        kornia.contrib.extract_tensor_patches(
            torch.arange(16).view(1, 1, 4, 4),
            window_size=(2, 2),
            stride=(2, 2),
        ),
    )
    trace_kwargs = {
        'original_size': (4, 4),
        'window_size': (2, 2),
        'stride': (2, 2),
    }
    test_args = (
        kornia.contrib.extract_tensor_patches(
            torch.flip(torch.arange(32), (0,)).view(2, 1, 4, 4),
            window_size=(2, 2),
            stride=(2, 2),
        ),
    )
    test_kwargs = {
        'original_size': (4, 4),
        'window_size': (2, 2),
        'stride': (2, 2),
    }
    _test_function(
        kornia.contrib.combine_tensor_patches,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_distance_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn(1, 1, 5, 5),
    )
    trace_kwargs = {
        'kernel_size': 3,
        'h': 0.35
    }
    test_args = (
        torch.randn(5, 1, 5, 5),
    )
    test_kwargs = {
        'kernel_size': 3,
        'h': 0.5
    }
    _test_function(
        kornia.contrib.distance_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_diamond_square(target_framework, mode, backend_compile):
    trace_args = (
        (1, 1, 8, 8),
    )
    trace_kwargs = {
        'roughness': 0.5,
        'random_scale': 1.0,
        'normalize_range': (0, 1),
    }
    test_args = (
        (5, 1, 8, 8),
    )
    test_kwargs = {
        'roughness': 0.7,
        'random_scale': 0.9,
        'normalize_range': (-1, 1),
    }
    _test_function(
        kornia.contrib.diamond_square,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
