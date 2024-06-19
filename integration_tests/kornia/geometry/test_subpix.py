from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_conv_soft_argmax2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(20, 16, 50, 32),
    )
    trace_kwargs = {
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (1, 1),
        'temperature': torch.tensor(1.0),
        'normalized_coordinates': True,
        'eps': 1e-8,
        'output_value': True,
    }
    test_args = (
        torch.rand(10, 16, 50, 32),
    )
    test_kwargs = {
        'kernel_size': (3, 3),
        'stride': (1, 1),
        'padding': (1, 1),
        'temperature': torch.tensor(0.5),
        'normalized_coordinates': True,
        'eps': 1e-8,
        'output_value': True,
    }
    _test_function(
        kornia.geometry.subpix.conv_soft_argmax2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_conv_soft_argmax3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(20, 16, 3, 50, 32),
    )
    trace_kwargs = {
        'kernel_size': (3, 3, 3),
        'stride': (1, 1, 1),
        'padding': (1, 1, 1),
        'temperature': torch.tensor(1.0),
        'normalized_coordinates': False,
        'eps': 1e-8,
        'output_value': True,
        'strict_maxima_bonus': 0.0,
    }
    test_args = (
        torch.rand(10, 16, 5, 50, 32),
    )
    test_kwargs = {
        'kernel_size': (3, 3, 3),
        'stride': (1, 1, 1),
        'padding': (1, 1, 1),
        'temperature': torch.tensor(0.5),
        'normalized_coordinates': False,
        'eps': 1e-8,
        'output_value': True,
        'strict_maxima_bonus': 0.0,
    }
    _test_function(
        kornia.geometry.subpix.conv_soft_argmax3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_conv_quad_interp3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(20, 16, 3, 50, 32),
    )
    trace_kwargs = {
        'strict_maxima_bonus': 10.0,
        'eps': 1e-7,
    }
    test_args = (
        torch.rand(10, 16, 5, 50, 32),
    )
    test_kwargs = {
        'strict_maxima_bonus': 5.0,
        'eps': 1e-7,
    }
    _test_function(
        kornia.geometry.subpix.conv_quad_interp3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_spatial_softmax2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {
        'temperature': torch.tensor(1.0),
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
    )
    test_kwargs = {
        'temperature': torch.tensor(0.5),
    }
    _test_function(
        kornia.geometry.subpix.spatial_softmax2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_spatial_expectation2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {
        'normalized_coordinates': False,
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
    )
    test_kwargs = {
        'normalized_coordinates': False,
    }
    _test_function(
        kornia.geometry.subpix.spatial_expectation2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_spatial_soft_argmax2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {
        'temperature': torch.tensor(1.0),
        'normalized_coordinates': True,
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
    )
    test_kwargs = {
        'temperature': torch.tensor(0.5),
        'normalized_coordinates': True,
    }
    _test_function(
        kornia.geometry.subpix.spatial_soft_argmax2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_render_gaussian2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[1.0, 1.0]]),
        torch.tensor([[1.0, 1.0]]),
        (5, 5),
    )
    trace_kwargs = {
        'normalized_coordinates': False,
    }
    test_args = (
        torch.tensor([[2.0, 2.0]]),
        torch.tensor([[0.5, 0.5]]),
        (10, 10),
    )
    test_kwargs = {
        'normalized_coordinates': False,
    }
    _test_function(
        kornia.geometry.subpix.render_gaussian2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_nms2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        (3, 3),
    )
    trace_kwargs = {
        'mask_only': False,
    }
    test_args = (
        torch.rand(10, 1, 5, 5),
        (3, 3),
    )
    test_kwargs = {
        'mask_only': False,
    }
    _test_function(
        kornia.geometry.subpix.nms2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_nms3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
        (3, 3, 3),
    )
    trace_kwargs = {
        'mask_only': False,
    }
    test_args = (
        torch.rand(10, 1, 5, 5, 5),
        (3, 3, 3),
    )
    test_kwargs = {
        'mask_only': False,
    }
    _test_function(
        kornia.geometry.subpix.nms3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
