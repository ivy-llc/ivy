from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_bilateral_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
        0.1,
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (5, 5),
        0.2,
        (2.0, 2.0),
    )
    test_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    _test_function(
        kornia.filters.bilateral_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_blur_pool2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        3,
    )
    trace_kwargs = {'stride': 2}
    test_args = (
        torch.rand(5, 3, 8, 8),
        3,  # NOTE: changing this kernel size fails the test; also true for some of the other tests in this file
    )
    test_kwargs = {'stride': 2}
    _test_function(
        kornia.filters.blur_pool2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_box_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
    )
    trace_kwargs = {'border_type': 'reflect', 'separable': False}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (3, 3),
    )
    test_kwargs = {'border_type': 'reflect', 'separable': False}
    _test_function(
        kornia.filters.box_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_gaussian_blur2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect', 'separable': True}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (3, 3),
        (1.5, 1.5),
    )
    test_kwargs = {'border_type': 'reflect', 'separable': True}
    _test_function(
        kornia.filters.gaussian_blur2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_guided_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 3, 5, 5),
        (3, 3),
        0.1,
    )
    trace_kwargs = {'border_type': 'reflect', 'subsample': 1}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(5, 3, 5, 5),
        (3, 3),
        0.1,
    )
    test_kwargs = {'border_type': 'reflect', 'subsample': 1}
    _test_function(
        kornia.filters.guided_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_joint_bilateral_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 3, 5, 5),
        (3, 3),
        0.1,
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    test_args = (
        torch.rand(4, 3, 5, 5),
        torch.rand(4, 3, 5, 5),
        (5, 5),
        0.2,
        (2.0, 2.0),
    )
    test_kwargs = {'border_type': 'reflect', 'color_distance_type': 'l1'}
    _test_function(
        kornia.filters.joint_bilateral_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_max_blur_pool2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        3,
    )
    trace_kwargs = {'stride': 2, 'max_pool_size': 2, 'ceil_mode': False}
    test_args = (
        torch.rand(5, 3, 8, 8),
        3,
    )
    test_kwargs = {'stride': 2, 'max_pool_size': 2, 'ceil_mode': False}
    _test_function(
        kornia.filters.max_blur_pool2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_median_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.filters.median_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_motion_blur(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        5,
        45.0,
        1.0,
    )
    trace_kwargs = {'border_type': 'constant', 'mode': 'nearest'}
    test_args = (
        torch.rand(5, 3, 5, 5),
        5,
        90.0,
        0.5,
    )
    test_kwargs = {'border_type': 'constant', 'mode': 'nearest'}
    _test_function(
        kornia.filters.motion_blur,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_unsharp_mask(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        (3, 3),
        (1.5, 1.5),
    )
    trace_kwargs = {'border_type': 'reflect'}
    test_args = (
        torch.rand(5, 3, 5, 5),
        (5, 5),
        (2.0, 2.0),
    )
    test_kwargs = {'border_type': 'reflect'}
    _test_function(
        kornia.filters.unsharp_mask,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_canny(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'low_threshold': 0.1,
        'high_threshold': 0.2,
        'kernel_size': (5, 5),
        'sigma': (1, 1),
        'hysteresis': True,
        'eps': 1e-6,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'low_threshold': 0.2,
        'high_threshold': 0.3,
        'kernel_size': (5, 5),
        'sigma': (1, 1),
        'hysteresis': True,
        'eps': 1e-6,
    }
    _test_function(
        kornia.filters.canny,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_laplacian(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 4, 5, 5),
        3,
    )
    trace_kwargs = {
        'border_type': 'reflect',
        'normalized': True,
    }
    test_args = (
        torch.rand(5, 4, 5, 5),
        3,
    )
    test_kwargs = {
        'border_type': 'reflect',
        'normalized': True,
    }
    _test_function(
        kornia.filters.laplacian,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_sobel(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'normalized': True,
        'eps': 1e-6,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'normalized': True,
        'eps': 1e-5,
    }
    _test_function(
        kornia.filters.sobel,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_spatial_gradient(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'mode': 'sobel',
        'order': 1,
        'normalized': True,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'mode': 'sobel',
        'order': 1,
        'normalized': True,
    }
    _test_function(
        kornia.filters.spatial_gradient,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_spatial_gradient3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2, 4, 4),
    )
    trace_kwargs = {
        'mode': 'diff',
        'order': 1,
    }
    test_args = (
        torch.rand(5, 4, 2, 4, 4),
    )
    test_kwargs = {
        'mode': 'diff',
        'order': 1,
    }
    _test_function(
        kornia.filters.spatial_gradient3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_filter2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'padding': 'same'}
    test_args = (
        torch.rand(2, 1, 5, 5),
        torch.rand(1, 3, 3),
    )
    test_kwargs = {'padding': 'same'}
    _test_function(
        kornia.filters.filter2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_filter2d_separable(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        torch.rand(1, 3),
        torch.rand(1, 3),
    )
    trace_kwargs = {'padding': 'same'}
    test_args = (
        torch.rand(2, 1, 5, 5),
        torch.rand(1, 3),
        torch.rand(1, 3),
    )
    test_kwargs = {'padding': 'same'}
    _test_function(
        kornia.filters.filter2d_separable,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_filter3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
        torch.rand(1, 3, 3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 1, 5, 5, 5),
        torch.rand(1, 3, 3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.filters.filter3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_gaussian_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3, 2.5)
    trace_kwargs = {}
    test_args = (5, 1.5)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_gaussian_erf_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3, 2.5)
    trace_kwargs = {}
    test_args = (5, 1.5)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_erf_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_gaussian_discrete_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3, 2.5)
    trace_kwargs = {}
    test_args = (5, 1.5)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_discrete_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_gaussian_kernel2d(target_framework, mode, backend_compile):
    trace_args = ((5, 5), (1.5, 1.5))
    trace_kwargs = {}
    test_args = ((3, 5), (1.5, 1.5))
    test_kwargs = {}
    _test_function(
        kornia.filters.get_gaussian_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_hanning_kernel1d(target_framework, mode, backend_compile):
    trace_args = (4,)
    trace_kwargs = {}
    test_args = (8,)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_hanning_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_hanning_kernel2d(target_framework, mode, backend_compile):
    trace_args = ((4, 4),)
    trace_kwargs = {}
    test_args = ((8, 8),)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_hanning_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_laplacian_kernel1d(target_framework, mode, backend_compile):
    trace_args = (3,)
    trace_kwargs = {}
    test_args = (5,)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_laplacian_kernel1d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_laplacian_kernel2d(target_framework, mode, backend_compile):
    trace_args = (3,)
    trace_kwargs = {}
    test_args = (5,)
    test_kwargs = {}
    _test_function(
        kornia.filters.get_laplacian_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_motion_kernel2d(target_framework, mode, backend_compile):
    trace_args = (5, 0.0)
    trace_kwargs = {'direction': 0.0, 'mode': 'nearest'}
    test_args = (3, 215.0)
    test_kwargs = {'direction': -0.5, 'mode': 'nearest'}
    _test_function(
        kornia.filters.get_motion_kernel2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
