"""Tests transpiling a subset of the kornia functional api."""

from helpers import _test_function
import kornia
import torch
import pytest


def test_rgb_to_grayscale(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 4, 4),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 4, 4),)
    test_kwargs = {}
    _test_function(
        "kornia.color.rgb_to_grayscale",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_bgr_to_rgba(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        "kornia.color.bgr_to_rgba",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_rgb_to_hsv(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {"eps": 1e-8}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {"eps": 1e-2}
    _test_function(
        "kornia.color.rgb_to_hsv",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_luv_to_rgb(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {"eps": 1e-12}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {"eps": 1e-12}
    _test_function(
        "kornia.color.luv_to_rgb",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_ycbcr_to_rgb(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        "kornia.color.ycbcr_to_rgb",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_xyz_to_rgb(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        "kornia.color.xyz_to_rgb",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_sepia(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 4, 4),)
    trace_kwargs = {
        "rescale": True,
        "eps": 1e-6,
    }
    test_args = (torch.rand(5, 3, 4, 4),)
    test_kwargs = {
        "rescale": True,
        "eps": 1e-6,
    }
    _test_function(
        "kornia.color.sepia",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_combine_tensor_patches(target_framework, backend_compile):
    trace_args = (
        kornia.contrib.extract_tensor_patches(
            torch.arange(16).view(1, 1, 4, 4),
            window_size=(2, 2),
            stride=(2, 2),
        ),
    )
    trace_kwargs = {
        "original_size": (4, 4),
        "window_size": (2, 2),
        "stride": (2, 2),
    }
    test_args = (
        kornia.contrib.extract_tensor_patches(
            torch.flip(torch.arange(32), (0,)).view(2, 1, 4, 4),
            window_size=(2, 2),
            stride=(2, 2),
        ),
    )
    test_kwargs = {
        "original_size": (4, 4),
        "window_size": (2, 2),
        "stride": (2, 2),
    }
    _test_function(
        "kornia.contrib.combine_tensor_patches",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_adjust_contrast_with_mean_subtraction(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        2.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        "kornia.enhance.adjust_contrast_with_mean_subtraction",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_posterize(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        3,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        4,
    )
    test_kwargs = {}
    _test_function(
        "kornia.enhance.posterize",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e0,
    )


def test_image_histogram2d(target_framework, backend_compile):
    trace_args = (torch.rand(1, 1, 10, 10),)
    trace_kwargs = {
        "min": 0.0,
        "max": 255.0,
        "n_bins": 256,
        "bandwidth": None,
        "centers": None,
        "return_pdf": False,
        "kernel": "triangular",
        "eps": 1e-10,
    }
    test_args = (torch.rand(5, 1, 10, 10),)
    test_kwargs = {
        "min": 0.0,
        "max": 255.0,
        "n_bins": 256,
        "bandwidth": None,
        "centers": None,
        "return_pdf": False,
        "kernel": "triangular",
        "eps": 1e-10,
    }
    _test_function(
        "kornia.enhance.image_histogram2d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_normalize_min_max(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 4, 4), 0.0, 1.0)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 4, 4), -1.0, 1.0)
    test_kwargs = {}
    _test_function(
        "kornia.enhance.normalize_min_max",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_normalize_laf(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 5, 2, 3),
        torch.rand(2, 3, 64, 64),
    )
    test_kwargs = {}
    _test_function(
        "kornia.feature.normalize_laf",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_laf_to_boundary_points(target_framework, backend_compile):
    trace_args = (torch.rand(1, 5, 2, 3),)
    trace_kwargs = {"n_pts": 50}
    test_args = (torch.rand(2, 5, 2, 3),)
    test_kwargs = {"n_pts": 100}
    _test_function(
        "kornia.feature.laf_to_boundary_points",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_ellipse_to_laf(target_framework, backend_compile):
    trace_args = (torch.rand(1, 10, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(2, 10, 5),)
    test_kwargs = {}
    _test_function(
        "kornia.feature.ellipse_to_laf",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_make_upright(target_framework, backend_compile):
    trace_args = (torch.rand(1, 5, 2, 3),)
    trace_kwargs = {"eps": 1e-9}
    test_args = (torch.rand(2, 5, 2, 3),)
    test_kwargs = {"eps": 1e-6}
    _test_function(
        "kornia.feature.make_upright",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_rotate_laf(target_framework, backend_compile):
    trace_args = (
        torch.randn((1, 5, 2, 3)),
        torch.randn((1, 5, 1)),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((2, 10, 2, 3)),
        torch.randn((2, 10, 1)),
    )
    test_kwargs = {}
    _test_function(
        "kornia.feature.rotate_laf",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_laf_from_center_scale_ori(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2),
        torch.randn(1, 5, 1, 1),
        torch.randn(1, 5, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 2),
        torch.randn(5, 10, 1, 1),
        torch.randn(5, 10, 1),
    )
    test_kwargs = {}
    _test_function(
        "kornia.feature.laf_from_center_scale_ori",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_laf_to_three_points(target_framework, backend_compile):
    trace_args = (torch.rand(1, 5, 2, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(2, 10, 2, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.feature.laf_to_three_points",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_get_gaussian_kernel2d(target_framework, backend_compile):
    trace_args = ((5, 5), (1.5, 1.5))
    trace_kwargs = {}
    test_args = ((3, 5), (1.5, 1.5))
    test_kwargs = {}
    _test_function(
        "kornia.filters.get_gaussian_kernel2d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_get_hanning_kernel2d(target_framework, backend_compile):
    trace_args = ((4, 4),)
    trace_kwargs = {}
    test_args = ((8, 8),)
    test_kwargs = {}
    _test_function(
        "kornia.filters.get_hanning_kernel2d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_get_laplacian_kernel2d(target_framework, backend_compile):
    trace_args = (3,)
    trace_kwargs = {}
    test_args = (5,)
    test_kwargs = {}
    _test_function(
        "kornia.filters.get_laplacian_kernel2d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_charbonnier_loss(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {"reduction": "none"}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {"reduction": "none"}
    _test_function(
        "kornia.losses.charbonnier_loss",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_geman_mcclure_loss(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {"reduction": "none"}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {"reduction": "none"}
    _test_function(
        "kornia.losses.geman_mcclure_loss",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_confusion_matrix(target_framework, backend_compile):
    trace_args = (
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 0]),
        3,
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 2, 1]),
        3,
    )
    test_kwargs = {}
    _test_function(
        "kornia.metrics.confusion_matrix",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_mean_iou_bbox(target_framework, backend_compile):
    trace_args = (
        torch.tensor([[40, 40, 60, 60], [30, 40, 50, 60]]),
        torch.tensor([[40, 50, 60, 70], [30, 40, 40, 50]]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[20, 20, 40, 40], [10, 30, 30, 50]]),
        torch.tensor([[20, 30, 40, 50], [10, 20, 20, 30]]),
    )
    test_kwargs = {}
    _test_function(
        "kornia.metrics.mean_iou_bbox",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_dilation(target_framework, backend_compile):
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
        "kornia.morphology.dilation",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_opening(target_framework, backend_compile):
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
        "kornia.morphology.opening",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_gradient(target_framework, backend_compile):
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
        "kornia.morphology.gradient",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_top_hat(target_framework, backend_compile):
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
        "kornia.morphology.top_hat",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_create_meshgrid3d(target_framework, backend_compile):
    trace_args = (2, 2, 2)
    trace_kwargs = {}
    test_args = (4, 4, 4)
    test_kwargs = {}
    _test_function(
        "kornia.utils.create_meshgrid3d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_bbox_generator3d(target_framework, backend_compile):
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
        "kornia.geometry.bbox.bbox_generator3d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_infer_bbox_shape3d(target_framework, backend_compile):
    trace_args = (
        torch.tensor(
            [
                [
                    [0, 1, 2],
                    [10, 1, 2],
                    [10, 21, 2],
                    [0, 21, 2],
                    [0, 1, 32],
                    [10, 1, 32],
                    [10, 21, 32],
                    [0, 21, 32],
                ]
            ]
        ),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor(
            [
                [
                    [3, 4, 5],
                    [43, 4, 5],
                    [43, 54, 5],
                    [3, 54, 5],
                    [3, 4, 65],
                    [43, 4, 65],
                    [43, 54, 65],
                    [3, 54, 65],
                ]
            ]
        ),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.bbox.infer_bbox_shape3d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_undistort_image(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.eye(3)[None],
        torch.rand(1, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 3, 10, 10),
        torch.eye(3)[None],
        torch.rand(1, 4),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.calibration.undistort_image",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_distort_points(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2),
        torch.eye(3)[None],
        torch.rand(1, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 3, 2),
        torch.eye(3)[None],
        torch.rand(1, 4),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.calibration.distort_points",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_tilt_projection(target_framework, backend_compile):
    trace_args = (
        torch.tensor(0.1),
        torch.tensor(0.2),
    )
    trace_kwargs = {"return_inverse": False}
    test_args = (
        torch.tensor(0.3),
        torch.tensor(0.4),
    )
    test_kwargs = {"return_inverse": False}
    _test_function(
        "kornia.geometry.calibration.tilt_projection",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_project_points_orthographic(target_framework, backend_compile):
    trace_args = (torch.rand(2, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.camera.project_points_orthographic",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_undistort_points_kannala_brandt(target_framework, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 8),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 8),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.camera.undistort_points_kannala_brandt",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_deg2rad(target_framework, backend_compile):
    trace_args = (torch.tensor(180.0),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.deg2rad",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_cart2pol(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 3), torch.rand(1, 3, 3))
    trace_kwargs = {"eps": 1.0e-8}
    test_args = (torch.rand(5, 3, 3), torch.rand(5, 3, 3))
    test_kwargs = {"eps": 1.0e-8}
    _test_function(
        "kornia.geometry.conversions.cart2pol",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_angle_to_rotation_matrix(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.angle_to_rotation_matrix",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_convert_points_from_homogeneous(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.convert_points_from_homogeneous",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_convert_affinematrix_to_homography(target_framework, backend_compile):
    trace_args = (torch.rand(1, 2, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 2, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.convert_affinematrix_to_homography",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_denormalize_pixel_coordinates(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 2),
        64,
        64,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        128,
        128,
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.denormalize_pixel_coordinates",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_normalize_pixel_coordinates3d(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3),
        32,
        64,
        64,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
        64,
        128,
        128,
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.normalize_pixel_coordinates3d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_normalize_homography3d(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4),
        (8, 32, 32),
        (16, 64, 64),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4),
        (8, 32, 32),
        (16, 64, 64),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.normalize_homography3d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_quaternion_to_axis_angle(target_framework, backend_compile):
    trace_args = (torch.randn(1, 4),)
    trace_kwargs = {}
    test_args = (torch.randn(5, 4),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.quaternion_to_axis_angle",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_quaternion_to_rotation_matrix(target_framework, backend_compile):
    trace_args = (torch.randn(1, 4),)
    trace_kwargs = {}
    test_args = (torch.randn(5, 4),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.quaternion_to_rotation_matrix",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_quaternion_log_to_exp(target_framework, backend_compile):
    trace_args = (torch.randn(1, 3),)
    trace_kwargs = {"eps": 1.0e-12}
    test_args = (torch.randn(5, 3),)
    test_kwargs = {"eps": 1.0e-12}
    _test_function(
        "kornia.geometry.conversions.quaternion_log_to_exp",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_vector_to_skew_symmetric_matrix(target_framework, backend_compile):
    trace_args = (torch.tensor([1.0, 2.0, 3.0]),)
    trace_kwargs = {}
    test_args = (torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.vector_to_skew_symmetric_matrix",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_rotation_matrix_to_axis_angle(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.rotation_matrix_to_axis_angle",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_axis_angle_to_rotation_matrix(target_framework, backend_compile):
    trace_args = (torch.tensor([[0.0, 0.0, 0.0], [1.5708, 0.0, 0.0]]),)
    trace_kwargs = {}
    test_args = (torch.tensor([[0.0, 0.0, 0.0], [1.5708, 0.0, 0.0]]),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.axis_angle_to_rotation_matrix",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_quaternion_from_euler(target_framework, backend_compile):
    trace_args = (
        torch.tensor(0.0),
        torch.tensor(0.0),
        torch.tensor(0.0),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([0.0, 1.0, 2.0]),
        torch.tensor([2.0, 1.0, 0.0]),
        torch.tensor([1.0, 2.0, 0.0]),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.quaternion_from_euler",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_Rt_to_matrix4x4(target_framework, backend_compile):
    trace_args = (
        torch.randn(1, 3, 3),
        torch.rand(1, 3, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn(5, 3, 3),
        torch.rand(5, 3, 1),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.Rt_to_matrix4x4",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_worldtocam_to_camtoworld_Rt(target_framework, backend_compile):
    trace_args = (
        torch.randn(1, 3, 3),
        torch.rand(1, 3, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn(5, 3, 3),
        torch.rand(5, 3, 1),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.worldtocam_to_camtoworld_Rt",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_ARKitQTVecs_to_ColmapQTVecs(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 4),
        torch.rand(1, 3, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4),
        torch.rand(5, 3, 1),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.conversions.ARKitQTVecs_to_ColmapQTVecs",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_unproject_meshgrid(target_framework, backend_compile):
    trace_args = (
        4,
        4,
        torch.eye(3),
    )
    trace_kwargs = {"normalize_points": False, "device": "cpu"}
    test_args = (
        5,
        5,
        torch.eye(3),
    )
    test_kwargs = {"normalize_points": False, "device": "cpu"}
    _test_function(
        "kornia.geometry.depth.unproject_meshgrid",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_warp_frame_depth(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.rand(1, 1, 4, 4),
        torch.eye(4)[None],
        torch.eye(3)[None],
    )
    trace_kwargs = {"normalize_points": False}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(5, 1, 5, 5),
        torch.eye(4)[None].repeat(5, 1, 1),
        torch.eye(3)[None].repeat(5, 1, 1),
    )
    test_kwargs = {"normalize_points": False}
    _test_function(
        "kornia.geometry.depth.warp_frame_depth",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_essential_from_fundamental(target_framework, backend_compile):
    trace_args = (
        torch.rand(3, 3),
        torch.rand(3, 3),
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 3),
        torch.rand(3, 3),
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.epipolar.essential_from_fundamental",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_relative_camera_motion(target_framework, backend_compile):
    trace_args = (
        torch.rand(3, 3),
        torch.rand(3, 1),
        torch.rand(3, 3),
        torch.rand(3, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 3),
        torch.rand(3, 1),
        torch.rand(3, 3),
        torch.rand(3, 1),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.epipolar.relative_camera_motion",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_compute_correspond_epilines(target_framework, backend_compile):
    trace_args = (
        torch.rand(2, 8, 2),
        torch.rand(2, 3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 8, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.epipolar.compute_correspond_epilines",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_get_perpendicular(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 2, 3),
        torch.rand(1, 2, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 5, 3),
        torch.rand(2, 5, 2),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.epipolar.get_perpendicular",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_sampson_epipolar_distance(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {"squared": True, "eps": 1e-8}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {"squared": True, "eps": 1e-8}
    _test_function(
        "kornia.geometry.epipolar.sampson_epipolar_distance",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_left_to_right_epipolar_distance(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.epipolar.left_to_right_epipolar_distance",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_intrinsics_like(target_framework, backend_compile):
    trace_args = (0.5, torch.rand(1, 3, 256, 256))
    trace_kwargs = {}
    test_args = (0.8, torch.rand(2, 3, 256, 256))
    test_kwargs = {}
    _test_function(
        "kornia.geometry.epipolar.intrinsics_like",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_cross_product_matrix(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.epipolar.cross_product_matrix",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_inverse_transformation(target_framework, backend_compile):
    trace_args = (torch.rand(1, 4, 4),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 4, 4),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.linalg.inverse_transformation",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_batched_dot_product(target_framework, backend_compile):
    trace_args = (
        torch.rand(3, 5),
        torch.rand(3, 5),
    )
    trace_kwargs = {"keepdim": False}
    test_args = (
        torch.rand(5, 3, 5),
        torch.rand(5, 3, 5),
    )
    test_kwargs = {"keepdim": False}
    _test_function(
        "kornia.geometry.linalg.batched_dot_product",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_multiply_deg_two_one_poly(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 20),
        torch.rand(1, 20),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 20),
        torch.rand(3, 20),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.solvers.multiply_deg_two_one_poly",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_determinant_to_polynomial(target_framework, backend_compile):
    trace_args = (torch.rand(1, 3, 13),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 13),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.solvers.determinant_to_polynomial",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_nms3d(target_framework, backend_compile):
    if target_framework == "numpy":
        pytest.skip()  # stateful class tests are not supported

    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
        (3, 3, 3),
    )
    trace_kwargs = {
        "mask_only": False,
    }
    test_args = (
        torch.rand(10, 1, 5, 5, 5),
        (3, 3, 3),
    )
    test_kwargs = {
        "mask_only": False,
    }
    _test_function(
        "kornia.geometry.subpix.nms3d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_warp_points_tps(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
        torch.rand(1, 3, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 2),
        torch.rand(5, 10, 2),
        torch.rand(5, 10, 2),
        torch.rand(5, 3, 2),
    )
    test_kwargs = {}
    _test_function(
        "kornia.geometry.transform.warp_points_tps",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_remap(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 5, 5),
        torch.rand(1, 5, 5),
    )
    trace_kwargs = {
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": None,
        "normalized_coordinates": False,
    }
    test_args = (
        torch.rand(1, 3, 10, 10),
        torch.rand(1, 10, 10),
        torch.rand(1, 10, 10),
    )
    test_kwargs = {
        "mode": "bilinear",
        "padding_mode": "zeros",
        "align_corners": None,
        "normalized_coordinates": False,
    }
    _test_function(
        "kornia.geometry.transform.remap",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_get_rotation_matrix2d(target_framework, backend_compile):
    trace_args = (torch.rand(1, 2), 45.0 * torch.ones(1), torch.rand(1, 2))
    trace_kwargs = {}
    test_args = (torch.rand(1, 2), 90.0 * torch.ones(1), 2.0 * torch.ones(1, 2))
    test_kwargs = {}
    _test_function(
        "kornia.geometry.transform.get_rotation_matrix2d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_get_shear_matrix2d(target_framework, backend_compile):
    trace_args = (torch.tensor([[0.0, 0.0]]), torch.tensor([1.0]), torch.tensor([0.5]))
    trace_kwargs = {}
    test_args = (torch.tensor([[1.0, 1.0]]), torch.tensor([1.5]), torch.tensor([0.75]))
    test_kwargs = {}
    _test_function(
        "kornia.geometry.transform.get_shear_matrix2d",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_invert_affine_transform(target_framework, backend_compile):
    trace_args = (torch.rand(1, 2, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 2, 3),)
    test_kwargs = {}
    _test_function(
        "kornia.geometry.transform.invert_affine_transform",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )


def test_crop_by_indices(target_framework, backend_compile):
    trace_args = (
        torch.rand(1, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=torch.float32),
    )
    trace_kwargs = {"size": (40, 40), "interpolation": "bilinear"}
    test_args = (
        torch.rand(5, 3, 64, 64),
        torch.tensor(
            [[[10, 10], [50, 10], [50, 50], [10, 50]]] * 5, dtype=torch.float32
        ),
    )
    test_kwargs = {"size": (40, 40), "interpolation": "bilinear"}
    _test_function(
        "kornia.geometry.transform.crop_by_indices",
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile=backend_compile,
        tolerance=1e-3,
    )
