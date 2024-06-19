from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_gftt_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 7, 7),
    )
    trace_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    test_args = (
        torch.rand(5, 1, 7, 7),
    )
    test_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    _test_function(
        kornia.feature.gftt_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_harris_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 7, 7),
    )
    trace_kwargs = {'k': 0.04, 'grads_mode': 'sobel', 'sigmas': None}
    test_args = (
        torch.rand(5, 1, 7, 7),
    )
    test_kwargs = {'k': 0.04, 'grads_mode': 'sobel', 'sigmas': None}
    _test_function(
        kornia.feature.harris_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_hessian_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 7, 7),
    )
    trace_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    test_args = (
        torch.rand(5, 1, 7, 7),
    )
    test_kwargs = {'grads_mode': 'sobel', 'sigmas': None}
    _test_function(
        kornia.feature.hessian_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_dog_response(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 5, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.dog_response,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_dog_response_single(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
    )
    trace_kwargs = {'sigma1': 1.0, 'sigma2': 1.6}
    test_args = (
        torch.rand(5, 1, 5, 5),
    )
    test_kwargs = {'sigma1': 0.5, 'sigma2': 1.2}
    _test_function(
        kornia.feature.dog_response_single,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-3,
        mode=mode,
    )


def test_get_laf_descriptors(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 32, 32),
        torch.rand(1, 3, 2, 2),
        kornia.feature.HardNet8(True),
    )
    trace_kwargs = {'patch_size': 32, 'grayscale_descriptor': True}
    test_args = (
        torch.rand(5, 1, 32, 32),
        torch.rand(5, 3, 2, 2),
        kornia.feature.HardNet8(True),
    )
    test_kwargs = {'patch_size': 32, 'grayscale_descriptor': True}
    _test_function(
        kornia.feature.get_laf_descriptors,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_match_nn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.match_nn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_match_mnn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.match_mnn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_match_snn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {'th': 0.8}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {'th': 0.8}
    _test_function(
        kornia.feature.match_snn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_match_smnn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 128),
        torch.rand(1, 128),
    )
    trace_kwargs = {'th': 0.95}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
    )
    test_kwargs = {'th': 0.95}
    _test_function(
        kornia.feature.match_smnn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_match_fginn(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 128),
        torch.rand(3, 128),
        torch.rand(1, 3, 2, 3),
        torch.rand(1, 3, 2, 3),
    )
    trace_kwargs = {'th': 0.8, 'spatial_th': 10.0, 'mutual': False}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 5, 2, 3),
    )
    test_kwargs = {'th': 0.8, 'spatial_th': 10.0, 'mutual': False}
    _test_function(
        kornia.feature.match_fginn,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_match_adalam(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 128),
        torch.rand(3, 128),
        torch.rand(1, 3, 2, 3),
        torch.rand(1, 3, 2, 3),
    )
    trace_kwargs = {'config': None, 'hw1': None, 'hw2': None}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 5, 2, 3),
    )
    test_kwargs = {'config': None, 'hw1': None, 'hw2': None}
    _test_function(
        kornia.feature.match_adalam,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_extract_patches_from_pyramid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {'PS': 32}
    test_args = (
        torch.rand(1, 3, 64, 64),  # TODO: changing the batch size of these causes the trace_graph test to fail
        torch.rand(1, 5, 2, 3),
    )
    test_kwargs = {'PS': 16}
    _test_function(
        kornia.feature.extract_patches_from_pyramid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


# NOTE: this test can take a while to run (10+ mins)
# def test_extract_patches_simple(target_framework, mode, backend_compile):
#     trace_args = (
#         torch.rand(1, 3, 32, 32),
#         torch.rand(1, 5, 2, 3),
#     )
#     trace_kwargs = {'PS': 32, 'normalize_lafs_before_extraction': True}
#     test_args = (
#         torch.rand(2, 3, 64, 64),
#         torch.rand(2, 5, 2, 3),
#     )
#     test_kwargs = {'PS': 16, 'normalize_lafs_before_extraction': False}
#     _test_function(
#         kornia.feature.extract_patches_simple,
#         trace_args,
#         trace_kwargs,
#         test_args,
#         test_kwargs,
#         target_framework,
#         backend_compile,
#         tolerance=1e-4,
#         mode=mode,
#     )


def test_normalize_laf(target_framework, mode, backend_compile):
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
        kornia.feature.normalize_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_denormalize_laf(target_framework, mode, backend_compile):
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
        kornia.feature.denormalize_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_laf_to_boundary_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {'n_pts': 50}
    test_args = (
        torch.rand(2, 5, 2, 3),
    )
    test_kwargs = {'n_pts': 100}
    _test_function(
        kornia.feature.laf_to_boundary_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_ellipse_to_laf(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 10, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.ellipse_to_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_make_upright(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {'eps': 1e-9}
    test_args = (
        torch.rand(2, 5, 2, 3),
    )
    test_kwargs = {'eps': 1e-6}
    _test_function(
        kornia.feature.make_upright,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_scale_laf(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 5, 2, 3),
        2.0,
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.scale_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_laf_scale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.get_laf_scale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_laf_center(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.get_laf_center,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rotate_laf(target_framework, mode, backend_compile):
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
        kornia.feature.rotate_laf,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_laf_orientation(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 2, 3)),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((2, 10, 2, 3)),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.get_laf_orientation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_set_laf_orientation(target_framework, mode, backend_compile):
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
        kornia.feature.set_laf_orientation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_laf_from_center_scale_ori(target_framework, mode, backend_compile):
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
        kornia.feature.laf_from_center_scale_ori,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_laf_is_inside_image(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 1, 32, 32),
    )
    trace_kwargs = {'border': 0}
    test_args = (
        torch.rand(2, 10, 2, 3),
        torch.rand(2, 1, 64, 64),
    )
    test_kwargs = {'border': 1}
    _test_function(
        kornia.feature.laf_is_inside_image,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_laf_to_three_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.laf_to_three_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_laf_from_three_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 6),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 10, 6),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.laf_from_three_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_perspective_transform_lafs(target_framework, mode, backend_compile):
    trace_args = (
        torch.eye(3).repeat(1, 1, 1),
        torch.rand(1, 5, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.eye(3).repeat(2, 1, 1),
        torch.rand(2, 10, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.feature.perspective_transform_lafs,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
