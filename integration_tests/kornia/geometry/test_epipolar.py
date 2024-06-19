from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

# NOTE: takes a while to run this test
# def test_find_essential(target_framework, mode, backend_compile):
#     trace_args = (
#         torch.rand(1, 8, 2),
#         torch.rand(1, 8, 2),
#     )
#     trace_kwargs = {'weights': torch.rand(1, 8)}
#     test_args = (
#         torch.rand(5, 8, 2),
#         torch.rand(5, 8, 2),
#     )
#     test_kwargs = {'weights': torch.rand(5, 8)}
#     _test_function(
#         kornia.geometry.epipolar.find_essential,
#         trace_args,
#         trace_kwargs,
#         test_args,
#         test_kwargs,
#         target_framework, 
#         backend_compile,
#         tolerance=1e-4,
#         mode=mode,
#     )


def test_essential_from_fundamental(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.essential_from_fundamental,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_essential_from_Rt(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.essential_from_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_decompose_essential_matrix(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.decompose_essential_matrix,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_motion_from_essential(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.motion_from_essential,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_motion_from_essential_choose_solution(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 3),
        torch.rand(3, 3),
        torch.rand(3, 3),
        torch.rand(8, 2),
        torch.rand(8, 2),
    )
    trace_kwargs = {'mask': torch.ones(8).bool()}
    test_args = (
        torch.rand(3, 3),
        torch.rand(3, 3),
        torch.rand(3, 3),
        torch.rand(8, 2),
        torch.rand(8, 2),
    )
    test_kwargs = {'mask': torch.ones(8).bool()}
    _test_function(
        kornia.geometry.epipolar.motion_from_essential_choose_solution,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_relative_camera_motion(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.relative_camera_motion,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_find_fundamental(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 8, 2),
        torch.rand(2, 8, 2),
    )
    trace_kwargs = {'weights': torch.rand(2, 8), 'method': '8POINT'}
    test_args = (
        torch.rand(5, 8, 2),
        torch.rand(5, 8, 2),
    )
    test_kwargs = {'weights': torch.rand(5, 8), 'method': '8POINT'}
    _test_function(
        kornia.geometry.epipolar.find_fundamental,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_fundamental_from_essential(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.fundamental_from_essential,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_fundamental_from_projections(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 4),
        torch.rand(3, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 4),
        torch.rand(3, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.fundamental_from_projections,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_compute_correspond_epilines(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.compute_correspond_epilines,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 8, 2),
    )
    trace_kwargs = {'eps': 1e-8}
    test_args = (
        torch.rand(5, 8, 2),
    )
    test_kwargs = {'eps': 1e-8}
    _test_function(
        kornia.geometry.epipolar.normalize_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_transformation(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 3),
    )
    trace_kwargs = {'eps': 1e-8}
    test_args = (
        torch.rand(3, 3),
    )
    test_kwargs = {'eps': 1e-8}
    _test_function(
        kornia.geometry.epipolar.normalize_transformation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_perpendicular(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.get_perpendicular,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_closest_point_on_epipolar_line(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2, 2),
        torch.rand(1, 2, 2),
        torch.rand(2, 3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.get_closest_point_on_epipolar_line,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_sampson_epipolar_distance(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'squared': True, 'eps': 1e-8}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {'squared': True, 'eps': 1e-8}
    _test_function(
        kornia.geometry.epipolar.sampson_epipolar_distance,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_symmetrical_epipolar_distance(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'squared': True, 'eps': 1e-8}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {'squared': True, 'eps': 1e-8}
    _test_function(
        kornia.geometry.epipolar.symmetrical_epipolar_distance,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_left_to_right_epipolar_distance(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.left_to_right_epipolar_distance,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_right_to_left_epipolar_distance(target_framework, mode, backend_compile):
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
        kornia.geometry.epipolar.right_to_left_epipolar_distance,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_projection_from_KRt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 3),
        torch.eye(3),
        torch.rand(3, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 3, 3),
        torch.stack([torch.eye(3) for _ in range(2)]),
        torch.rand(2, 3, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.projection_from_KRt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_projections_from_fundamental(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(2, 3, 3),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.projections_from_fundamental,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_intrinsics_like(target_framework, mode, backend_compile):
    trace_args = (0.5, torch.rand(1, 3, 256, 256))
    trace_kwargs = {}
    test_args = (0.8, torch.rand(2, 3, 256, 256))
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.intrinsics_like,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_scale_intrinsics(target_framework, mode, backend_compile):
    trace_args = (torch.rand(3, 3), 0.5)
    trace_kwargs = {}
    test_args = (torch.rand(2, 3, 3), 1.2)
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.scale_intrinsics,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_random_intrinsics(target_framework, mode, backend_compile):
    trace_args = (0.1, 1.0)
    trace_kwargs = {}
    test_args = (0.2, 2.0)
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.random_intrinsics,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_cross_product_matrix(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.cross_product_matrix,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_triangulate_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4),
        torch.rand(1, 3, 4),
        torch.rand(1, 10, 2),
        torch.rand(1, 10, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2, 3, 4),
        torch.rand(2, 3, 4),
        torch.rand(2, 20, 2),
        torch.rand(2, 20, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.epipolar.triangulate_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
