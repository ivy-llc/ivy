from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_rad2deg(target_framework, mode, backend_compile):
    trace_args = (torch.tensor(3.1415926535),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 3),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.rad2deg,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_deg2rad(target_framework, mode, backend_compile):
    trace_args = (torch.tensor(180.),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 3),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.deg2rad,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_pol2cart(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 3), torch.rand(1, 3, 3))
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 3), torch.rand(5, 3, 3))
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.pol2cart,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_cart2pol(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 3), torch.rand(1, 3, 3))
    trace_kwargs = {'eps': 1.0e-8}
    test_args = (torch.rand(5, 3, 3), torch.rand(5, 3, 3))
    test_kwargs = {'eps': 1.0e-8}
    _test_function(
        kornia.geometry.conversions.cart2pol,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_angle_to_rotation_matrix(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.angle_to_rotation_matrix,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_convert_points_from_homogeneous(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.convert_points_from_homogeneous,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_convert_points_to_homogeneous(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.convert_points_to_homogeneous,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_convert_affinematrix_to_homography(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.convert_affinematrix_to_homography,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_denormalize_pixel_coordinates(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.denormalize_pixel_coordinates,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_pixel_coordinates(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.normalize_pixel_coordinates,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_denormalize_pixel_coordinates3d(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.denormalize_pixel_coordinates3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_pixel_coordinates3d(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.normalize_pixel_coordinates3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_points_with_intrinsics(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2),
        torch.eye(3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.eye(3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.normalize_points_with_intrinsics,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_denormalize_points_with_intrinsics(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2),
        torch.eye(3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.eye(3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.denormalize_points_with_intrinsics,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_homography(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 3),
        (32, 32),
        (64, 64),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 3),
        (32, 32),  # TODO: changing these values fails the test
        (64, 64),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.normalize_homography,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_denormalize_homography(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 3),
        (64, 64),
        (32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 3),
        (64, 64),
        (32, 32),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.denormalize_homography,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_homography3d(target_framework, mode, backend_compile):
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
        kornia.geometry.normalize_homography3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_quaternion_to_axis_angle(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn(1, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.quaternion_to_axis_angle,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_quaternion_to_rotation_matrix(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn(1, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.quaternion_to_rotation_matrix,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_quaternion_log_to_exp(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn(1, 3),
    )
    trace_kwargs = {'eps': 1.0e-12}
    test_args = (
        torch.randn(5, 3),
    )
    test_kwargs = {'eps': 1.0e-12}
    _test_function(
        kornia.geometry.conversions.quaternion_log_to_exp,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_quaternion_exp_to_log(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn(1, 4),
    )
    trace_kwargs = {'eps': 1.0e-12}
    test_args = (
        torch.randn(5, 4),
    )
    test_kwargs = {'eps': 1.0e-12}
    _test_function(
        kornia.geometry.conversions.quaternion_exp_to_log,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize_quaternion(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn(1, 4),
    )
    trace_kwargs = {'eps': 1.0e-12}
    test_args = (
        torch.randn(5, 4),
    )
    test_kwargs = {'eps': 1.0e-12}
    _test_function(
        kornia.geometry.conversions.normalize_quaternion,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_vector_to_skew_symmetric_matrix(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([1.0, 2.0, 3.0]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.vector_to_skew_symmetric_matrix,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rotation_matrix_to_axis_angle(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.rotation_matrix_to_axis_angle,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rotation_matrix_to_quaternion(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'eps': 1e-8}
    test_args = (
        torch.rand(5, 3, 3),
    )
    test_kwargs = {'eps': 1e-4}
    _test_function(
        kornia.geometry.conversions.rotation_matrix_to_quaternion,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_axis_angle_to_quaternion(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand((1, 3)),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand((2, 3)),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.axis_angle_to_quaternion,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_axis_angle_to_rotation_matrix(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0., 0.], [1.5708, 0., 0.]]),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[0., 0., 0.], [1.5708, 0., 0.]]),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.axis_angle_to_rotation_matrix,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_quaternion_from_euler(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor(0.),
        torch.tensor(0.),
        torch.tensor(0.),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([0., 1., 2.]),
        torch.tensor([2., 1., 0.]),
        torch.tensor([1., 2., 0.]),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.quaternion_from_euler,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_euler_from_quaternion(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor(1.),
        torch.tensor(0.),
        torch.tensor(0.),
        torch.tensor(0.),
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor(2.),
        torch.tensor(1.),
        torch.tensor(1.),
        torch.tensor(1.),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.euler_from_quaternion,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_Rt_to_matrix4x4(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.Rt_to_matrix4x4,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_matrix4x4_to_Rt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.matrix4x4_to_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_worldtocam_to_camtoworld_Rt(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.worldtocam_to_camtoworld_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_camtoworld_to_worldtocam_Rt(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.camtoworld_to_worldtocam_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_camtoworld_graphics_to_vision_4x4(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.camtoworld_graphics_to_vision_4x4,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_camtoworld_vision_to_graphics_4x4(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.conversions.camtoworld_vision_to_graphics_4x4,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_camtoworld_graphics_to_vision_Rt(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.camtoworld_graphics_to_vision_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_camtoworld_vision_to_graphics_Rt(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.camtoworld_vision_to_graphics_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_ARKitQTVecs_to_ColmapQTVecs(target_framework, mode, backend_compile):
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
        kornia.geometry.conversions.ARKitQTVecs_to_ColmapQTVecs,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
