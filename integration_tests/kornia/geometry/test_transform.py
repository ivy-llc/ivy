from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_warp_perspective(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.eye(3).unsqueeze(0),
        (4, 4),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.eye(3).unsqueeze(0),
        (4, 4),  # TODO: changing this fails the test
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.warp_perspective,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_perspective3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5, 5),
        torch.eye(4).unsqueeze(0),
        (4, 4, 4),
    )
    trace_kwargs = {'flags': 'bilinear', 'border_mode': 'zeros', 'align_corners': False}
    test_args = (
        torch.rand(5, 3, 5, 5, 5),
        torch.eye(4).unsqueeze(0),
        (4, 4, 4),
    )
    test_kwargs = {'flags': 'bilinear', 'border_mode': 'zeros', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.warp_perspective3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.eye(2, 3).unsqueeze(0),
        (4, 4),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.eye(2, 3).unsqueeze(0),
        (4, 4),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.warp_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_affine3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5, 5),
        torch.eye(3, 4).unsqueeze(0),
        (4, 4, 4),
    )
    trace_kwargs = {'flags': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 5, 5, 5),
        torch.eye(3, 4).unsqueeze(0),
        (4, 4, 4),
    )
    test_kwargs = {'flags': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.warp_affine3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_image_tps(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
        torch.rand(1, 3, 2),
    )
    trace_kwargs = {'align_corners': False}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 5, 2),
        torch.rand(5, 5, 2),
        torch.rand(5, 3, 2),
    )
    test_kwargs = {'align_corners': False}
    _test_function(
        kornia.geometry.transform.warp_image_tps,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_points_tps(target_framework, mode, backend_compile):
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
        kornia.geometry.transform.warp_points_tps,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_grid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 5, 2),
        torch.eye(3).unsqueeze(0),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 10, 10, 2),
        torch.eye(3).unsqueeze(0),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.warp_grid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_grid3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 5, 5, 3),
        torch.eye(4).unsqueeze(0),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 10, 10, 10, 3),
        torch.eye(4).unsqueeze(0),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.warp_grid3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_remap(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 5, 5),
        torch.rand(1, 5, 5),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': None, 'normalized_coordinates': False}
    test_args = (
        torch.rand(1, 3, 10, 10),
        torch.rand(1, 10, 10),
        torch.rand(1, 10, 10),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': None, 'normalized_coordinates': False}
    _test_function(
        kornia.geometry.transform.remap,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2, 3, 5),
        torch.eye(2, 3).unsqueeze(0),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 2, 3, 5),
        torch.eye(2, 3).unsqueeze(0).repeat(5, 1, 1),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rotate(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.tensor([90.]),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([45., 45., 45., 45., 45.]),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.rotate,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_translate(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.tensor([[1., 0.]]),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([[1., 0.]]).repeat(5, 1),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.translate,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_scale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.rand(1),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.rand(5),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': True}
    _test_function(
        kornia.geometry.transform.scale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_shear(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.tensor([[0.5, 0.0]]),
    )
    trace_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': False}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([[0.1, 0.3]]).repeat(5, 1),
    )
    test_kwargs = {'mode': 'bilinear', 'padding_mode': 'zeros', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.shear,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_hflip(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.hflip,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_vflip(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.vflip,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rot180(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.rot180,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_resize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        (6, 8),
    )
    trace_kwargs = {'interpolation': 'bilinear'}
    test_args = (
        torch.rand(5, 3, 4, 4),
        (12, 16),
    )
    test_kwargs = {'interpolation': 'bilinear'}
    _test_function(
        kornia.geometry.transform.resize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rescale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        (2, 3),
    )
    trace_kwargs = {'interpolation': 'bilinear'}
    test_args = (
        torch.rand(5, 3, 4, 4),
        (1.5, 2),
    )
    test_kwargs = {'interpolation': 'bilinear'}
    _test_function(
        kornia.geometry.transform.rescale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_elastic_transform2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        torch.rand(1, 2, 5, 5),
    )
    trace_kwargs = {'kernel_size': (63, 63), 'sigma': (32.0, 32.0), 'alpha': (1.0, 1.0)}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(5, 2, 5, 5),
    )
    test_kwargs = {'kernel_size': (31, 31), 'sigma': (16.0, 16.0), 'alpha': (0.5, 0.5)}
    _test_function(
        kornia.geometry.transform.elastic_transform2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_pyrdown(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 4),
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False, 'factor': 2.0}
    test_args = (
        torch.rand(5, 1, 8, 8),
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False, 'factor': 2.0}
    _test_function(
        kornia.geometry.transform.pyrdown,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_pyrup(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False}
    test_args = (
        torch.rand(5, 1, 4, 4),
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.pyrup,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_build_pyramid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 8, 8),
        3,
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False}
    test_args = (
        torch.rand(5, 3, 16, 16),
        4,
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.build_pyramid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_build_laplacian_pyramid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 8, 8),
        3,
    )
    trace_kwargs = {'border_type': 'reflect', 'align_corners': False}
    test_args = (
        torch.rand(5, 3, 16, 16),
        4,
    )
    test_kwargs = {'border_type': 'reflect', 'align_corners': False}
    _test_function(
        kornia.geometry.transform.build_laplacian_pyramid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_upscale_double(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 8, 8),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.upscale_double,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_perspective_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]]),
        torch.tensor([[[1., 0.], [0., 0.], [0., 1.], [1., 1.]]])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[[0., 0.], [2., 0.], [2., 2.], [0., 2.]]]),
        torch.tensor([[[2., 0.], [0., 0.], [0., 2.], [2., 2.]]])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_perspective_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_perspective_transform3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]]),
        torch.tensor([[[1., 0., 0.], [0., 0., 0.], [0., 1., 0.], [1., 1., 0.]]])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]]]),
        torch.tensor([[[2., 0., 0.], [0., 0., 0.], [0., 2., 0.], [2., 2., 0.]]])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_perspective_transform3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_projective_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([[30., 45., 60.]]),
        torch.tensor([[1., 1., 1.]])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1., 1.]]),
        torch.tensor([[45., 60., 75.]]),
        torch.tensor([[1.5, 1.5, 1.5]])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_projective_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_rotation_matrix2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2),
        45. * torch.ones(1),
        torch.rand(1, 2)
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(1, 2),
        90. * torch.ones(1),
        2.0 * torch.ones(1, 2)
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_rotation_matrix2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_shear_matrix2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0.]]),
        torch.tensor([1.0]),
        torch.tensor([0.5])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1.]]),
        torch.tensor([1.5]),
        torch.tensor([0.75])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_shear_matrix2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_shear_matrix3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        torch.tensor([0.2]),
        torch.tensor([0.3]),
        torch.tensor([0.4]),
        torch.tensor([0.6])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1., 1.]]),
        torch.tensor([1.5]),
        torch.tensor([0.75]),
        torch.tensor([0.4]),
        torch.tensor([0.5]),
        torch.tensor([0.6]),
        torch.tensor([0.8])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_shear_matrix3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_affine_matrix2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0.]]),
        torch.tensor([[0., 0.]]),
        torch.ones(1, 2),
        45. * torch.ones(1),
        torch.tensor([1.0]),
        torch.tensor([0.5])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1.]]),
        torch.tensor([[1., 1.]]),
        2.0 * torch.ones(1, 2),
        90. * torch.ones(1),
        torch.tensor([1.5]),
        torch.tensor([0.75])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_affine_matrix2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_affine_matrix3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[0., 0., 0.]]),
        torch.tensor([[0., 0., 0.]]),
        torch.ones(1, 3),
        torch.tensor([[45., 45., 45.]]),
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        torch.tensor([0.2]),
        torch.tensor([0.3]),
        torch.tensor([0.4]),
        torch.tensor([0.6])
    )
    trace_kwargs = {}
    test_args = (
        torch.tensor([[1., 1., 1.]]),
        torch.tensor([[1., 1., 1.]]),
        2.0 * torch.ones(1, 3),
        torch.tensor([[90., 90., 90.]]),
        torch.tensor([1.5]),
        torch.tensor([0.75]),
        torch.tensor([0.4]),
        torch.tensor([0.5]),
        torch.tensor([0.6]),
        torch.tensor([0.8])
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_affine_matrix3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_invert_affine_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.invert_affine_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_projection_from_Rt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 3),
        torch.rand(1, 3, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 3),
        torch.rand(5, 3, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.projection_from_Rt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_get_tps_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 5, 2),
        torch.rand(1, 5, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 5, 2),
        torch.rand(5, 5, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.get_tps_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_crop_by_indices(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=torch.float32),
    )
    trace_kwargs = {'size': (40, 40), 'interpolation': 'bilinear'}
    test_args = (
        torch.rand(5, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]]*5, dtype=torch.float32),
    )
    test_kwargs = {'size': (40, 40), 'interpolation': 'bilinear'}
    _test_function(
        kornia.geometry.transform.crop_by_indices,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_crop_by_boxes(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=torch.float32),
        torch.tensor([[[0, 0], [40, 0], [40, 40], [0, 40]]], dtype=torch.float32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]]*5, dtype=torch.float32),
        torch.tensor([[[0, 0], [40, 0], [40, 40], [0, 40]]]*5, dtype=torch.float32),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.crop_by_boxes,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_center_crop(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 64, 64),
        (32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 64, 64),
        (32, 32),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.center_crop,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_crop_and_resize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]], dtype=torch.float32),
        (32, 32),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 64, 64),
        torch.tensor([[[10, 10], [50, 10], [50, 50], [10, 50]]]*5, dtype=torch.float32),
        (32, 32),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.transform.crop_and_resize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
