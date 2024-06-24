from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_project_points_z1(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.project_points_z1,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_unproject_points_z1(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2),
        torch.rand(1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(2),
        torch.rand(1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.unproject_points_z1,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_dx_project_points_z1(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.dx_project_points_z1,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_project_points_orthographic(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.project_points_orthographic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_unproject_points_orthographic(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.unproject_points_orthographic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_dx_project_points_orthographic(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.dx_project_points_orthographic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_distort_points_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.distort_points_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_undistort_points_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.undistort_points_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_dx_distort_points_affine(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 2),
        torch.rand(2, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.camera.dx_distort_points_affine,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_distort_points_kannala_brandt(target_framework, mode, backend_compile):
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
        kornia.geometry.camera.distort_points_kannala_brandt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_undistort_points_kannala_brandt(target_framework, mode, backend_compile):
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
        kornia.geometry.camera.undistort_points_kannala_brandt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_dx_distort_points_kannala_brandt(target_framework, mode, backend_compile):
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
        kornia.geometry.camera.dx_distort_points_kannala_brandt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
