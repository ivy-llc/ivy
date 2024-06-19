from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_depth_from_disparity(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(4, 1, 4, 4),
        torch.rand(1),
        torch.rand(1),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 5, 5),
        torch.rand(1),
        torch.rand(1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.depth.depth_from_disparity,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_depth_to_3d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 4),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'normalize_points': False}
    test_args = (
        torch.rand(5, 1, 5, 5),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {'normalize_points': False}
    _test_function(
        kornia.geometry.depth.depth_to_3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_depth_to_3d_v2(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(4, 4),
        torch.eye(3),
    )
    trace_kwargs = {'normalize_points': False}
    test_args = (
        torch.rand(5, 5),
        torch.eye(3),
    )
    test_kwargs = {'normalize_points': False}
    _test_function(
        kornia.geometry.depth.depth_to_3d_v2,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_unproject_meshgrid(target_framework, mode, backend_compile):
    trace_args = (
        4,
        4,
        torch.eye(3),
    )
    trace_kwargs = {'normalize_points': False, 'device': 'cpu', 'dtype': torch.float32}
    test_args = (
        5,
        5,
        torch.eye(3),
    )
    test_kwargs = {'normalize_points': False, 'device': 'cpu', 'dtype': torch.float32}
    _test_function(
        kornia.geometry.depth.unproject_meshgrid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_depth_to_normals(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 4),
        torch.eye(3)[None],
    )
    trace_kwargs = {'normalize_points': False}
    test_args = (
        torch.rand(1, 1, 5, 5),
        torch.eye(3)[None],
    )
    test_kwargs = {'normalize_points': False}
    _test_function(
        kornia.geometry.depth.depth_to_normals,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_warp_frame_depth(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.rand(1, 1, 4, 4),
        torch.eye(4)[None],
        torch.eye(3)[None],
    )
    trace_kwargs = {'normalize_points': False}
    test_args = (
        torch.rand(5, 3, 5, 5),
        torch.rand(5, 1, 5, 5),
        torch.eye(4)[None].repeat(5, 1, 1),
        torch.eye(3)[None].repeat(5, 1, 1),
    )
    test_kwargs = {'normalize_points': False}
    _test_function(
        kornia.geometry.depth.warp_frame_depth,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
