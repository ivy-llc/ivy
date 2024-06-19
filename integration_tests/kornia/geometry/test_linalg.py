from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_relative_transformation(target_framework, mode, backend_compile):
    trace_args = (
        torch.eye(4),
        torch.eye(4),
    )
    trace_kwargs = {}
    test_args = (
        torch.eye(4).repeat(5, 1, 1),
        torch.eye(4).repeat(5, 1, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.linalg.relative_transformation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_compose_transformations(target_framework, mode, backend_compile):
    trace_args = (
        torch.eye(4),
        torch.eye(4),
    )
    trace_kwargs = {}
    test_args = (
        torch.eye(4).repeat(5, 1, 1),
        torch.eye(4).repeat(5, 1, 1),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.linalg.compose_transformations,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_inverse_transformation(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.linalg.inverse_transformation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_transform_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.eye(4).view(1, 4, 4),
        torch.rand(2, 4, 3),
    )
    trace_kwargs = {}
    test_args = (
        torch.eye(4).repeat(5, 1, 1),
        torch.rand(5, 4, 3),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.linalg.transform_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_point_line_distance(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3),
        torch.rand(3),
    )
    trace_kwargs = {'eps': 1e-9}
    test_args = (
        torch.rand(5, 3),
        torch.rand(5, 3),
    )
    test_kwargs = {'eps': 1e-9}
    _test_function(
        kornia.geometry.linalg.point_line_distance,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_squared_norm(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(5),
    )
    trace_kwargs = {'keepdim': False}
    test_args = (
        torch.rand(10, 5),
    )
    test_kwargs = {'keepdim': False}
    _test_function(
        kornia.geometry.linalg.squared_norm,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_batched_dot_product(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 5),
        torch.rand(3, 5),
    )
    trace_kwargs = {'keepdim': False}
    test_args = (
        torch.rand(5, 3, 5),
        torch.rand(5, 3, 5),
    )
    test_kwargs = {'keepdim': False}
    _test_function(
        kornia.geometry.linalg.batched_dot_product,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_euclidean_distance(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 5),
        torch.rand(3, 5),
    )
    trace_kwargs = {'keepdim': False, 'eps': 1e-6}
    test_args = (
        torch.rand(5, 3, 5),
        torch.rand(5, 3, 5),
    )
    test_kwargs = {'keepdim': False, 'eps': 1e-6}
    _test_function(
        kornia.geometry.linalg.euclidean_distance,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
