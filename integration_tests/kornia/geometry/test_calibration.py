from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_undistort_image(target_framework, mode, backend_compile):
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
        kornia.geometry.calibration.undistort_image,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_undistort_points(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.eye(3)[None],
        torch.rand(1, 4),
    )
    trace_kwargs = {'num_iters': 5}
    test_args = (
        torch.rand(1, 6, 2),
        torch.eye(3)[None],
        torch.rand(1, 4),
    )
    test_kwargs = {'num_iters': 5}
    _test_function(
        kornia.geometry.calibration.undistort_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_distort_points(target_framework, mode, backend_compile):
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
        kornia.geometry.calibration.distort_points,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_tilt_projection(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor(0.1),
        torch.tensor(0.2),
    )
    trace_kwargs = {'return_inverse': False}
    test_args = (
        torch.tensor(0.3),
        torch.tensor(0.4),
    )
    test_kwargs = {'return_inverse': False}
    _test_function(
        kornia.geometry.calibration.tilt_projection,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_solve_pnp_dlt(target_framework, mode, backend_compile):
    trace_args = (
        torch.tensor([[
            [5.0, -5.0, 0.0], [0.0, 0.0, 1.5],
            [2.5, 3.0, 6.0], [9.0, -2.0, 3.0],
            [-4.0, 5.0, 2.0], [-5.0, 5.0, 1.0]
        ]], dtype=torch.float64),
        torch.tensor([[
            [1409.1504, -800.936], [407.0207, -182.1229],
            [392.7021, 177.9428], [1016.838, -2.9416],
            [-63.1116, 142.9204], [-219.3874, 99.666]
        ]], dtype=torch.float64),
        torch.tensor([[
            [500.0, 0.0, 250.0],
            [0.0, 500.0, 250.0],
            [0.0, 0.0, 1.0]
        ]], dtype=torch.float64),
    )
    trace_kwargs = {'svd_eps': 1e-4}
    test_args = (
        torch.tensor([[
            [10.0, -10.0, 0.0], [0.0, 0.0, 3.0],
            [5.0, 6.0, 12.0], [18.0, -4.0, 6.0],
            [-8.0, 10.0, 4.0], [-10.0, 10.0, 2.0]
        ]], dtype=torch.float64),
        torch.tensor([[
            [2818.3008, -1601.872], [814.0414, -364.2458],
            [785.4042, 355.8856], [2033.676, -5.8832],
            [-126.2232, 285.8408], [-438.7748, 199.332]
        ]], dtype=torch.float64),
        torch.tensor([[
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0]
        ]], dtype=torch.float64),
    )
    test_kwargs = {'svd_eps': 1e-4}
    _test_function(
        kornia.geometry.calibration.solve_pnp_dlt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
