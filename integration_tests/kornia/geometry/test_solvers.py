from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_solve_quadratic(target_framework, mode, backend_compile):
    trace_args = (torch.tensor([[1., -3., 2.]]),)
    trace_kwargs = {}
    test_args = (torch.tensor([[1., -2., 1.]]),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.solvers.solve_quadratic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_solve_cubic(target_framework, mode, backend_compile):
    trace_args = (torch.tensor([[1., -6., 11., -6.]]),)
    trace_kwargs = {}
    test_args = (torch.tensor([[1., -4., 6., -4.]]),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.solvers.solve_cubic,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_multiply_deg_one_poly(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 9),
        torch.rand(1, 9),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(3, 9),
        torch.rand(3, 9),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.solvers.multiply_deg_one_poly,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_multiply_deg_two_one_poly(target_framework, mode, backend_compile):
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
        kornia.geometry.solvers.multiply_deg_two_one_poly,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_determinant_to_polynomial(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 13),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 13),)
    test_kwargs = {}
    _test_function(
        kornia.geometry.solvers.determinant_to_polynomial,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
