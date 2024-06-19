from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_find_homography_dlt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
    )
    trace_kwargs = {'weights': torch.rand(1, 4), 'solver': 'svd'}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
    )
    test_kwargs = {'weights': torch.rand(5, 4), 'solver': 'svd'}
    _test_function(
        kornia.geometry.homography.find_homography_dlt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_find_homography_dlt_iterated(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
        torch.rand(1, 4),
    )
    trace_kwargs = {'soft_inl_th': 3.0, 'n_iter': 5}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {'soft_inl_th': 4.0, 'n_iter': 5}
    _test_function(
        kornia.geometry.homography.find_homography_dlt_iterated,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_find_homography_lines_dlt(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2, 2),
        torch.rand(1, 4, 2, 2),
    )
    trace_kwargs = {'weights': torch.rand(1, 4)}
    test_args = (
        torch.rand(5, 4, 2, 2),
        torch.rand(5, 4, 2, 2),
    )
    test_kwargs = {'weights': torch.rand(5, 4)}
    _test_function(
        kornia.geometry.homography.find_homography_lines_dlt,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_find_homography_lines_dlt_iterated(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2, 2),
        torch.rand(1, 4, 2, 2),
        torch.rand(1, 4),
    )
    trace_kwargs = {'soft_inl_th': 4.0, 'n_iter': 5}
    test_args = (
        torch.rand(5, 4, 2, 2),
        torch.rand(5, 4, 2, 2),
        torch.rand(5, 4),
    )
    test_kwargs = {'soft_inl_th': 3.0, 'n_iter': 5}
    _test_function(
        kornia.geometry.homography.find_homography_lines_dlt_iterated,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_line_segment_transfer_error_one_way(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2, 2),
        torch.rand(1, 4, 2, 2),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'squared': True}
    test_args = (
        torch.rand(5, 4, 2, 2),
        torch.rand(5, 4, 2, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {'squared': True}
    _test_function(
        kornia.geometry.homography.line_segment_transfer_error_one_way,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_oneway_transfer_error(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
        torch.rand(1, 3, 3),
    )
    trace_kwargs = {'squared': False, 'eps': 1e-8}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
        torch.rand(5, 3, 3),
    )
    test_kwargs = {'squared': False, 'eps': 1e-7}
    _test_function(
        kornia.geometry.homography.oneway_transfer_error,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_sample_is_valid_for_homography(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 2),
        torch.rand(1, 4, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 2),
        torch.rand(5, 4, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.geometry.homography.sample_is_valid_for_homography,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_symmetric_transfer_error(target_framework, mode, backend_compile):
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
    test_kwargs = {'squared': True, 'eps': 1e-7}
    _test_function(
        kornia.geometry.homography.symmetric_transfer_error,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
