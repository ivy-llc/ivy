from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #


def test_rgb_to_grayscale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_grayscale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_bgr_to_grayscale(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.bgr_to_grayscale,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_grayscale_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.grayscale_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_bgr(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_bgr,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_bgr_to_rgb(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.bgr_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_linear_rgb(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_linear_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_linear_rgb_to_rgb(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 3, 5, 5),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 5, 5),)
    test_kwargs = {}
    _test_function(
        kornia.color.linear_rgb_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_bgr_to_rgba(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.bgr_to_rgba,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_rgba(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_rgba,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgba_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgba_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgba_to_bgr(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 4, 4),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 4, 4, 4),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgba_to_bgr,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_hls(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 5),
    )
    trace_kwargs = {'eps': 1e-8}
    test_args = (
        torch.rand(5, 3, 4, 5),
    )
    test_kwargs = {'eps': 1e-8}
    _test_function(
        kornia.color.rgb_to_hls,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_hls_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.hls_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_hsv(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {'eps': 1e-8}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {'eps': 1e-2}
    _test_function(
        kornia.color.rgb_to_hsv,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_hsv_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.hsv_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_luv(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {
        'eps': 1e-12
    }
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {
        'eps': 1e-12
    }
    _test_function(
        kornia.color.rgb_to_luv,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_luv_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {'eps': 1e-12}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {'eps': 1e-12}
    _test_function(
        kornia.color.luv_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_lab(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_lab,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_lab_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {'clip': True}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {'clip': True}
    _test_function(
        kornia.color.lab_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_ycbcr(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_ycbcr,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_ycbcr_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.ycbcr_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_yuv(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_yuv,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_yuv_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.yuv_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_yuv420(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 6),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 6),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_yuv420,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_yuv420_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 6),
        torch.rand(1, 2, 2, 3) * 2.0 - 0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 6),
        torch.rand(5, 2, 2, 3) * 2.0 - 0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.yuv420_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_yuv422(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 6, 8),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 6, 8),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_yuv422,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_yuv422_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 6, 6),
        torch.rand(1, 2, 3, 3) - 0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 6, 6),
        torch.rand(5, 2, 3, 3) - 0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.yuv422_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_xyz(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_xyz,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_xyz_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.color.xyz_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_rgb_to_raw(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 5, 5),
        kornia.color.CFA.BG,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 5, 5),
        kornia.color.CFA.BG,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.rgb_to_raw,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_raw_to_rgb(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 6),
        kornia.color.CFA.RG,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 6),
        kornia.color.CFA.RG,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.raw_to_rgb,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_raw_to_rgb_2x2_downscaled(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 6),
        kornia.color.CFA.RG,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 6),
        kornia.color.CFA.RG,
    )
    test_kwargs = {}
    _test_function(
        kornia.color.raw_to_rgb_2x2_downscaled,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_sepia(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
    )
    trace_kwargs = {
        'rescale': True,
        'eps': 1e-6,
    }
    test_args = (
        torch.rand(5, 3, 4, 4),
    )
    test_kwargs = {
        'rescale': True,  # TODO: changing this to False fails
        'eps': 1e-6,
    }
    _test_function(
        kornia.color.sepia,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )

