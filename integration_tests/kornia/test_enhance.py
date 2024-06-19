from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_add_weighted(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        0.5,
        torch.rand(1, 1, 5, 5),
        0.5,
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 5, 5),
        0.7,
        torch.rand(5, 1, 5, 5),
        0.8,
        0.8,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.add_weighted,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_brightness(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        1.3,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_brightness,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_contrast(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        2.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_contrast,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_contrast_with_mean_subtraction(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        2.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_contrast_with_mean_subtraction,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_gamma(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        2.2,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.4,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_gamma,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_hue(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 2, 2),
        -0.2,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_hue,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_saturation(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 2, 2),
        1.5,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_saturation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_sigmoid(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        0.5,
        0.1,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.7,
        0.05,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_sigmoid,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_adjust_log(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        1.2,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.adjust_log,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_invert(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.invert,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_posterize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        3,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        4,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.posterize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_sharpness(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 5, 5),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 5, 5),
        1.0,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.sharpness,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_solarize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 2, 2),
        0.5,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 2, 2),
        0.7,
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.solarize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_equalize(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 2, 3, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 2, 3, 3),)
    test_kwargs = {}
    _test_function(
        kornia.enhance.equalize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_equalize_clahe(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 10, 20),)
    trace_kwargs = {'clip_limit': 40.0, 'grid_size': (8, 8), 'slow_and_differentiable': False}
    test_args = (torch.rand(2, 3, 10, 20),)
    test_kwargs = {'clip_limit': 20.0, 'grid_size': (4, 4), 'slow_and_differentiable': False}
    _test_function(
        kornia.enhance.equalize_clahe,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_equalize3d(target_framework, mode, backend_compile):
    trace_args = (torch.rand(1, 2, 3, 3, 3),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 2, 3, 3, 3),)
    test_kwargs = {}
    _test_function(
        kornia.enhance.equalize3d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework, 
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )



def test_histogram(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 10),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    trace_kwargs = {'epsilon': 1e-10}
    test_args = (
        torch.rand(5, 10),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    test_kwargs = {'epsilon': 1e-10}
    _test_function(
        kornia.enhance.histogram,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_histogram2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(2, 32),
        torch.rand(2, 32),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    trace_kwargs = {'epsilon': 1e-10}
    test_args = (
        torch.rand(5, 32),
        torch.rand(5, 32),
        torch.linspace(0, 255, 128),
        torch.tensor(0.9),
    )
    test_kwargs = {'epsilon': 1e-10}
    _test_function(
        kornia.enhance.histogram2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_image_histogram2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 10, 10),
    )
    trace_kwargs = {
        'min': 0.0,
        'max': 255.0,
        'n_bins': 256,
        'bandwidth': None,
        'centers': None,
        'return_pdf': False,
        'kernel': 'triangular',
        'eps': 1e-10
    }
    test_args = (
        torch.rand(5, 1, 10, 10),
    )
    test_kwargs = {
        'min': 0.0,
        'max': 255.0,
        'n_bins': 256,
        'bandwidth': None,
        'centers': None,
        'return_pdf': False,
        'kernel': 'triangular',
        'eps': 1e-10
    }
    _test_function(
        kornia.enhance.image_histogram2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_normalize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([0.5, 0.5, 0.5])
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([0.4, 0.4, 0.4]),
        torch.tensor([0.6, 0.6, 0.6])
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.normalize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )

def test_normalize_min_max(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        0.0,
        1.0
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        -1.0,
        1.0
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.normalize_min_max,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )

def test_denormalize(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 4, 4),
        torch.tensor([0.5, 0.5, 0.5]),
        torch.tensor([0.5, 0.5, 0.5])
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 4, 4),
        torch.tensor([0.4, 0.4, 0.4]),
        torch.tensor([0.6, 0.6, 0.6])
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.denormalize,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )

def test_zca_mean(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(10, 20),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 20),
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.zca_mean,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )

def test_zca_whiten(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(10, 20),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 20),
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.zca_whiten,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )

def test_linear_transform(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(10, 3, 4, 5),
        torch.eye(10 * 3 * 4),
        torch.zeros(1, 10 * 3 * 4)
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 10, 3, 4, 5),
        torch.eye(10 * 3 * 4),
        torch.zeros(1, 10 * 3 * 4)
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.linear_transform,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )

def test_jpeg_codec_differentiable(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(3, 3, 64, 64),
        torch.tensor([99.0])
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 3, 64, 64),
        torch.tensor([50.0])
    )
    test_kwargs = {}
    _test_function(
        kornia.enhance.jpeg_codec_differentiable,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
