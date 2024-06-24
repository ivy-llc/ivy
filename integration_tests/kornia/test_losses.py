from helpers import _test_function
import kornia
import torch


# Tests #
# ----- #

def test_ssim_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 5, 5),
        torch.rand(1, 4, 5, 5),
        5,
    )
    trace_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    test_args = (
        torch.rand(5, 4, 5, 5),
        torch.rand(5, 4, 5, 5),
        7,
    )
    test_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    _test_function(
        kornia.losses.ssim_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_ssim3d_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 4, 5, 5, 5),
        torch.rand(1, 4, 5, 5, 5),
        5,
    )
    trace_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    test_args = (
        torch.rand(5, 4, 5, 5, 5),
        torch.rand(5, 4, 5, 5, 5),
        7,
    )
    test_kwargs = {'max_val': 1.0, 'eps': 1e-12, 'reduction': 'mean', 'padding': 'same'}
    _test_function(
        kornia.losses.ssim3d_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_psnr_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
        1.0,
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
        1.0,
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.psnr_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_total_variation(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'sum'}
    test_args = (
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'sum'}
    _test_function(
        kornia.losses.total_variation,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_inverse_depth_smoothness_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 1, 4, 5),
        torch.rand(1, 3, 4, 5),
    )
    trace_kwargs = {}
    test_args = (
        torch.rand(5, 1, 4, 5),
        torch.rand(5, 3, 4, 5),
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.inverse_depth_smoothness_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_charbonnier_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.charbonnier_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_welsch_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.welsch_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_cauchy_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.cauchy_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_geman_mcclure_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand(1, 3, 32, 32),
        torch.rand(1, 3, 32, 32),
    )
    trace_kwargs = {'reduction': 'none'}
    test_args = (
        torch.rand(5, 3, 32, 32),
        torch.rand(5, 3, 32, 32),
    )
    test_kwargs = {'reduction': 'none'}
    _test_function(
        kornia.losses.geman_mcclure_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_binary_focal_loss_with_logits(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 3, 5)),
        torch.randint(2, (1, 3, 5)),
    )
    trace_kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
    test_args = (
        torch.randn((5, 3, 5)),
        torch.randint(2, (5, 3, 5)),
    )
    test_kwargs = {"alpha": 0.5, "gamma": 3.1, "reduction": 'mean'}
    _test_function(
        kornia.losses.binary_focal_loss_with_logits,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_focal_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.randint(5, (1, 3, 5)),
    )
    trace_kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
    test_args = (
        torch.randn((5, 5, 3, 5)),
        torch.randint(5, (5, 3, 5)),
    )
    test_kwargs = {"alpha": 0.7, "gamma": 2.5, "reduction": 'mean'}
    _test_function(
        kornia.losses.focal_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_dice_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    trace_kwargs = {"average": "micro", "eps": 1e-8}
    test_args = (
        torch.randn((5, 5, 3, 5)),
        torch.empty(5, 3, 5, dtype=torch.long).random_(5),
    )
    test_kwargs = {"average": "micro", "eps": 1e-8}
    _test_function(
        kornia.losses.dice_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_tversky_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    trace_kwargs = {"alpha": 0.5, "beta": 0.5, "eps": 1e-8}
    test_args = (
        torch.randn(5, 5, 3, 5),
        torch.empty(5, 3, 5, dtype=torch.long).random_(5),
    )
    test_kwargs = {"alpha": 0.5, "beta": 0.5, "eps": 1e-8}
    _test_function(
        kornia.losses.tversky_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_lovasz_hinge_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 1, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(1),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((5, 1, 3, 5)),
        torch.empty(5, 3, 5, dtype=torch.long).random_(1),
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.lovasz_hinge_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_lovasz_softmax_loss(target_framework, mode, backend_compile):
    trace_args = (
        torch.randn((1, 5, 3, 5)),
        torch.empty(1, 3, 5, dtype=torch.long).random_(5),
    )
    trace_kwargs = {}
    test_args = (
        torch.randn((5, 5, 3, 5)),
        torch.empty(5, 3, 5, dtype=torch.long).random_(5),
    )
    test_kwargs = {}
    _test_function(
        kornia.losses.lovasz_softmax_loss,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_js_div_loss_2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand((1, 1, 2, 4)),
        torch.rand((1, 1, 2, 4)),
    )
    trace_kwargs = {"reduction": "mean"}
    test_args = (
        torch.rand((5, 1, 2, 4)),
        torch.rand((5, 1, 2, 4)),
    )
    test_kwargs = {"reduction": "mean"}
    _test_function(
        kornia.losses.js_div_loss_2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )


def test_kl_div_loss_2d(target_framework, mode, backend_compile):
    trace_args = (
        torch.rand((1, 1, 2, 4)),
        torch.rand((1, 1, 2, 4)),
    )
    trace_kwargs = {"reduction": "mean"}
    test_args = (
        torch.rand((5, 1, 2, 4)),
        torch.rand((5, 1, 2, 4)),
    )
    test_kwargs = {"reduction": "mean"}
    _test_function(
        kornia.losses.kl_div_loss_2d,
        trace_args,
        trace_kwargs,
        test_args,
        test_kwargs,
        target_framework,
        backend_compile,
        tolerance=1e-4,
        mode=mode,
    )
