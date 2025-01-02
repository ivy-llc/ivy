"""
Runs a big script of calls to transpiled transpiled_kornia, testing lazy transpilation of modules.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow as tf
import torch
from ivy.transpiler import transpile
from ivy_tests.test_transpiler.kornia.helpers import (
    _check_allclose,
    _nest_array_to_numpy,
    _nest_torch_tensor_to_new_framework,
    download_image,
)


# Helpers #
# ------- #

def _array_to_new_backend(
    x,
    target,
):
    """
    Converts a torch tensor to an array/tensor in a different framework.
    If the input is not a torch tensor, the input if returned without modification.
    """

    if isinstance(x, torch.Tensor):
        if target == "torch":
            return x
        y = x.detach().numpy()
        if target == "jax":
            y = jnp.array(y)
        elif target == "tensorflow":
            y = tf.convert_to_tensor(y)
        return y
    else:
        return x


# Main #
# ---- #

@pytest.mark.parametrize("target", ["jax", "tensorflow"])
def test_module(target):
    import kornia

    transpiled_kornia = transpile(kornia, source="torch", target=target)

    img = _array_to_new_backend(torch.rand((1, 3, 144, 256)), target)
    coords = _array_to_new_backend(torch.tensor([[[125, 40.0], [185.0, 75.0]]]), target)
    transpiled_kornia.tensor_to_image(img)

    if target != "tensorflow":
        # test instantiated objs as inputs #1
        resize_op = transpiled_kornia.augmentation.AugmentationSequential(
            transpiled_kornia.augmentation.Resize((100, 200), antialias=True),
            data_keys=["input", "keypoints"],
        )
        img_resize, coords_resize = resize_op(img, coords)
        transpiled_kornia.tensor_to_image(img_resize)

        crop_op = transpiled_kornia.augmentation.AugmentationSequential(
            transpiled_kornia.augmentation.CenterCrop((100, 200)),
            data_keys=["input", "keypoints"],
        )
        img_resize, coords_resize = crop_op(img, coords)
        transpiled_kornia.tensor_to_image(img_resize)

        # test instantiated objs as inputs #2
        transpiled_aug_list = transpiled_kornia.augmentation.container.PatchSequential(
            transpiled_kornia.augmentation.container.ImageSequential(
                transpiled_kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                transpiled_kornia.augmentation.RandomPerspective(0.2, p=0.5),
                transpiled_kornia.augmentation.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            transpiled_kornia.augmentation.RandomAffine(360, p=1.0),
            transpiled_kornia.augmentation.container.ImageSequential(
                transpiled_kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=0.5),
                transpiled_kornia.augmentation.RandomPerspective(0.2, p=0.5),
                transpiled_kornia.augmentation.RandomSolarize(0.1, 0.1, p=0.5),
            ),
            transpiled_kornia.augmentation.RandomSolarize(0.1, 0.1, p=0.1),
            grid_size=(2,2),
            patchwise_apply=True,
            same_on_batch=True,
            random_apply=False,
        )

        torch_args = (
            torch.randn(2, 3, 224, 224),
        )
        transpiled_args = _nest_torch_tensor_to_new_framework(torch_args, target)
        transpiled_out = transpiled_aug_list(*transpiled_args)

    # test classmethod calls
    rotation_matrix = torch.eye(3)
    transpiled_rotation_matrix = _array_to_new_backend(rotation_matrix, target)
    transpiled_from_matrix = transpiled_kornia.geometry.liegroup.So3.from_matrix(
        transpiled_rotation_matrix
    )

    # test staticmethod calls
    if target != "tensorflow":
        theta = torch.tensor([3.1415 / 2])
        transpiled_theta = _array_to_new_backend(theta, target)
        transpiled_exp = transpiled_kornia.geometry.liegroup.So2.exp(transpiled_theta)

    input1 = _array_to_new_backend(torch.rand((1, 1, 5, 5)), target)
    input2 = _array_to_new_backend(torch.rand((1, 1, 5, 5)), target)
    output = transpiled_kornia.enhance.add_weighted(input1, 0.5, input2, 0.5, 1.0)

    x = _array_to_new_backend(torch.ones((2, 5, 3, 3)), target)
    y = _array_to_new_backend(torch.tensor([0.25, 0.50]), target)
    transpiled_kornia.enhance.adjust_brightness(x, y).shape

    x = _array_to_new_backend(torch.rand((1, 4, 3, 3, 3)), target)
    mean = _array_to_new_backend(torch.zeros((1, 4)), target)
    std = _array_to_new_backend(255.0 * torch.ones((1, 4)), target)
    out = transpiled_kornia.enhance.Denormalize(mean, std)(x)

    x = _array_to_new_backend(torch.ones((1, 1, 3, 3)), target)
    transpiled_kornia.enhance.AdjustBrightness(1.0)(x)

    x = _array_to_new_backend(torch.ones((2, 3, 3, 3)), target)
    y = _array_to_new_backend(torch.ones((2)), target)
    out = transpiled_kornia.enhance.AdjustSaturation(y)(x)

    x = _array_to_new_backend(torch.ones((2, 3, 3, 3)), target)
    y = _array_to_new_backend(torch.ones((2)) * 3.141516, target)
    transpiled_kornia.enhance.AdjustHue(y)(x).shape

    x = _array_to_new_backend(
        torch.tensor([[0, 1], [1, 0], [-1, 0]], dtype=torch.float32), target
    )
    transpiled_kornia.enhance.zca_whiten(x)

    input = _array_to_new_backend(torch.rand((2, 3, 4, 5)), target)
    gray = transpiled_kornia.color.rgb_to_grayscale(input)

    input = _array_to_new_backend(torch.rand((2, 3, 4, 5)), target)
    lab = transpiled_kornia.color.RgbToLab()
    output = lab(input)

    url = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"
    download_image(url)

    # test enums as inputs
    translated_image = transpiled_kornia.io.load_image(
        "panda.jpg", transpiled_kornia.io.ImageLoadType.RGB32
    )

    inputs = _array_to_new_backend(torch.ones(1, 3, 3, 3), target)
    aug = transpiled_kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)
    aug(inputs)

    inputs = _array_to_new_backend(torch.ones(1, 3, 3, 3), target)
    aug = transpiled_kornia.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0)
    aug(inputs)

    img = _array_to_new_backend(torch.ones(1, 1, 24, 24), target)
    out = transpiled_kornia.augmentation.RandomBoxBlur((7, 7))(img)
    out.shape

    inputs = _array_to_new_backend(torch.rand(1, 3, 3, 3), target)
    aug = transpiled_kornia.augmentation.RandomBrightness(brightness=(0.5, 2.0), p=1.0)
    aug(inputs)

    img = _array_to_new_backend(torch.ones(1, 3, 3, 3), target)
    aug = transpiled_kornia.augmentation.RandomChannelDropout(
        num_drop_channels=1, fill_value=0.0, p=1.0
    )
    aug(img)

    input = _array_to_new_backend(torch.randn(1, 3, 2, 2), target)
    aug = transpiled_kornia.augmentation.RandomPlanckianJitter(mode="CIED")
    aug(input)

    img = _array_to_new_backend(
        torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]), target
    )
    pad = transpiled_kornia.augmentation.PadTo((4, 5), pad_value=1.0)
    out = pad(img)

    # input = _array_to_new_backend(torch.rand(1, 1, 3, 3), target)
    # aug = transpiled_kornia.augmentation.RandomAffine((-15., 20.), p=1.)
    # out = aug(input)
    # out, aug.transform_matrix


@pytest.mark.parametrize("target", ["jax", "numpy", "tensorflow"])
def test_module_wrapping(target):
    import kornia

    kornia = transpile(kornia, source="torch", target=target)

    assert hasattr(kornia.geometry.transform.affwarp.affine, "__wrapped__")
    assert hasattr(kornia.geometry.transform.affine, "__wrapped__")
