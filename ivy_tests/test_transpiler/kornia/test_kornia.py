import pytest
import ivy
import torch
import numpy as np
from ivy.transpiler import transpile
from ivy_tests.test_transpiler.kornia.helpers import (
    _nest_torch_tensor_to_new_framework,
    _check_allclose,
    _nest_array_to_numpy,
    download_image,
)


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
        "numpy",
    ],
)
def test_translate_kornia_distort_points_affine(target):
    ivy.set_backend(target)
    import kornia

    fn = kornia.geometry.distort_points_affine

    translated_fn = transpile(fn, source="torch", target=target)

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

    # Input 1:
    translated_args = _nest_torch_tensor_to_new_framework(trace_args, target)
    translated_kwargs = _nest_torch_tensor_to_new_framework(trace_kwargs, target)

    orig_res = fn(*trace_args, **trace_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )

    # Input 2:
    translated_args = _nest_torch_tensor_to_new_framework(test_args, target)
    translated_kwargs = _nest_torch_tensor_to_new_framework(test_kwargs, target)

    orig_res = fn(*test_args, **test_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
        "numpy",
    ],
)
def test_translate_kornia_rad2deg(target):
    ivy.set_backend(target)
    import kornia

    fn = kornia.geometry.rad2deg

    translated_fn = transpile(fn, source="torch", target=target)

    trace_args = (torch.tensor(3.1415926535),)
    trace_kwargs = {}
    test_args = (torch.rand(5, 3, 3),)
    test_kwargs = {}

    # Input 1:
    translated_args = _nest_torch_tensor_to_new_framework(trace_args, target)
    translated_kwargs = _nest_torch_tensor_to_new_framework(trace_kwargs, target)

    orig_res = fn(*trace_args, **trace_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )

    # Input 2:
    translated_args = _nest_torch_tensor_to_new_framework(test_args, target)
    translated_kwargs = _nest_torch_tensor_to_new_framework(test_kwargs, target)

    orig_res = fn(*test_args, **test_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
        "numpy",
    ],
)
def test_translate_kornia_distance_transform(target):
    ivy.set_backend(target)
    import kornia

    fn = kornia.contrib.distance_transform

    translated_fn = transpile(fn, source="torch", target=target)

    trace_args = (torch.randn(1, 1, 5, 5),)
    trace_kwargs = {"kernel_size": 3, "h": 0.35}
    test_args = (torch.randn(5, 1, 5, 5),)
    test_kwargs = {"kernel_size": 3, "h": 0.5}

    # Input 1:
    translated_args = _nest_torch_tensor_to_new_framework(
        trace_args, target, shallow=False
    )
    translated_kwargs = _nest_torch_tensor_to_new_framework(
        trace_kwargs, target, shallow=False
    )

    orig_res = fn(*trace_args, **trace_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )

    # Input 2:
    translated_args = _nest_torch_tensor_to_new_framework(
        test_args, target, shallow=False
    )
    translated_kwargs = _nest_torch_tensor_to_new_framework(
        test_kwargs, target, shallow=False
    )

    orig_res = fn(*test_args, **test_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
        "numpy",
    ],
)
def test_translate_kornia_match_adalam(target):
    ivy.set_backend(target)
    import kornia

    fn = kornia.feature.match_adalam

    translated_fn = transpile(fn, source="torch", target=target)

    trace_args = (
        torch.rand(3, 128),
        torch.rand(3, 128),
        torch.rand(1, 3, 2, 3),
        torch.rand(1, 3, 2, 3),
    )
    trace_kwargs = {"config": None, "hw1": None, "hw2": None}
    test_args = (
        torch.rand(5, 128),
        torch.rand(5, 128),
        torch.rand(1, 5, 2, 3),
        torch.rand(1, 5, 2, 3),
    )
    test_kwargs = {"config": None, "hw1": None, "hw2": None}
    # Input 1:
    translated_args = _nest_torch_tensor_to_new_framework(
        trace_args, target, shallow=False
    )
    translated_kwargs = _nest_torch_tensor_to_new_framework(
        trace_kwargs, target, shallow=False
    )

    orig_res = fn(*trace_args, **trace_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )

    # Input 2:
    translated_args = _nest_torch_tensor_to_new_framework(
        test_args, target, shallow=False
    )
    translated_kwargs = _nest_torch_tensor_to_new_framework(
        test_kwargs, target, shallow=False
    )

    orig_res = fn(*test_args, **test_kwargs)
    translated_res = translated_fn(*translated_args, **translated_kwargs)

    _check_allclose(
        _nest_array_to_numpy(orig_res),
        _nest_array_to_numpy(translated_res),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
        "numpy",
    ],
)
def test_kornia_load_image(target):
    ivy.set_backend(target)
    import kornia

    url = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"
    download_image(url)

    pt_load_image = kornia.io.load_image
    PTImageLoadType = kornia.io.ImageLoadType

    translated_load_image = transpile(
        kornia.io.load_image, source="torch", target=target
    )
    TranslatedImageLoadType = transpile(
        kornia.io.ImageLoadType, source="torch", target=target
    )

    pt_imag = pt_load_image("panda.jpg", PTImageLoadType.RGB32)
    translated_image = translated_load_image("panda.jpg", TranslatedImageLoadType.RGB32)

    _check_allclose(
        _nest_array_to_numpy(pt_imag),
        _nest_array_to_numpy(translated_image),
        tolerance=1e-2,
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_kornia_Boxes(target):
    ivy.set_backend(target)
    from kornia.geometry.boxes import Boxes

    # test `.from_tensor`
    boxes_xyxy = torch.as_tensor([[0, 3, 1, 4], [5, 1, 8, 4]])
    boxes = Boxes.from_tensor(boxes_xyxy, mode="xyxy")

    translated_boxes_xyxy = _nest_torch_tensor_to_new_framework(boxes_xyxy, target)
    TranslatedBoxes = transpile(Boxes, source="torch", target=target)
    translated_boxes = TranslatedBoxes.from_tensor(translated_boxes_xyxy, mode="xyxy")

    _check_allclose(
        _nest_array_to_numpy(boxes.data),
        _nest_array_to_numpy(translated_boxes.data),
        tolerance=1e-2,
    )

    # test `.to_mask`
    input = torch.tensor(
        [
            [
                [1.0, 1.0],
                [4.0, 1.0],
                [4.0, 3.0],
                [1.0, 3.0],
            ]
        ]
    )
    boxes = Boxes(input)
    res = boxes.to_mask(5, 5)

    translated_input = _nest_torch_tensor_to_new_framework(input, target)
    translated_boxes = TranslatedBoxes(translated_input)
    translated_res = translated_boxes.to_mask(5, 5)

    # TODO(haris): the logits are not allclose because of an inplace-update issue. See (https://github.com/kornia/kornia/blob/a48ed83117c8d92262c9aedc0cea5ad461f5c8da/kornia/geometry/boxes.py#L574)
    # uncomment this once the inplace update functionality has been implemented
    # _check_allclose(
    #     _nest_array_to_numpy(res),
    #     _nest_array_to_numpy(translated_res),
    #     tolerance=1e-2,
    # )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_kornia_ColorJiggle(target):
    ivy.set_backend(target)
    import kornia

    inputs = torch.ones(1, 3, 3, 3)
    aug = kornia.augmentation.ColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)
    inputs = torch.ones(1, 3, 3, 3)
    pt_sample = aug(inputs)

    TranslatedColorJiggle = transpile(
        kornia.augmentation.ColorJiggle, source="torch", target=target
    )
    translated_inputs = _nest_torch_tensor_to_new_framework(inputs, target)
    translated_aug = TranslatedColorJiggle(0.1, 0.1, 0.1, 0.1, p=1.0)
    translated_sample = translated_aug(translated_inputs)

    # round the values to the nearest integer as ColorJiggle samples from a prob distribution
    pt_sample = np.round(pt_sample)
    translated_sample = np.round(translated_sample)
    assert np.allclose(pt_sample, translated_sample, atol=1e-3)


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
    ],
)
def test_kornia_Se3_conflicting_method(target):
    import kornia

    quaternion_data = torch.tensor([1.0, 0.0, 0.0, 0.0])
    translation_data = torch.tensor([1.0, 1.0, 1.0])
    torch_quaternion = kornia.geometry.quaternion.Quaternion(quaternion_data)
    torch_se3 = kornia.geometry.liegroup.Se3(torch_quaternion, translation_data)

    TranspiledSe3 = transpile(
        kornia.geometry.liegroup.Se3, source="torch", target=target
    )
    TranspiledQuaternion = transpile(
        kornia.geometry.quaternion.Quaternion, source="torch", target=target
    )

    transpiled_translation = _nest_torch_tensor_to_new_framework(
        translation_data, target
    )
    transpiled_quaternion = TranspiledQuaternion(
        _nest_torch_tensor_to_new_framework(quaternion_data, target)
    )
    transpiled_se3 = TranspiledSe3(transpiled_quaternion, transpiled_translation)

    # Test .inverse()
    torch_inverse = torch_se3.inverse()
    transpiled_inverse = transpiled_se3.inverse()
    _check_allclose(
        _nest_array_to_numpy(torch_inverse.r.q.data),
        _nest_array_to_numpy(transpiled_inverse.r.q.data),
    )
    _check_allclose(
        _nest_array_to_numpy(torch_inverse.t),
        _nest_array_to_numpy(transpiled_inverse.t),
    )


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
        "jax",
        "numpy",
    ],
)
def test_mean_average_precision(target):
    # tests BaseNameCanonicalizer._rename_node case 5.
    import kornia

    args = (
        [torch.tensor([[100, 50, 150, 100.0]])],
        [torch.tensor([1.0])],
        [torch.tensor([0.7])],
        [torch.tensor([[100, 50, 150, 100.0]])],
        [torch.tensor([1.0])],
        2,
    )
    kwargs = {}

    translated_mean_average_precision = transpile(
        kornia.metrics.mean_average_precision, source="torch", target=target
    )

    torch_out = kornia.metrics.mean_average_precision(*args, **kwargs)
    transpiled_args = _nest_torch_tensor_to_new_framework(args, target)
    transpiled_kwargs = _nest_torch_tensor_to_new_framework(kwargs, target)
    transpiled_out = translated_mean_average_precision(
        *transpiled_args, **transpiled_kwargs
    )

    _check_allclose(
        _nest_array_to_numpy(torch_out),
        _nest_array_to_numpy(transpiled_out),
        tolerance=1e-6,
    )
