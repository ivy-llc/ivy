import pytest
import ivy
import numpy as np
import torch
from ivy.transpiler import transpile
from ivy_tests.test_transpiler.translations.helpers import (
    get_torch_cnn_model_inputs,
)
import tensorflow as tf
import os


def get_target_list():
    # Add targets here
    return ["tensorflow", "jax"]


# Note: Keep this test at the top of the file
# to simulate that no transpilation has
# taken place in the outputs directory
@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_simple_transform(target):
    ivy.set_backend(target)
    from ivy.transpiler.examples.SimpleModelNoConv.s2s_simplemodel import (
        SimpleModel,
    )

    # Keep `reuse_existing=False`
    TranslatedSimpleModel = transpile(
        SimpleModel, source="torch", target=target, reuse_existing=False
    )

    # initialize the models
    model = SimpleModel()
    translated_model = TranslatedSimpleModel()

    # create inputs and build layers
    x_np = np.random.rand(32, 28 * 28).astype("float32")
    x_torch = torch.from_numpy(x_np)
    x_translated = ivy.native_array(x_np, dtype=ivy.float32)
    translated_model(x_translated)

    # Sync models and assert logits to be all close
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(model, translated_model)

    if target == "tensorflow":
        # Clone model
        cloned_model = tf.keras.models.clone_model(translated_model)

        # build layers before calling set_weights
        cloned_model(x_translated)
        cloned_model.set_weights(translated_model.get_weights())
        ivy.sync_models(model, cloned_model)
    else:
        # TODO: add cloning logic for JAX backend
        cloned_model = translated_model
    # Run inference
    model.eval()
    translated_model.eval()
    cloned_model.eval()

    out = model(x_torch)
    translated_out = translated_model(x_translated)
    cloned_out = cloned_model(x_translated)

    assert np.allclose(out.detach().numpy(), ivy.to_numpy(translated_out), atol=1e-2)
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(cloned_out), atol=1e-2)


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_alexnet(target):
    ivy.set_backend(target)
    from ivy.transpiler.examples.AlexNet.s2s_alexnet import AlexNet

    TranslatedAlexNet = transpile(AlexNet, source="torch", target=target)

    # Instantiate
    model = AlexNet(10)
    translated_model = TranslatedAlexNet(10)

    # build layers
    x, x_translated = get_torch_cnn_model_inputs("alexnet", target)
    translated_model(x_translated)

    # Sync models and assert logits to be all close
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(model, translated_model)

    if target == "tensorflow":
        # Clone model
        cloned_model = tf.keras.models.clone_model(translated_model)

        # build layers before calling set_weights
        cloned_model(x_translated)
        cloned_model.set_weights(translated_model.get_weights())
        ivy.sync_models(model, cloned_model)
    else:
        # TODO: add cloning logic for JAX backend
        cloned_model = translated_model

    x, x_translated = get_torch_cnn_model_inputs("alexnet", target)

    # run inference
    model.eval()
    translated_model.eval()
    cloned_model.eval()

    out = model(x)
    translated_out = translated_model(x_translated)
    cloned_out = cloned_model(x_translated)

    # TODO: Need to ensure the `atol` can be decreased to 1e-3/1e-4
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(translated_out), atol=0.05)
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(cloned_out), atol=0.05)


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_unet(target):
    ivy.set_backend(target)
    from ivy.transpiler.examples.UNet.s2s_unet import UNet

    TranslatedUNet = transpile(UNet, source="torch", target=target)

    # Instantiate
    model = UNet(3, 10)
    translated_model = TranslatedUNet(3, 10)

    # build layers
    x, x_translated = get_torch_cnn_model_inputs("unet", target)
    translated_model(x_translated)

    # Sync models and assert logits to be all close
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(model, translated_model)

    if target == "tensorflow":
        # Clone model
        cloned_model = tf.keras.models.clone_model(translated_model)

        # build layers before calling set_weights
        cloned_model(x_translated)
        cloned_model.set_weights(translated_model.get_weights())
        ivy.sync_models(model, cloned_model)
    else:
        # TODO: add cloning logic for JAX backend
        cloned_model = translated_model

    x, x_translated = get_torch_cnn_model_inputs("unet", target)

    # run inference
    model.eval()
    translated_model.eval()
    cloned_model.eval()
    out = model(x)
    translated_out = translated_model(x_translated)
    cloned_out = cloned_model(x_translated)

    assert np.allclose(out.detach().numpy(), ivy.to_numpy(translated_out), atol=1e-3)
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(cloned_out), atol=1e-3)


@pytest.mark.parametrize("target", get_target_list())
def test_translate_RelaxedBernoulli(target):
    ivy.set_backend(target)
    from torch.distributions import RelaxedBernoulli

    TranslatedRelaxedBernoulli = transpile(
        RelaxedBernoulli, source="torch", target=target
    )

    # Instantiate
    np_temp = np.array([2.2]).astype("float32")
    np_probs = np.array([0.00001, 0.00002, 0.99999, 0.99999]).astype("float32")
    pt_dist = RelaxedBernoulli(torch.from_numpy(np_temp), torch.from_numpy(np_probs))
    translated_dist = TranslatedRelaxedBernoulli(
        ivy.native_array(np_temp), ivy.native_array(np_probs)
    )

    # Assert outputs to be allclose
    pt_logts = pt_dist.logits.detach().numpy()
    translated_logts = ivy.to_numpy(translated_dist.logits)
    assert np.allclose(pt_logts, translated_logts, atol=1e-3)

    # Random sampling
    pt_sample = pt_dist.sample().detach().numpy()
    translated_sample = ivy.to_numpy(translated_dist.sample())

    # round the values to the nearest integer as we are random sampling from a prob distribution
    pt_sample = np.round(pt_sample)
    translated_sample = np.round(translated_sample)
    assert np.allclose(pt_sample, translated_sample, atol=1e-3)


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_swin2sr(target):
    ivy.set_backend(target)
    from ivy.transpiler.examples.Swin2SR.s2s_swin2sr import Swin2SR

    upscale = 8
    window_size = 16
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size

    TranslatedSwin2SR = transpile(Swin2SR, source="torch", target=target)

    # instantiate
    translated_model = TranslatedSwin2SR(
        upscale=2,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.0,
        depths=[
            2,
            2,
        ],
        embed_dim=60,
        num_heads=[
            2,
            2,
        ],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
    )

    model = Swin2SR(
        upscale=2,
        img_size=(height, width),
        window_size=window_size,
        img_range=1.0,
        depths=[
            2,
            2,
        ],
        embed_dim=60,
        num_heads=[
            2,
            2,
        ],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
    )

    # create inputs and build layers
    x_np = np.random.rand(1, 3, height, width).astype("float32")
    x_torch = torch.from_numpy(x_np)
    x_translated = ivy.native_array(x_np, dtype=ivy.float32)
    translated_model(x_translated)

    # Sync models and assert logits to be all close
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(model, translated_model)

    if target == "tensorflow":
        # Clone model
        cloned_model = tf.keras.models.clone_model(translated_model)

        # build layers before calling set_weights
        cloned_model(x_translated)
        cloned_model.set_weights(translated_model.get_weights())
        ivy.sync_models(model, cloned_model)
    else:
        # TODO: add cloning logic for JAX backend
        cloned_model = translated_model

    # Run inference
    model.eval()
    translated_model.eval()
    cloned_model.eval()
    out = model(x_torch)
    translated_out = translated_model(x_translated)
    cloned_out = cloned_model(x_translated)

    # TODO: Need to ensure the `atol` can be decreased to 1e-3/1e-4
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(translated_out), atol=0.5)
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(cloned_out), atol=0.5)


# TODO: This test will start passing once `ivy_repo` is updated with `erfinv` and `erfinv_` is added
# to torch frontend Tensor class


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_distilled_vit(target):
    ivy.set_backend(target)
    from ivy.transpiler.examples.DistilledVisionTransformer.s2s_distilledvit import (
        DistilledVisionTransformer,
    )

    default_cfg = {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth",
        "num_classes": 21843,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.5, 0.5, 0.5),
        "std": (0.5, 0.5, 0.5),
        "first_conv": "patch_embed.proj",
        "classifier": "head",
    }
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-1]

    model_kwargs = dict(
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        representation_size=768,
    )
    num_classes = model_kwargs.pop("num_classes", default_num_classes)
    img_size = model_kwargs.pop("img_size", default_img_size)
    _ = model_kwargs.pop("representation_size", None)

    TranslatedDistilledVisionTransformer = transpile(
        DistilledVisionTransformer, source="torch", target=target
    )

    # instantiate
    translated_model = TranslatedDistilledVisionTransformer(
        img_size=img_size,
        num_classes=num_classes,
        representation_size=None,
        **model_kwargs,
    )
    translated_model.default_cfg = default_cfg

    model = DistilledVisionTransformer(
        img_size=img_size,
        num_classes=num_classes,
        representation_size=None,
        **model_kwargs,
    )
    model.default_cfg = default_cfg

    # create inputs and build layers
    x_np = np.random.randn(32, 3, 224, 224).astype("float32")
    x_torch = torch.tensor(x_np, dtype=torch.float32)
    x_translated = ivy.native_array(x_np, dtype=ivy.float32)
    translated_model(x_translated)

    # Sync models and assert logits to be all close
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(model, translated_model)

    if target == "tensorflow":
        # Clone model
        cloned_model = tf.keras.models.clone_model(translated_model)

        # build layers before calling set_weights
        cloned_model(x_translated)
        cloned_model.set_weights(translated_model.get_weights())
        ivy.sync_models(model, cloned_model)
    else:
        # TODO: add cloning logic for JAX backend
        cloned_model = translated_model

    # Run inference
    model.eval()
    translated_model.eval()
    cloned_model.eval()
    out = model(x_torch)
    translated_out = translated_model(x_translated)
    cloned_out = cloned_model(x_translated)

    assert np.allclose(out.detach().numpy(), ivy.to_numpy(translated_out), atol=1e-3)
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(cloned_out), atol=1e-3)


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_cflow(target):
    ivy.set_backend(target)
    from ivy.transpiler.examples.CFLow.helpers import get_args
    from ivy.transpiler.examples.CFLow.s2s_cflow import (
        load_decoder_arch,
        load_encoder_arch,
    )

    # get the defualt configs
    c = get_args()
    L = c.pool_layers  # number of pooled layers
    c.condition_vec = 128
    c.clamp_alpha = 1.9  # see paper equation 2 for explanation
    c.condition_vec = 128
    c.dropout = 0.0
    c.device = "cpu"

    translated_enc_arch = transpile(load_encoder_arch, source="torch", target=target)
    translated_dec_arch = transpile(load_decoder_arch, source="torch", target=target)
    pt_enc_arch = load_encoder_arch
    pt_dec_arch = load_decoder_arch

    # build the encoders
    translated_encoder, pool_layers, pool_dims = translated_enc_arch(c, L)
    pt_encoder, pool_layers, pool_dims = pt_enc_arch(c, L)
    x_np = np.random.rand(1, 3, 256, 256).astype("float32")
    x_translated = ivy.native_array(x_np)
    x_torch = torch.from_numpy(x_np)
    translated_encoder(x_translated)

    # build the decoders
    translated_decoders = [translated_dec_arch(c, pool_dim) for pool_dim in pool_dims]
    pt_decoders = [pt_dec_arch(c, pool_dim) for pool_dim in pool_dims]
    x_ep = np.random.rand(256, 512).astype("float32")
    x_cp = np.random.rand(256, 128).astype("float32")
    x_ep_translated = ivy.native_array(x_ep)
    x_cp_translated = ivy.native_array(x_cp)
    x_ep_pt = torch.from_numpy(x_ep)
    x_cp_pt = torch.from_numpy(x_cp)
    translated_decoders[0](x_ep_translated, [x_cp_translated])

    # sync the encoder
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(pt_encoder, translated_encoder)

    if target == "tensorflow":
        # clone encoder
        cloned_encoder = tf.keras.models.clone_model(translated_encoder)

        # build layers before calling set_weights
        cloned_encoder(x_translated)
        cloned_encoder.set_weights(translated_encoder.get_weights())
        ivy.sync_models(pt_encoder, cloned_encoder)
    else:
        # TODO: add cloning logic for JAX backend
        cloned_encoder = translated_encoder

    # run inference
    pt_encoder.eval()
    translated_encoder.eval()
    cloned_encoder.eval()
    orig_out = pt_encoder(x_torch)
    translated_out = translated_encoder(x_translated)
    cloned_out = cloned_encoder(x_translated)

    assert np.allclose(
        orig_out.detach().numpy(), ivy.to_numpy(translated_out), atol=1e-2
    )
    assert np.allclose(orig_out.detach().numpy(), ivy.to_numpy(cloned_out), atol=1e-2)

    # sync the decoder
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(pt_decoders[0], translated_decoders[0])

    # NOTE: Cannot directly clone the decoder as it contains a dynamic architecture
    # where the structure of the internal layers is unknown. The only way
    # to clone such a model is for the user to write their own custom cloning
    # function that can replicate the model's creation.
    # Reference: (https://github.com/unifyai/tracer-transpiler/blob/source2source/source_to_source_translator/examples/CFLow/helpers.py#L32)

    # clone decorders
    # cloned_decoders = []
    # for transl_model in translated_decoders:
    #     cloned_decoder = tf.keras.models.clone_model(transl_model)
    #     cloned_decoder.set_weights(transl_model.get_weights())
    #     cloned_decoders.append(cloned_decoder)

    # run inference
    pt_decoders[0].eval()
    translated_decoders[0].eval()
    # cloned_decoders[0].eval()
    z_tf, log_jac_det_translated = translated_decoders[0](
        x_ep_translated, [x_cp_translated]
    )
    # z_translated_cloned, log_jac_det_translated_cloned = cloned_decoders[0](x_ep_tf, [x_cp_tf])
    z_pt, log_jac_det_pt = pt_decoders[0](x_ep_pt, [x_cp_pt])

    assert np.allclose(z_pt.detach().numpy(), ivy.to_numpy(z_tf), atol=1e-2)
    # assert np.allclose(z_pt.detach().numpy(), ivy.to_numpy(z_translated_cloned), atol=1e-2)
    assert np.allclose(
        log_jac_det_pt.detach().numpy(), ivy.to_numpy(log_jac_det_translated), atol=1e-2
    )
    # assert np.allclose(
    #     log_jac_det_pt.detach().numpy(), ivy.to_numpy(log_jac_det_translated_cloned), atol=1e-2
    # )


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_glpdepth(target):
    ivy.set_backend(target)
    from ivy.transpiler.examples.GLPDepth.s2s_glpdepth import GLPDepth

    TranslatedGLPDepth = transpile(GLPDepth, source="torch", target=target)

    # initialize the models
    model = GLPDepth()
    translated_model = TranslatedGLPDepth()
    # create inputs and build layers
    x_np = np.random.rand(1, 3, 416, 416).astype("float32")
    x_torch = torch.from_numpy(x_np)
    x_translated = ivy.native_array(x_np, dtype=ivy.float32)
    translated_model(x_translated)

    # Sync models and assert logits to be all close
    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(model, translated_model)

    if target == "tensorflow":
        # Clone model
        cloned_model = tf.keras.models.clone_model(translated_model)

        # build layers before calling set_weights
        cloned_model(x_translated)
        cloned_model.set_weights(translated_model.get_weights())
        ivy.sync_models(model, cloned_model)
    else:
        # TODO: add cloning logic for JAX backend
        cloned_model = translated_model

    # Run inference
    model.eval()
    translated_model.eval()
    cloned_model.eval()

    out = model(x_torch)["pred_d"]
    translated_out = translated_model(x_translated)["pred_d"]
    cloned_out = cloned_model(x_translated)["pred_d"]

    # TODO: figure out the issue of the high atol for these models
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(translated_out), atol=1e-1)
    assert np.allclose(out.detach().numpy(), ivy.to_numpy(cloned_out), atol=1e-1)


# TODO: there are some missing frontends here, causing torch code to be present
# in the translated model - though the test still passes


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_lstm(target):
    ivy.set_backend(target)

    translated_LSTM = transpile(torch.nn.LSTM, source="torch", target=target)

    model = torch.nn.LSTM(2, 2, 1)
    translated_model = translated_LSTM(2, 2, 1)

    orig_x = torch.rand((20, 32, 2))
    translated_x = ivy.native_array(orig_x.detach().numpy())

    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(model, translated_model)

    model.eval()
    translated_model.eval()

    out = model(orig_x)
    translated_out = translated_model(translated_x)
    assert np.allclose(
        out[0].detach().numpy(), ivy.to_numpy(translated_out[0]), atol=1e-4
    )
    assert np.allclose(
        out[1][0].detach().numpy(), ivy.to_numpy(translated_out[1][0]), atol=1e-4
    )


@pytest.mark.parametrize("target", get_target_list())
def test_translate_torch_inplace(target):
    ivy.set_backend(target)

    def inplace_fn():
        M = torch.zeros(4, 16)
        M[..., -1].fill_(1)
        return M

    translated_inplace_fn = transpile(inplace_fn, source="torch", target=target)

    # Instantiate
    pt_out = inplace_fn()
    translated_out = translated_inplace_fn()

    # Assert outputs to be allclose
    pt_logts = pt_out.detach().numpy()
    translated_logts = ivy.to_numpy(translated_out)
    assert np.allclose(pt_logts, translated_logts)
