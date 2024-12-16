from datasets import load_dataset
import ivy
import numpy as np
from ivy.transpiler import transpile
import torch
from transformers import (
    AutoImageProcessor,
    Swin2SRModel,
    Swin2SRConfig,
)
import os
import pytest

from helpers import _backend_compile, _target_to_simplified, sync_models_HF_torch_to_tf


@pytest.mark.parametrize(
    "target_framework, backend_compile",
    [
        ("tensorflow", False),
        ("jax", False),
        # ("tensorflow", True),
    ],
)
def test_Swin2SR(target_framework, backend_compile):
    ivy.set_backend(target_framework)
    dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
    image = dataset["test"]["image"][0]
    image_processor = AutoImageProcessor.from_pretrained(
        "caidas/swin2SR-classical-sr-x2-64"
    )

    # modify the config to avoid OOM issues
    swin2sr_config = Swin2SRConfig()
    swin2sr_config.embed_dim = 2
    swin2sr_config.depths = [2, 2]
    swin2sr_config.num_heads = [2, 2]

    torch_model = Swin2SRModel.from_pretrained(
        "caidas/swin2SR-classical-sr-x2-64",
        config=swin2sr_config,
        ignore_mismatched_sizes=True,
    )
    torch_inputs = image_processor(image, return_tensors="pt")
    with torch.no_grad():
        torch_outputs = torch_model(**torch_inputs)
    torch_last_hidden_states = torch_outputs.last_hidden_state

    TranslatedSwin2SRModel = transpile(
        Swin2SRModel, source="torch", target=target_framework
    )
    if target_framework == "tensorflow":
        # TODO: fix the issue with from_pretrained not working due to name-mismatch b/w PT model and translated TF model
        translated_model = TranslatedSwin2SRModel.from_pretrained(
            "caidas/swin2SR-classical-sr-x2-64",
            from_pt=True,
            config=swin2sr_config,
            ignore_mismatched_sizes=True,
        )
    else:
        # TODO: fix the from_pretrained issue with FlaxPretrainedModel class.
        translated_model = TranslatedSwin2SRModel(swin2sr_config)

    os.environ["USE_NATIVE_FW_LAYERS"] = "true"
    os.environ["APPLY_TRANSPOSE_OPTIMIZATION"] = "true"
    ivy.sync_models(torch_model, translated_model)

    if backend_compile:
        translated_model = _backend_compile(translated_model, target_framework)

    transpiled_inputs = image_processor(
        image, return_tensors=_target_to_simplified(target_framework)
    )
    transpiled_outputs = translated_model(**transpiled_inputs)
    transpiled_last_hidden_states = transpiled_outputs.last_hidden_state

    assert np.allclose(
        torch_last_hidden_states.numpy(),
        ivy.to_numpy(transpiled_last_hidden_states),
        atol=1e-3,
    )
