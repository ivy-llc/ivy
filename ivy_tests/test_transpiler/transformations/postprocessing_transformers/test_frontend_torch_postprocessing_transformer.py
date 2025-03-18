import pytest
import gast
import os
from ivy.functional.frontends import torch
import ivy
from ivy.transpiler.transformations.configurations.frontend_torch_postprocessing_transformer_config import (
    FrontendTorchCodePostProcessorConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.postprocessing_transformer.frontend_torch_postprocessing_transformer import (
    FrontendTorchCodePostProcessor,
)
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code


def transform_function(func):
    # Set up the configurations container
    container = ConfigurationsContainer()
    container.load_configurations(source="torch_frontend", target="ivy")

    # Create BaseObjectLike from the function
    object_like = BaseObjectLike.from_object(func)
    root = gast.parse(object_like.source_code)

    # Instantiate the transformer and transform the object
    configuration = FrontendTorchCodePostProcessorConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    converter = FrontendTorchCodePostProcessor(root, transformer, configuration)
    converter.transform()

    # Return the transformed source code
    return ast_to_source_code(root).strip()


# Test cases
def test_isinstance_torch_tensor():

    def check_types():
        x = ...  # placeholder for some value
        y = ...  # placeholder for some value
        z = ...  # placeholder for some value
        w = ...  # placeholder for some value

        Tensor = torch.Tensor

        class OtherType:
            pass

        if isinstance(x, Tensor):
            print("x is a Tensor")
        if isinstance(y, torch.Tensor):
            print("y is a torch.Tensor")
        if isinstance(z, (Tensor, OtherType)):
            print("z is either a Tensor or OtherType")
        if isinstance(w, (torch.Tensor, OtherType)):
            print("w is either a torch.Tensor or OtherType")

    transformed = transform_function(check_types)

    assert (
        "isinstance(x, (ivy.Array, ivy.Variable)):\n        print('x is a Tensor')"
        in transformed
    )
    assert (
        "isinstance(y, (ivy.Array, ivy.Variable)):\n        print('y is a torch.Tensor')"
        in transformed
    )
    assert (
        "isinstance(z, (ivy.Array, ivy.Variable, OtherType)):\n        print('z is either a Tensor or OtherType')"
        in transformed
    )
    assert (
        "isinstance(w, (ivy.Array, ivy.Variable, OtherType)):\n        print('w is either a torch.Tensor or OtherType')"
        in transformed
    )


def test_frontend_class_attributes():

    def check_types(x):
        x.data
        x.ivy_array
        x.ivy_array.data

    # not an ivy api so attributes should not be deleted
    transformed = transform_function(check_types)
    assert "x.ivy_array.data" in transformed
    assert "x.ivy_array" in transformed
    assert "x.data" in transformed

    # part of ivy api so attributes should  be deleted
    transformed = transform_function(torch.Tensor.item)
    assert "return tf.squeeze(arr)" in transformed

    transformed = transform_function(torch.Tensor.expm1_)
    assert "ret = torch_frontend.expm1(arr)" in transformed
    assert "arr = ivy.inplace_update(arr, ret)" in transformed


if __name__ == "__main__":
    pytest.main([__file__])
