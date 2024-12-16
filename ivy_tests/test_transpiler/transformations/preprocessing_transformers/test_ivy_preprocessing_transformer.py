import pytest
import gast
import ivy
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.preprocessing_transformer.ivy_preprocessing_transformer import (
    IvyCodePreProcessor,
)
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code


def transform_function(func):
    # Set up the configurations container
    container = ConfigurationsContainer()
    container.load_configurations(source="ivy", target="tensorflow")

    # Create BaseObjectLike from the function
    object_like = BaseObjectLike.from_object(func)
    root = gast.parse(object_like.source_code)

    # Instantiate the transformer and transform the object
    configuration = BaseTransformerConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    converter = IvyCodePreProcessor(root, transformer, configuration)
    converter.transform()

    # Return the transformed source code
    return ast_to_source_code(root).strip()


def test_visit_Assign_unpacking():
    def foo(x):
        N, H, W, C = x.shape

    transformed = transform_function(foo)
    assert "N, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]" in transformed


def test_visit_Assign_shape_attribute():
    def foo(_check_bounds_and_get_shape, x):
        shape = _check_bounds_and_get_shape(x).shape

    transformed = transform_function(foo)
    assert "shape = _check_bounds_and_get_shape(x)" in transformed


def test_visit_Assign_conversion():
    def foo(y):
        x = ivy.to_ivy(y)
        x2 = ivy.to_native(y)
        x3 = ivy.args_to_native(y)

    transformed = transform_function(foo)
    assert "x = y" in transformed
    assert "x2 = y" in transformed
    assert "x3 = y" in transformed


def test_visit_If_remove_elif():
    def foo(condition1):
        if condition1:
            x = 1
        elif ivy.is_ivy_array(y):
            y = 2
        else:
            z = 3

    transformed = transform_function(foo)
    assert "elif ivy.is_ivy_array(y):" not in transformed
    assert "y = 2" not in transformed
    assert "if condition1:" in transformed
    assert "x = 1" in transformed
    assert "z = 3" in transformed


if __name__ == "__main__":
    pytest.main([__file__])
