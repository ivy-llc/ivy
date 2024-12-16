import inspect
import math
import textwrap
import pytest
import gast
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.inplace_update_transformer.base_transformer import (
    BaseInplaceUpdateTransformer,
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
    configuration = BaseTransformerConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    inplace_update_transformer = BaseInplaceUpdateTransformer(
        root, transformer, configuration
    )
    inplace_update_transformer.transform()

    # Return the transformed source code
    return ast_to_source_code(root).strip()


# Test cases
def test_inplace_update_methods():

    ivy_erf__frnt_ = lambda x: x
    ivy_fill__bknd_ = lambda x: x

    def example(x):
        ivy_erf__frnt_(x, -1.0)
        ivy_fill__bknd_(x, 1.0)
        return x

    transformed = transform_function(example)

    assert "x = ivy_erf__frnt_(x, -1.0)" in transformed
    assert "x = ivy_fill__bknd_(x, 1.0)" in transformed


def test_inplace_update_methods_with_subscripts():

    ivy_erf__frnt_ = lambda x: x
    ivy_add__frnt_ = lambda x: x
    ivy_fill__bknd_ = lambda x: x

    def example(x):
        ivy_erf__frnt_(x[..., -10], -1.0)
        ivy_add__frnt_(x[0], 5)
        ivy_fill__bknd_(x[..., 1:5, 0], 1.0)
        return x

    transformed = transform_function(example)

    assert "x[..., -10] = ivy_erf__frnt_(x[..., -10], -1.0)" in transformed
    assert "x[0] = ivy_add__frnt_(x[0], 5)" in transformed
    assert "x[..., 1:5, 0] = ivy_fill__bknd_(x[..., 1:5, 0], 1.0)" in transformed


def test_inplace_update_functions():

    ivy_normal__frnt_ = lambda x: x
    ivy_random__frnt = lambda x: x

    def example(x, bound=3, fan_out=3, generator=None):
        ivy_normal__frnt_(x, 0, math.sqrt(2.0 / fan_out))
        ivy_random__frnt(x, -bound, bound, generator=generator)
        return x

    transformed = transform_function(example)

    assert "x = ivy_normal__frnt_(x, 0, math.sqrt(2.0 / fan_out))" in transformed
    assert "x = ivy_random__frnt(x, -bound, bound, generator=generator)" in transformed


def test_inplace_update_functions_with_attributes():

    ivy_kaiming_uniform_ = lambda x: x
    ivy_uniform_ = lambda x: x
    Translated_ones_ = lambda x: x

    def example(self, x, bound=3):
        ivy_kaiming_uniform_(self.weight, a=math.sqrt(5))
        ivy_uniform_(self.bias, -bound, bound)
        Translated_ones_(self.weight)
        return x

    transformed = transform_function(example)

    assert (
        "self.weight = ivy_kaiming_uniform_(self.weight, a=math.sqrt(5))" in transformed
    )
    assert "self.bias = ivy_uniform_(self.bias, -bound, bound)" in transformed


def test_inplace_update_attributes():

    def example(self):
        self.num_batches_tracked.add_(1)
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    transformed = transform_function(example)

    assert "self.num_batches_tracked = self.num_batches_tracked.add_(1)" in transformed
    assert "self.running_mean = self.running_mean.zero_()" in transformed
    assert "self.running_var = self.running_var.fill_(1)" in transformed
    assert "self.num_batches_tracked = self.num_batches_tracked.zero_()" in transformed


def test_inplace_update_in_return():

    ivy_random__frnt = lambda x: x

    def example(self, tensor, bound=3, generator=None):
        self.num_batches_tracked.add_(1)
        return ivy_random__frnt(tensor, -bound, bound, generator=generator)

    transformed = transform_function(example)

    assert "self.num_batches_tracked = self.num_batches_tracked.add_(1)" in transformed
    assert (
        "return ivy_random__frnt(tensor, -bound, bound, generator=generator)"
        in transformed
    )


def test_no_inplace_update():

    ivy_full_frnt = lambda x: x

    def example(z):
        z.update()
        y = z.add()
        y = ivy_full_frnt(1.0)
        return y.pow(3)

    transformed = transform_function(example)

    # Parse the source codes into AST
    transformed_tree = gast.parse(transformed)
    original_tree = gast.parse(textwrap.dedent(inspect.getsource(example).strip()))

    # Compare ASTs
    def compare_ast(transformed_tree, original_tree):
        return gast.dump(transformed_tree) == gast.dump(original_tree)

    # Assert the two ASTs are equivalent
    assert compare_ast(transformed_tree, original_tree)


if __name__ == "__main__":
    pytest.main([__file__])
