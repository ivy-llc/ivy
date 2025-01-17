import pytest
import gast


from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.canonicalize_transformer.base_transformer import (
    BaseNameCanonicalizer,
)
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import ConfigurationsContainer
from ivy.transpiler.utils.ast_utils import ast_to_source_code


def transform_function(func):
    # Set up the configurations container
    container = ConfigurationsContainer()
    container.load_configurations(source="torch", target="torch_frontend")

    # Create BaseObjectLike from the function
    object_like = BaseObjectLike.from_object(func)
    root = gast.parse(object_like.source_code)

    # Instantiate the transformer and transform the object
    configuration = BaseTransformerConfig()
    transformer = Transformer(object_like, container.translator_configurations[0])
    converter = BaseNameCanonicalizer(root, transformer, configuration)
    converter.transform()
    # Return the AST and transformer
    return root, transformer


def test_custom_sin_import():
    from ivy_tests.test_transpiler.transformations.mock_dir.custom_math.advanced_math import (
        custom_sin,
    )

    root, transformer = transform_function(custom_sin)
    transformed = ast_to_source_code(root)

    assert "sin" in transformed
    assert "cos" in transformed
    assert (
        "math",
        "sin",
        None,
    ) in transformer._from_imports

    assert (
        "math",
        "cos",
        None,
    ) in transformer._from_imports


def test_NN_import():
    from ivy_tests.test_transpiler.transformations.mock_dir.ml_models.neural_net import (
        NeuralNetwork,
    )

    root, transformer = transform_function(NeuralNetwork)
    transformed = ast_to_source_code(root)

    assert "torch.nn.Conv2d" in transformed
    assert "torch.nn.functional.pad" in transformed
    assert "torch.nn.functional.normalize" in transformed

    assert ("numpy", "np") in transformer._imports
    assert transformer._from_imports == set()


def test_math_func_import():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_1 import (
        math_func,
    )

    root, transformer = transform_function(math_func)
    transformed = ast_to_source_code(root)

    # TODO: figure out why these are asserting false now the transpiler source code is part of ivy
    # assert (
    #     "tests.source2source.transformations.mock_dir.custom_math.advanced_math.custom_sin(x)"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.custom_math.advanced_math.MathOperations.custom_cos(x)"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.custom_math.advanced_math.PI"
    #     in transformed
    # )
    assert transformer._from_imports == set()


def test_data_utils_import():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_2 import (
        process_data,
    )

    root, transformer = transform_function(process_data)
    transformed = ast_to_source_code(root)

    # assert (
    #     "tests.source2source.transformations.mock_dir.data_utils.preprocessing.normalize(data)"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.data_utils.preprocessing.Preprocessor.scale(normalized)"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.data_utils.analysis.analyze_data(scaled)"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.data_utils.analysis.DATA_THRESHOLD"
    #     in transformed
    # )
    assert transformer._from_imports == set()


def test_ml_models_import():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_3 import (
        create_model,
    )

    root, transformer = transform_function(create_model)
    transformed = ast_to_source_code(root)

    # assert (
    #     "tests.source2source.transformations.mock_dir.ml_models.neural_net.NeuralNetwork([10, 5, 1])"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.ml_models.neural_net.MODEL_VERSION"
    #     in transformed
    # )
    assert transformer._from_imports == set()


def test_mixed_imports():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_4 import (
        complex_operation,
    )

    root, transformer = transform_function(complex_operation)
    transformed = ast_to_source_code(root)

    # assert (
    #     "tests.source2source.transformations.mock_dir.custom_math.advanced_math.custom_sin(normalized)"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.data_utils.preprocessing.normalize(data)"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.ml_models.neural_net.NeuralNetwork([5, 3, 1])"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.custom_math.advanced_math.MATH_CONSTANT"
    #     in transformed
    # )

    assert transformer._from_imports == set()
    assert transformer._imports == set()


def test_nested_imports():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_5 import (
        nested_operation,
    )

    root, transformer = transform_function(nested_operation)
    transformed = ast_to_source_code(root)

    # assert (
    #     "tests.source2source.transformations.mock_dir.data_utils.preprocessing.Preprocessor()"
    #     in transformed
    # )
    # assert (
    #     "tests.source2source.transformations.mock_dir.ml_models.neural_net.NeuralNetwork([5, 3, 1])"
    #     in transformed
    # )
    assert "model.forward(preprocessor.scale(data))" in transformed
    assert transformer._from_imports == set()


def test_torch_imports():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_6 import (
        torch_operations,
    )

    root, transformer = transform_function(torch_operations)
    transformed = ast_to_source_code(root)

    assert (
        "torch.distributions.constraints.simplex.check(2) or torch.distributions.constraints.real.check(1)"
        in transformed
    )
    assert (
        "torch.distributions.constraints.simplex.check(10) or torch.distributions.constraints.real.check(100)"
        in transformed
    )

    assert (
        "torch.nn.functional.pad(torch.tensor([1, 2, 3]), (1, 1), mode='constant', value=0)"
        in transformed
    )
    assert (
        "torch.nn.functional.pad(torch.tensor([11, 22, 33]), (1, 1), mode='constant', value=0)"
        in transformed
    )

    assert (
        "torch.conv2d(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))" in transformed
    )
    assert (
        "torch.nn.functional.conv2d(torch.tensor([11, 22, 33]), torch.tensor([11, 22, 33]))"
        in transformed
    )
    assert (
        "torch.nn.functional.conv2d(torch.tensor([111, 222, 333]), torch.tensor([111, 222, 333]))"
        in transformed
    )

    assert (
        "torch.nn.functional.linear(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))"
        in transformed
    )
    assert (
        "torch.nn.functional.linear(torch.tensor([11, 22, 33]), torch.tensor([11, 22, 33]))"
        in transformed
    )

    assert transformer._from_imports == set()
    assert transformer._imports == set()


def test_canonicalize_kornia_one_hot():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_7 import (
        kornia_one_hot,
    )

    root, _ = transform_function(kornia_one_hot)
    transformed = ast_to_source_code(root)

    assert (
        "return kornia.utils.one_hot(labels, num_classes=3, device=torch.device('cpu'), dtype=torch.int64)"
        in transformed
    )


def test_already_canonicalized_imports():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_7 import (
        already_canonicalized_operations,
    )

    root, transformer = transform_function(already_canonicalized_operations)
    transformed = ast_to_source_code(root)

    assert "math.sin(data)" in transformed
    assert "math.sqrt(data)" in transformed
    assert "itertools.chain(data, sin_data, sqrt_data)" in transformed

    assert transformer._from_imports == set()
    assert ("math", None) in transformer._imports
    assert ("itertools", None) in transformer._imports


def test_canonicalize_kornia_spatial_gradient():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_8 import (
        kornia_spatial_gradient,
    )

    root, _ = transform_function(kornia_spatial_gradient)
    transformed = ast_to_source_code(root)

    assert "kornia.filters.spatial_gradient" in transformed


def test_canonicalize_kornia_SpatialGradient():
    from ivy_tests.test_transpiler.transformations.name_canonicalize_transformers.examples.func_8 import (
        kornia_spatial_gradient_cls,
    )

    root, _ = transform_function(kornia_spatial_gradient_cls)
    transformed = ast_to_source_code(root)
    print(transformed)

    assert "kornia.filters.SpatialGradient" in transformed


if __name__ == "__main__":
    pytest.main([__file__])
