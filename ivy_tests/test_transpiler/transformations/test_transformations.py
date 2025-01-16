import pytest
import ivy
import gast
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.translations.data.object_like import (
    BaseObjectLike,
)
from ivy.transpiler.configurations_container import (
    ConfigurationsContainer,
)
from ivy.transpiler.utils.ast_utils import ast_to_source_code
from ivy_tests.test_transpiler.transformations.helpers import load_fn_from_str


def fn_w_list_comp(x):
    x = [i**2 if i % 2 == 0 else i**3 for i in x]
    x = [i // 2 for i in x if i % 2 == 0]
    return x


@pytest.mark.parametrize(
    "target",
    [
        "tensorflow",
    ],
)
def test_transform_fn_w_list_comp(target):
    ivy.set_backend(target)
    from ivy.transpiler.transformations.configurations.ivy_postprocessing_transformer_config import (
        IvyCodePostProcessorConfig,
    )
    from ivy.transpiler.transformations.transformers.postprocessing_transformer.ivy_to_tf_postprocessing_transformer import (
        IvyToTFCodePostProcessor,
    )

    # Set up the configurations container
    container = ConfigurationsContainer()
    container.load_configurations(source="ivy", target="tensorflow")

    # Parse the source code of the object to transform
    object_like = BaseObjectLike.from_object(fn_w_list_comp)
    root = gast.parse(object_like.source_code)

    # Instantiate the transformer and transform the object
    configuration = IvyCodePostProcessorConfig()

    # TODO: generalize this for other target frameworks
    processor = IvyToTFCodePostProcessor(
        root,
        Transformer(object_like, container.translator_configurations[0]),
        configuration,
    )
    processor.transform()

    # Verify the transformation ran successfully
    # TODO: Need to figure out a better way of statically judging
    # if the transformation ran successfully i.e. whether the source
    # code did transform or remained the same because just an output
    # comparison cannot determine that we arrived at the correct transformed
    # source code
    transformed_src = ast_to_source_code(root).strip()
    transformed_fn = load_fn_from_str(transformed_src, "fn_w_list_comp")
    out = fn_w_list_comp([1, 2, 3, 4, 5, 6])
    transformed_out = transformed_fn([1, 2, 3, 4, 5, 6])

    assert out == transformed_out
    assert object_like.source_code != transformed_src
