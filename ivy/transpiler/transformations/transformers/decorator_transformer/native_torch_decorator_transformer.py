# global

# local
from ivy.transpiler.transformations.configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ivy.transpiler.transformations.transformer import Transformer
from ivy.transpiler.transformations.transformers.decorator_transformer.base_transformer import (
    BaseDecoratorRemover,
)


class NativeTorchDecoratorRemover(BaseDecoratorRemover):
    """
    A subclass to remove certain torch decorators from a given function.
    """

    def __init__(
        self,
        root,
        transformer: Transformer,
        configuration: BaseTransformerConfig,
    ):
        self.root = root
        self.transformer = transformer
        self.configuration = configuration
        # TODO: Move to the respective config file
        self.disallowed_decorators = set(
            [
                "torch.no_grad",
                "torch.jit.ignore",
                "torch.inference_mode",
                "torch.cuda.amp.custom_fwd",
                "_copy_to_script_wrapper",
                "replace_return_docstrings",
                "torch._jit_internal._overload_method",
                "add_code_sample_docstrings",
            ]
        )
