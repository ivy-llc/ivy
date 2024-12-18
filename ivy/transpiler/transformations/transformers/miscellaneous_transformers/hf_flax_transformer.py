# global
from ast import FunctionDef
import gast
from typing import Any

# local
from ...configurations.base_transformer_config import (
    BaseTransformerConfig,
)
from ...transformer import Transformer
from ..base_transformer import (
    BaseTransformer,
)
from ....utils.api_utils import is_submodule_of
from ....utils.ast_utils import ast_to_source_code


class HFPretrainedFlaxTransformer(BaseTransformer):
    """
    This class modifies the behavior of models inheriting from `FlaxPreTrainedModel`
    to make them compatible with `flax.nnx.Module`. The changes ensure that the model
    can be properly initialized using the `from_pretrained` method.
    """

    def __init__(
        self, root, transformer: Transformer, configuration: BaseTransformerConfig
    ) -> None:
        self.root = root
        self.transformer = transformer
        self.configuration = configuration

    def transform(self):
        if is_submodule_of(self.transformer.object_like, "PreTrainedModel"):
            self.visit(self.root)

    def visit_FunctionDef(self, node):
        if node.name == "__init__":
            # Fix 1: Inside the __init__, remove the call to super().__init__(config). This is because
            # FlaxPreTrainedModel.__init__ expects a different signature.

            # Fix 2: Inside the __init__, modify `self.config = config` to be `self._config = config`
            # because `config` is a read-only property of FlaxPreTrainedModel.

            for n in gast.walk(node):
                if (
                    isinstance(n, gast.Assign)
                    and isinstance(n.targets[0], gast.Attribute)
                    and n.targets[0].attr == "config"
                ):
                    n.targets[0].attr = "_config"
                elif (
                    isinstance(n, gast.Expr)
                    and isinstance(n.value, gast.Call)
                    and ast_to_source_code(n.value).strip()
                    == "super().__init__(config)"
                ):
                    node.body.remove(n)

        self.generic_visit(node)
        return node

    """
    -------NOTES--------

    # Fix 3: modify the `__init__` signature to match the one used by
    # FlaxPreTrainedModel.__init__, so that the model can be instantiated properly.
    def __init__(
        self,
        config,
        input_shape=(1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
    ):
        # Call the custom initialization function
        self.__flax_init(config, self, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    # Fix 4: add the following method for custom initialization to bypass
    # FlaxPreTrainedModel's assumption of using flax.linen.Module and instead
    # allow for initialization using flax.nnx.Module.
    def __flax_init(
        self,
        config,
        module,
        input_shape=(1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
    ):
        def _unflatten_dict_and_rename_params(flattened_dict):
            nested_dict = {}

            for key, value in flattened_dict.items():
                keys = key.split('.')

                if value is None:
                    continue

                # Rename 'weight' to 'kernel' in the last part of the key
                # TODO: Generalize this similar to how Hugging Face handles renaming parameters.
                if keys[-1] == 'weight':
                    keys[-1] = 'kernel'

                d = nested_dict

                # Traverse the nested structure and create dictionaries if necessary
                for subkey in keys[:-1]:
                    if subkey not in d:
                        d[subkey] = {}
                    d = d[subkey]

                # Set the value at the final key
                d[keys[-1]] = value

            return nested_dict

        # Ensure config is not None
        if config is None:
            raise ValueError("config cannot be None")

        # These are private and should be exposed as typed properties in derived classes.
        self._config = config
        # self._module = module (commented out but could be used if needed)

        # Public attributes generic to every derived class
        self.key = jax.random.PRNGKey(seed)
        self.dtype = dtype
        self.input_shape = input_shape
        self.generation_config = None

        # To check if the model was initialized automatically.
        self._is_initialized = _do_init

        # Randomly initialized parameters
        random_params = _unflatten_dict_and_rename_params(self.v)
        params_shape_tree = nnx.eval_shape(lambda params: params, random_params)

        # Get the shape of the parameters
        self._params_shape_tree = params_shape_tree

        # Save required parameters as a set
        self._required_params = set(flatten_dict(params_shape_tree).keys())

        # Initialize the parameters
        if _do_init:
            self.params = random_params
"""
