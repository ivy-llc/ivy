"""Exposes interface for transformers."""

# global
from typing import Optional

# local
import gast
from ..translations.configurations.base_translator_config import (
    BaseTranslatorConfig,
)
from ..utils import logging_utils
from ..utils.cache_utils import Cacher
from ..utils.origin_utils import attach_origin_info
from ..utils.naming_utils import NAME_GENERATOR
from ..transformations.transformers.base_transformer import (
    BaseTransformer,
)
from ..translations.data.object_like import (
    BaseObjectLike,
)
from ..transformations.transformers.rename_transformer import (
    BaseRenameTransformer,
)

from types import ModuleType
from typing import List, Set, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..transformations.transformers.globals_transformer.base_transformer import (
        GlobalObj,
    )
    from ..translations.data.object_like import (
        TypeObjectLike,
    )
    from ..translations.data.object_like import (
        FuncObjectLike,
    )
    from ..utils.ast_utils import ImportObj, FromImportObj


class Transformer(BaseTransformer):
    def __init__(
        self,
        object_like: BaseObjectLike,
        configuration: BaseTranslatorConfig,
        cacher: Optional[Cacher] = None,
        logger: Optional[logging_utils.Logger] = None,
        output_dir: str = "",
        reuse_existing: bool = True,
    ):
        self._source: str = configuration.source
        self._target: str = configuration.target
        self._imports: Set[ImportObj] = set()
        self._from_imports: Set[FromImportObj] = set()
        self._circular_ref_object_likes: Set[Union[FuncObjectLike, TypeObjectLike]] = (
            set()
        )
        self._globals: List[GlobalObj] = []
        self._method_conflict: Set[str] = set()
        self._output_dir: str = output_dir
        self._object_like: BaseObjectLike = object_like
        self._object_module: List[ModuleType] = [
            object_like.get_object_module(source=self._source)
        ]
        self.object_module_filenames = []
        self.configuration: BaseTranslatorConfig = configuration
        self.cacher: Cacher = cacher if cacher else Cacher(singleton=True)
        self.logger: logging_utils.Logger = logger if logger else logging_utils.Logger()
        self.reuse_existing = reuse_existing
        self.missing_frontends = list()

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target

    @property
    def imports(self):
        return self._imports

    @property
    def from_imports(self):
        return self._from_imports

    @property
    def circular_ref_object_likes(self):
        return self._circular_ref_object_likes

    @circular_ref_object_likes.setter
    def circular_ref_object_likes(self, obj):
        self._circular_ref_object_likes.add(obj)

    @property
    def globals(self):
        return self._globals

    @globals.setter
    def globals(self, value):
        self._globals = value

    @property
    def method_conflicts(self):
        return self._method_conflict

    @method_conflicts.setter
    def method_conflicts(self, method_name: str):
        self._method_conflict.add(method_name)

    @property
    def object_module(self):
        return self._object_module

    @object_module.setter
    def object_module(self, module: ModuleType):
        assert isinstance(
            module, ModuleType
        ), f"module should be of type {ModuleType}. Got {type(module)}"
        if (
            hasattr(module, "__file__")
            and module.__file__ not in self.object_module_filenames
        ):
            self._object_module.append(module)
            self.object_module_filenames.append(module.__file__)

    @property
    def object_like(self):
        return self._object_like

    @property
    def output_dir(self):
        return self._output_dir

    def _apply(self, transformer, node, log_level, profiling=False):
        if len(self.configuration.transformer_configs) > log_level - 1:
            configuration = self.configuration.transformer_configs[log_level - 1]
        else:
            configuration = BaseTranslatorConfig
        t = transformer(
            node,
            self,
            configuration=configuration,
        )
        t.profiling = profiling
        self.logger.log_transformed_code(
            log_level, self.root, transformer.__name__, position="BEFORE"
        )
        t.transform()
        self.logger.log_transformed_code(
            log_level, self.root, transformer.__name__, position="AFTER"
        )

    def transform(self, profiling=False, parent=None, from_global=False):
        self.logger.log(1, f"Source code: \n{self.object_like.source_code}")
        self.decorate_func_name: str = None
        self.root = gast.parse(self.object_like.source_code)
        # attach origin info to self.root. This attaches information about:
        # - the current obj the AST belongs to
        # - the parent obj from which the current obj originates
        # - whether the current obj is from a global object
        # - other meta info: source code, lino no, filepath etc..
        self.root = attach_origin_info(
            self.root, self.object_like, origin_object=parent, from_global=from_global
        )
        self.visit(self.root)

        # Return without applying the transformations if decorated with @not_translate
        options = getattr(
            self.object_like._get_obj(),
            self.configuration.configuration["CONVERSION_OPTIONS"],
            None,
        )
        if options is not None and options.not_convert:
            # rename the object and return
            obj_name = self.object_like.name
            new_name = NAME_GENERATOR.get_name(self.object_like)
            BaseRenameTransformer(self.root).rename(
                old_name=obj_name, new_name=new_name
            )
            return self.root

        for index, transformer in enumerate(self.configuration.transformers):
            self._apply(
                transformer, self.root, log_level=index + 1, profiling=profiling
            )

        self.logger.log_transformed_code(
            logging_utils.MAX_TRANSFORMERS, self.root, "All Transformers"
        )
        return self.root

    def transform_node(self, node, module):
        """variant of the transform method to apply transformations on a targeted node."""
        assert isinstance(
            module, ModuleType
        ), f"module should be of type {ModuleType}. Got {type(module)}"
        if (
            hasattr(module, "__file__")
            and module.__file__ not in self.object_module_filenames
            and not ("ivy" in module.__file__ and "__init__.py" in module.__file__)
            and self.object_module[0].__file__ != module.__file__
        ):
            self._object_module.insert(0, module)
            self.object_module_filenames.append(module.__file__)
        elif (
            hasattr(module, "__file__")
            and module.__file__ in self.object_module_filenames
        ):
            # reorder the modules such that `module` is at index 0
            self._object_module.remove(module)
            self._object_module.insert(0, module)
        # Apply the transformers to the node
        for index, transformer in enumerate(self.configuration.transformers):
            self.logger.log_transformed_code(
                index + 1, node, transformer.__name__, position="BEFORE"
            )
            transformer(
                node,
                self,
                configuration=self.configuration.transformer_configs[index],
            ).transform()
            self.logger.log_transformed_code(
                index + 1, node, transformer.__name__, position="AFTER"
            )

        # Log the transformed code
        self.logger.log_transformed_code(
            logging_utils.MAX_TRANSFORMERS, node, "All Transformers"
        )

        return node

    def visit_FunctionDef(self, node):
        if self.decorate_func_name is None:
            self.decorate_func_name = node.name

        self.generic_visit(node)
        return node

    def get_module_name(self):
        """
        Return the main function name which will be used as module name
        in generate_source_code.
        """
        assert self.decorate_func_name, "decorate_func_name shall not be None."
        return self.decorate_func_name
