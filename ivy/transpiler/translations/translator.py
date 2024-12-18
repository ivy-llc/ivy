"""Exposes interface for the translators."""

# global
import ivy
import logging
from types import FunctionType, MethodType
from typing import List, Optional, Union
import os
import importlib
import collections

# local
from ..utils.cache_utils import (
    Cacher,
    AtomicCacheUnit,
    PRELOAD_CACHE,
    cache_sort_key,
)
from ..transformations.transformer import Transformer
from ..utils.logging_utils import Logger
from ..utils.naming_utils import NAME_GENERATOR
from ..utils import api_utils
from ..utils.api_utils import get_function_from_modules
from ..translations.data.object_like import (
    BaseObjectLike,
    TypeObjectLike,
    FuncObjectLike,
)
from ..translations.configurations.base_translator_config import (
    BaseTranslatorConfig,
)
from ..utils.source_utils import (
    create_output_dir,
    format_all_files_in_directory,
    maybe_add_profiling_imports,
    maybe_add_profiling_decorators,
    safe_get_object_from_translated_directory,
)
from ..transformations.transformers.rename_transformer import (
    BaseRenameTransformer,
)
from ..exceptions.exceptions import (
    format_missing_frontends_msg,
)
from ..transformations import transformer_globals as glob


class Translator:
    """
    Main class to apply the S2S Translator.
    """

    depth: int = 0
    missing_frontends = list()

    def __init__(
        self,
        configuration: BaseTranslatorConfig,
        cacher: Optional[Cacher] = None,
        logger: Optional[Logger] = None,
        root_obj: Union[MethodType, FunctionType, type] = None,
        reuse_existing: bool = True,
    ) -> None:
        self.configuration: BaseTranslatorConfig = configuration
        self.cacher: Cacher = cacher if cacher else Cacher(singleton=True)
        self.logger: Logger = logger if logger else Logger()
        self.root_obj: Union[MethodType, FunctionType, type] = (
            root_obj if root_obj else None
        )
        self.source: str = self.configuration.source
        self.target: str = self.configuration.target
        self._output_dir: str = ""
        self.reuse_existing = reuse_existing

    def _reset_translation_state(self):
        """Reset the common states required for translation."""
        NAME_GENERATOR.reset_state()
        NAME_GENERATOR.set_prefixes(self.configuration.target)
        Translator.depth = 0
        Translator.missing_frontends = list()
        glob.CONFLICTING_METHODS = set()
        self.cacher.clear()
        if self.configuration.base_output_dir in api_utils.TRANSLATED_OBJ_PREFIX:
            api_utils.TRANSLATED_OBJ_PREFIX.remove(self.configuration.base_output_dir)

    def _set_env_variable(self, var_name: str, default: str):
        """Set an environment variable to a given default if not already set."""
        value = os.environ.get(var_name, None)
        if value is None or value.lower() == "true":
            os.environ[var_name] = default
        else:
            os.environ[var_name] = "false"

    def _reset_translation(self):
        """Reset translation-related state."""
        self._reset_translation_state()

    def _setup_translation(self, object: Union[MethodType, FunctionType, type]):
        """Set up the environment and state for translating a given object."""
        # Reset the translation state
        self._reset_translation_state()

        # Set environment variables
        self._set_env_variable("APPLY_TRANSPOSE_OPTIMIZATION", "true")
        self._set_env_variable("USE_NATIVE_FW_LAYERS", "true")
        self._set_env_variable("PROFILE_S2S", "false")
        os.environ["CONV_BLOCK_DETECTED"] = "false"

        # add the base output directory as one of the prefixes to flag objects within
        # this directory as belonging to the translated_api (i.e obj_like.is_translated_api = True)
        api_utils.TRANSLATED_OBJ_PREFIX.insert(0, self.configuration.base_output_dir)

        # Create output directory for the translation
        use_legacy_dir = os.environ.get("USE_LEGACY_DIR", "false") == "true"
        self._output_dir = create_output_dir(
            object,
            target=self.target,
            base_output_dir=self.configuration.base_output_dir,
            use_legacy_dir_structure=use_legacy_dir,
        )

        # Bind a unique name to the given object
        _ = NAME_GENERATOR.generate_name(object)

        # Store this object as the root object
        self.root_obj = object

        # Clear all caches
        self.cacher.clear()

        # Create and initialize the object-like structure
        object_like = BaseObjectLike.from_object(
            obj=object,
            parent=None,
            root_obj=self.root_obj,
            depth=self.depth,
            target=self.target,
            base_output_dir=self.configuration.base_output_dir,
        )

        # Add the object-like to the call stack of the Recursive transformer
        transformer = self.configuration.transformers[8]
        transformer.call_stack = [object_like]
        transformer._metaclasses = []

        return object_like

    def _setup_standard_translations(
        self,
    ) -> List[Union[MethodType, FunctionType, type]]:
        """Set up the standard objects to translate for the current source."""
        objects_to_translate: List[Union[MethodType, FunctionType, type]] = []
        for (
            standard_module,
            standard_objects,
        ) in self.configuration.standard_objects_to_translate.items():
            if standard_module.startswith("ivy_transpiled_outputs"):
                standard_module = standard_module.replace(
                    "ivy_transpiled_outputs", self.configuration.base_output_dir
                )
            for standard_object in standard_objects:
                # first try to retrive the live object from the current object_like's module
                module_to_search = importlib.import_module(standard_module)
                object_to_translate = get_function_from_modules(
                    standard_object, [module_to_search]
                )
                if object_to_translate is not None:
                    objects_to_translate.append(object_to_translate)

        return objects_to_translate

    def _standard_translate(self, object_like: BaseObjectLike, reuse_existing: bool):
        """
        Translate standard functions such as frontend `torch.Tensor` dunder methods or
        ivy util functions like `handle_array_like_without_promotion` etc. These objects
        are decoupled from the core recursive translation process and are always translated
        during the call to `ivy.transpile` depending on the source framework.
        """
        # Retrieve the standard objects that need to be translated by default corresponding
        # to the current source
        objects_to_translate: List[Union[MethodType, FunctionType, type]] = (
            self._setup_standard_translations()
        )

        # Iterate over each object and translate it
        for object_to_translate in objects_to_translate:
            object_like_to_translate: BaseObjectLike = BaseObjectLike.from_object(
                obj=object_to_translate,
                parent=object_like.parent,
                root_obj=None,
                from_global=False,
                ctx=object_like.ctx,
                depth=Translator.depth,
                base_output_dir=self.configuration.base_output_dir,
            )
            _ = NAME_GENERATOR.generate_name(object_like_to_translate)
            _ = Translator.simple_translate(
                object_like=object_like_to_translate,
                depth=Translator.depth,
                parent=object_like.parent,
                from_global=False,
                configuration=self.configuration,
                cacher=self.cacher,
                logger=self.logger,
                output_dir=self._output_dir,
                profiling=False,
                reuse_existing=reuse_existing,
            )

    @staticmethod
    def simple_translate(
        object_like: Union[FuncObjectLike, TypeObjectLike],
        configuration: BaseTranslatorConfig,
        cacher: Cacher,
        logger: Logger,
        root_object: Union[MethodType, FunctionType, type] = None,
        parent: BaseObjectLike = None,
        from_global: bool = False,
        output_dir: str = "",
        depth: int = 0,
        profiling: bool = False,
        reuse_existing: bool = True,
    ) -> Union[MethodType, FunctionType, type]:

        assert isinstance(
            object_like, (FuncObjectLike, TypeObjectLike)
        ), f"object_like must be either FuncObjectLike or TypeObjectLike, got {type(object_like)}"

        if object_like.from_conv_block:
            os.environ["CONV_BLOCK_DETECTED"] = "true"

        unwrapped_object = object_like.get_unwrapped_object()
        if not os.environ.get("UPDATE_S2S_CACHE", None) == "true":
            # Check if the object is in the preloaded cache
            if PRELOAD_CACHE.exist(
                unwrapped_object, configuration.source, configuration.target
            ):
                logging.debug("Fetching cached units from the cache...")
                cached_units = PRELOAD_CACHE.get(
                    unwrapped_object, configuration.source, configuration.target
                )
                # Generate source code for all units in the call tree
                for unit in sorted(cached_units, key=cache_sort_key):
                    cached_obj_like = unit.object_like
                    if not cacher.object_like_bytes_to_translated_object_str_cache.exist(
                        cached_obj_like
                    ):
                        unit.object_like.root_obj = root_object
                        unit.object_like.depth = depth
                        unit.object_like_bytes_to_translated_object_str_cache._cache.update(
                            cacher.object_like_bytes_to_translated_object_str_cache._cache
                        )
                        old_name = unit.ast_root.body[0].name
                        new_name = NAME_GENERATOR.generate_name(cached_obj_like)
                        BaseRenameTransformer(unit.ast_root).rename(old_name, new_name)
                        translated_obj_str = AtomicCacheUnit.generate_source_code(
                            unit,
                            output_dir=output_dir,
                            base_output_dir=configuration.base_output_dir,
                            from_cache=True,
                        )
                        cacher.object_like_bytes_to_translated_object_str_cache.cache(
                            cached_obj_like, translated_obj_str
                        )

        if cacher.object_like_bytes_to_translated_object_str_cache.exist(object_like):
            # early return
            translated_str = (
                cacher.object_like_bytes_to_translated_object_str_cache.get(object_like)
            )
            return translated_str

        if reuse_existing:
            # 1b. Return if `reuse_existing=True` and object exists in the translated directory
            # TODO: figure out why using importlib.import_module (during simple_translate) causes weird effects during `format_all_files_in_directory`
            # minimal example: transpile(kornia.io.load_image, source="torch", target="ivy", reuse_existing=True)
            # we are using `safe_get_object_from_translated_directory` instead as a workaround.
            translated_object_str = safe_get_object_from_translated_directory(
                object_like,
                translated_dir=output_dir,
                reuse_existing=reuse_existing,
                base_output_dir=configuration.base_output_dir,
            )
            if translated_object_str:
                # store the translated object in the translated objects cache
                # this is done this object can be correctly imported as a dependency
                # during sourcegen.
                cacher.object_like_bytes_to_translated_object_str_cache.cache(
                    object_like, translated_object_str
                )
                logging.debug(f"Reusing existing object ... {translated_object_str}")
                return translated_object_str

        # If not in cache, proceed with translation
        if not output_dir:
            output_dir = create_output_dir(
                object_like,
                base_output_dir=configuration.base_output_dir,
            )

        # Set the current call tree if this is a top-level function
        if depth == 0:
            cacher.current_call_tree = unwrapped_object

        ast_transformer = Transformer(
            object_like=object_like,
            configuration=configuration,
            cacher=cacher,
            logger=logger,
            output_dir=output_dir,
            reuse_existing=reuse_existing,
        )
        ast_root = ast_transformer.transform(
            profiling=profiling, parent=parent, from_global=from_global
        )
        # 3. Update the missing frontends list
        Translator.missing_frontends.extend(ast_transformer.missing_frontends)

        # 4. Store the transformed ast_root in the code to ast cache
        cacher.code_to_ast_cache.cache(object_like.source_code, ast_root)

        # Create an AtomicCacheUnit
        cache_unit = AtomicCacheUnit(
            ast_root=ast_root,
            object_like=object_like,
            globals=ast_transformer.globals,
            imports=ast_transformer.imports,
            from_imports=ast_transformer.from_imports,
            circular_reference_object_likes=ast_transformer.circular_ref_object_likes,
            source=ast_transformer.source,
            target=ast_transformer.target,
            object_like_bytes_to_translated_object_str_cache=cacher.object_like_bytes_to_translated_object_str_cache,
            import_statement_cache=cacher.import_statement_cache,
            global_statement_cache=cacher.global_statement_cache,
            emitted_source_cache=cacher.emitted_source_cache,
            depth=depth,
        )

        translated_obj_str = AtomicCacheUnit.generate_source_code(
            cache_unit,
            output_dir=output_dir,
            base_output_dir=configuration.base_output_dir,
            from_cache=False,
        )

        # Store the cache unit in the preloaded objects cache
        if os.environ.get("UPDATE_S2S_CACHE", None) == "true":
            PRELOAD_CACHE.set(
                key=unwrapped_object,
                unit=cache_unit,
                current_call_tree=cacher.current_call_tree,
                source=configuration.source,
                target=configuration.target,
            )

        # Also store the translated object in the translated objects cache
        cacher.object_like_bytes_to_translated_object_str_cache.cache(
            object_like, translated_obj_str
        )

        # Reset the current call tree if this is a top-level function
        if depth == 0:
            cacher.current_call_tree = None

        return translated_obj_str

    def translate(
        self,
        object: Union[MethodType, FunctionType, type],
        reuse_existing: bool = True,
        base_output_dir: str = None,
    ) -> Union[MethodType, FunctionType, type]:
        """
        Translates an input object-like from one framework to another.

        For two objects with same dedent code, the second object will reuse
        the transformed ast node of previous one.

        For example:
            # A.py
            def foo(x, y):
                z = x + y
                return z

            # B.py
            def foo(x, y):
                z = x + y
                return z

        If the translation of `A.foo` happens after `B.foo`, it will reuse
        the transformed ast node of `B.foo` to speed up the conversion.
        """
        from ..translations.configurations.ivy_to_source_translator_config import (
            IvyToSourceTranslatorConfig,
        )

        # 0a. Set backend if needed
        if isinstance(self.configuration, IvyToSourceTranslatorConfig):
            ivy.set_backend(self.configuration.target)

        # 0b. Set up translation
        object_like = self._setup_translation(object=object)

        # 0c. First translate any standard objects before proceeding with
        # the translation of the current object
        self._standard_translate(object_like, reuse_existing=reuse_existing)

        # 1. Simple translate
        translated_object_str = self.simple_translate(
            object_like=object_like,
            root_object=self.root_obj,
            configuration=self.configuration,
            cacher=self.cacher,
            logger=self.logger,
            output_dir=self._output_dir,
            reuse_existing=reuse_existing,
        )

        if Translator.missing_frontends:
            frequency = collections.Counter(Translator.missing_frontends).most_common()
            msg = format_missing_frontends_msg(frequency)
            DEBUG = int(os.getenv("DEBUG", 1))
            if DEBUG == 0 or DEBUG == 1:
                from ..main import _set_debug_level

                # set the level to 1 to print warnings to the terminal
                _set_debug_level(1)
                logging.warning(msg)
                # reset the level to 0 to avoid printing warnings to the terminal
                _set_debug_level(0)
            else:
                logging.warning(msg)
        # 3. Create profiling.py
        maybe_add_profiling_imports(self._output_dir)
        maybe_add_profiling_decorators(self._output_dir)

        # 4. Apply formatting/linting on the files within the directory
        translated_object = format_all_files_in_directory(
            self._output_dir,
            base_output_dir=base_output_dir,
            translated_object_name=translated_object_str,
            filename=object_like.filename[:-3],
            to_ignore=(
                "tensorflow__stateful",
                "tensorflow__stateful_layers",
                "jax__stateful",
                "jax__stateful_layers",
            ),
            logger=self.logger,
        )

        # 5. Save the preload cache
        if os.environ.get("UPDATE_S2S_CACHE", None) == "true":
            logging.debug("Starting cache saving.")
            filename = f"{self.source}_to_{self.target}_translation_cache.pkl"
            PRELOAD_CACHE.save_preloaded_cache(filename)

        # 6. Reset the translation
        self._reset_translation()

        # 7. Annotate the translated object to avoid repeated translations
        setattr(translated_object, "__already_s2s", self.configuration.target)

        # 8. Log and return
        logging.info(repr(translated_object) + " stored at path: " + self._output_dir)

        return translated_object
