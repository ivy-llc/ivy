"""Main file to hold DTOs for object likes."""

# global
from __future__ import annotations
import ivy
import dill
import inspect
from types import FunctionType, MethodType, ModuleType
from typing import Dict, Optional, Union, TYPE_CHECKING

# local
from ...utils.api_utils import (
    TRANSLATED_OBJ_SUFFIX,
    is_mixed_function,
    maybe_get_methods,
    maybe_get_properties,
    maybe_get_frontend_base_methods,
    maybe_get_frontend_base_module_source,
    is_backend_api,
    is_frontend_api,
    is_ivy_api,
    is_translated_api,
    is_hf_pretrained_class,
    is_helper_func,
    is_submodule_of,
    from_conv_block,
)
from ...utils.inspect_utils import (
    get_closure_vars,
    object_to_source_code,
)
from ...utils import pickling_utils
from ...utils.type_utils import Types
from ...exceptions.exceptions import (
    ProhibitedObjectAccessError,
    InvalidObjectException,
)
from ...transformations import transformer_globals as glob

if TYPE_CHECKING:
    from ...utils.ast_utils import TranslatedContext


class BaseObjectLike:
    """Base class for objects (DTOs) to be passed around in the translators."""

    _CACHABLE_ATTRIBUTES = [
        "name",
        "qualname",
        "module",
        "type",
        "target",
        "parent",
        "from_global",
        "from_conv_block",
        "filename",
        "base_output_dir",
        "is_root_obj",
        "filename",
        "ctx",
        "is_frontend_api",
        "is_ivy_api",
        "is_backend_api",
        "is_translated_api",
        "is_translated_ivy_api",
        "is_ivy_subclass",
        "is_hf_pretrained_class",
        "is_helper_function",
        "is_mixed_function",
        "object_hash",
    ]

    def __init__(
        self,
        obj: Optional[Union[MethodType, FunctionType, type]] = None,
        parent: Optional["BaseObjectLike"] = None,
        root_obj: Union[MethodType, FunctionType, type] = None,
        from_global: Optional[bool] = False,
        filename: str = "",
        base_output_dir: str = "",
        ctx: "TranslatedContext" = None,
        depth: Optional[int] = 0,
        is_root_obj: Optional[bool] = None,
        target: Optional[str] = "",
        **kwargs,
    ) -> None:
        # Handle the essential attributes
        self._obj = obj
        self.parent = parent
        self.root_obj = root_obj
        self.from_global = from_global
        self.filename = filename
        self.base_output_dir = base_output_dir
        assert base_output_dir is not None, "base_output_dir must be provided"
        self.ctx = ctx
        self.depth = depth
        self.target = target

        # Handle attributes derived from the obj if not provided
        self.name = kwargs["name"] if "name" in kwargs else self._derive_name()
        self.qualname = (
            kwargs["qualname"] if "qualname" in kwargs else self._derive_qualname()
        )
        self.module = kwargs["module"] if "module" in kwargs else self._derive_module()
        self.source_code = (
            kwargs["source_code"]
            if "source_code" in kwargs
            else self._derive_source_code()
        )
        self.type = kwargs["type"] if "type" in kwargs else self._derive_type()
        self.closure_vars = (
            kwargs["closure_vars"]
            if "closure_vars" in kwargs
            else get_closure_vars(self._get_obj())
        )

        # Handle base class-related attributes
        self.bases = (
            self._get_obj().__bases__ if hasattr(self._get_obj(), "__bases__") else []
        )
        self.base_class_index, self.base_module_source = (
            (
                kwargs["base_class_index"],
                kwargs["base_module_source"],
            )
            if "base_class_index" in kwargs and "base_module_source" in kwargs
            else self._derive_base_info()
        )
        self.base_methods: Dict[str, str] = (
            kwargs["base_methods"]
            if "base_methods" in kwargs
            else self._derive_base_methods()
        )

        # Handle mixed function relation attributes
        self.is_mixed_function = (
            kwargs["is_mixed_function"]
            if "is_mixed_function" in kwargs
            else is_mixed_function(self._get_obj())
        )
        self.compos_name, self.compos_module, self.mixed_condition_source = (
            (
                kwargs["compos_name"],
                kwargs["compos_module"],
                kwargs["mixed_condition_source"],
            )
            if "compos_name" in kwargs
            and "compos_module" in kwargs
            and "mixed_condition_source" in kwargs
            else self._derive_mixed_function()
        )

        self.ivy_decorators = [
            decor
            for decor in glob.ALL_IVY_DECORATORS
            if hasattr(self._get_obj(), decor)
        ]

        # Handle optional attributes that could be computed or passed
        self.methods = (
            kwargs["methods"] if "methods" in kwargs else self._derive_methods()
        )
        self.properties = (
            kwargs["properties"]
            if "properties" in kwargs
            else self._derive_properties()
        )
        self.code_object_bytes = (
            kwargs["code_object_bytes"]
            if "code_object_bytes" in kwargs
            else self._serialize_code_object()
        )
        self.closure_cell_contents = (
            kwargs["closure_cell_contents"]
            if "closure_cell_contents" in kwargs
            else self._derive_closure_cell_contents()
        )

        # Handle object hashing attributes
        self.object_hash = (
            kwargs["object_hash"] if "object_hash" in kwargs else self.get_object_hash()
        )

        self.is_frontend_api = (
            kwargs["is_frontend_api"]
            if "is_frontend_api" in kwargs
            else is_frontend_api(self._get_obj())
        )
        self.is_ivy_api = (
            kwargs["is_ivy_api"]
            if "is_ivy_api" in kwargs
            else is_ivy_api(self._get_obj())
        )
        self.is_backend_api = (
            kwargs["is_backend_api"]
            if "is_backend_api" in kwargs
            else is_backend_api(self._get_obj())
        )
        self.is_translated_api = (
            kwargs["is_translated_api"]
            if "is_translated_api" in kwargs
            else is_translated_api(self._get_obj())
        )
        self.is_translated_ivy_api = (
            kwargs["is_translated_ivy_api"]
            if "is_translated_ivy_api" in kwargs
            else (
                self.is_translated_api
                and any(self.name.endswith(substr) for substr in TRANSLATED_OBJ_SUFFIX)
            )
        )
        self.is_ivy_subclass = (
            kwargs["is_ivy_subclass"]
            if "is_ivy_subclass" in kwargs
            else (
                issubclass(self._get_obj(), ivy.Module)
                if self.type == Types.ClassType
                else False
            )
        )
        self.is_hf_pretrained_class = (
            kwargs["is_hf_pretrained_class"]
            if "is_hf_pretrained_class" in kwargs
            else is_hf_pretrained_class(self._get_obj())
        )

        self.is_helper_function = (
            kwargs["is_helper_function"]
            if "is_helper_function" in kwargs
            else is_helper_func(self)
        )

        self.from_conv_block = (
            kwargs["from_conv_block"]
            if "from_conv_block" in kwargs
            else from_conv_block(self._get_obj())
        )

    def __eq__(self, other: BaseObjectLike):
        """Check equality with another BaseObjectLike instance."""
        if isinstance(other, (FunctionType, MethodType, type)):
            return self.object_hash == pickling_utils.get_object_hash(other)
        return self.object_hash == other.object_hash

    def __ne__(self, other: BaseObjectLike):
        """Check inequality with another BaseObjectLike instance."""
        return not self.__eq__(other)

    def __hash__(self):
        """Return the hash of the object."""
        return int(self.object_hash, 16)

    def __repr__(self):
        """Return the string representation of the object."""
        return f"{self.type}(name={self.name}, module={self.module})"

    @property
    def obj(self):
        """
        Prevents direct access to the live object from outside the class.

        Raises:
            ProhibitedObjectAccessError: If attempted to access directly from outside the class.
        """
        if not hasattr(self, "_accessing_obj_from_class"):
            raise ProhibitedObjectAccessError()
        return self._obj

    @property
    def is_root_obj(self):
        if self.root_obj is None:
            return False

        return self.object_hash == pickling_utils.get_object_hash(self.root_obj)

    @is_root_obj.setter
    def is_root_obj(self, flag: bool):
        self._is_root_obj = flag

    def _get_obj(self):
        """
        Internal method to safely access the live object within the class.

        Returns:
            object: The live object stored in `_obj`.
        """
        # Temporarily set an internal flag to allow access within the class
        self._accessing_obj_from_class = True
        obj = self.obj
        del self._accessing_obj_from_class  # Clean up the flag after access
        return obj

    def _derive_name(self):
        return (
            self._get_obj().__name__
            if self._get_obj() and hasattr(self._get_obj(), "__name__")
            else ""
        )

    def _derive_qualname(self):
        return getattr(self._get_obj(), "__qualname__", "")

    def _derive_module(self):
        from ...utils.ast_utils import FileNameStrategy

        obj_mod = inspect.getmodule(self._get_obj())
        if obj_mod:
            module_name = FileNameStrategy.infer_filename_from_module_name(
                obj_mod.__name__, as_module=True, base_output_dir=self.base_output_dir
            )
            if (
                not module_name.endswith(".__init__")
                and hasattr(obj_mod, "__file__")
                and obj_mod.__file__.endswith("__init__.py")
            ):
                module_name = module_name + ".__init__"
            return module_name
        return ""

    def _derive_source_code(self):
        return object_to_source_code(self._get_obj())

    def _derive_type(self):
        return Types.get_type(self._get_obj())

    def _derive_methods(self):
        return (
            maybe_get_methods(self._get_obj())
            if self.type == Types.ClassType and self._get_obj()
            else list()
        )

    def _derive_properties(self):
        return (
            maybe_get_properties(self._get_obj())
            if self.type == Types.ClassType and self._get_obj()
            else list()
        )

    def _derive_closure_cell_contents(self):
        closure = (
            self._get_obj().__closure__
            if hasattr(self._get_obj(), "__closure__")
            else None
        )
        closure_vars = (
            None
            if not closure
            else [
                cell.cell_contents
                for cell in closure
                if isinstance(cell.cell_contents, (int, float, str, bool, type(None)))
            ]
        )
        return closure_vars

    def _derive_base_info(self):
        return maybe_get_frontend_base_module_source(self.bases)

    def _derive_base_methods(self):
        if self.base_class_index == -1:
            return {}
        return maybe_get_frontend_base_methods(self._get_obj())

    def _derive_mro(self):
        if self.type == Types.ClassType:
            obj = self._get_obj()
            if isinstance(obj, type):
                try:
                    return obj.mro()
                except TypeError as e:
                    pass
        return []

    def _derive_mixed_function(self):
        (
            compos_name,
            compos_module,
        ) = (
            None,
            None,
        )
        compos = (
            getattr(self._get_obj(), "compos", None) if self.is_mixed_function else None
        )
        if compos:
            compos_name = compos.__name__
            compos_module = compos.__module__

        mixed_condition_source = None
        mixed_condition = (
            getattr(self._get_obj(), "partial_mixed_handler", None)
            if self.is_mixed_function
            else None
        )
        if mixed_condition:
            mixed_condition_source = object_to_source_code(
                mixed_condition, handle_partial_mixed_lambdas=True
            ).strip()
        return compos_name, compos_module, mixed_condition_source

    def _serialize_code_object(self):
        """Serialize the __code__ attribute of the wrapped object, if available."""
        obj = inspect.unwrap(self._get_obj())
        if hasattr(obj, "__code__"):
            return dill.dumps(obj.__code__)

    def get_object_module(self: BaseObjectLike, source: str):
        """
        retrieves the module of the object and attaches additional globals
        that are needed during later translation steps.
        """
        import ivy

        obj_mod = inspect.getmodule(self._get_obj())
        if source == "torch_frontend":
            import ivy.functional.frontends.torch

            obj_mod.__dict__["torch"] = ivy.functional.frontends.torch
            obj_mod.__dict__["Tensor"] = ivy.functional.frontends.torch.Tensor
        elif source == "ivy":
            obj_mod.__dict__["ivy"] = ivy

        return obj_mod

    def has_same_code(
        self, other: Union[MethodType, FunctionType, type, BaseObjectLike]
    ) -> bool:
        if self.code_object_bytes is None:
            return False

        code_object = dill.loads(self.code_object_bytes)

        if isinstance(other, BaseObjectLike):
            if other.code_object_bytes is None:
                return False

            return code_object == dill.loads(other.code_object_bytes)

        obj = inspect.unwrap(other)
        if hasattr(obj, "__code__"):
            return code_object == obj.__code__

        return False

    def is_same_object(self, live_obj: Union[FunctionType, MethodType, type]) -> bool:
        """
        Check if the given live object matches the stored object in this instance.
        """
        if live_obj.__module__ != self.module:
            return False
        if getattr(live_obj, "__qualname__", "") != self.qualname:
            return False
        if hasattr(live_obj, "__code__") and self.type in {
            Types.FunctionType,
            Types.MethodType,
        }:
            return live_obj.__code__ == dill.loads(self.code_object_bytes)
        if self.type == Types.ClassType and isinstance(live_obj, type):
            return self.source_code == object_to_source_code(live_obj)
        if self.type == Types.ModuleType and isinstance(live_obj, ModuleType):
            return self.source_code == object_to_source_code(live_obj)

        return False  # Fallback for unhandled cases

    def get_object_hash(self):
        """Generate a unique key for caching and equality checks."""
        if getattr(self, "object_hash", None):
            return self.object_hash

        return pickling_utils.get_object_hash(self._get_obj())

    def get_unwrapped_object(self):
        """Gets the unwrapped object from the original object."""
        return inspect.unwrap(self._get_obj())

    @staticmethod
    def _infer_object_like(type_: Types):
        if type_ in {Types.MethodType, Types.FunctionType}:
            return FuncObjectLike
        elif type_ == Types.ClassType:
            return TypeObjectLike
        else:
            raise ValueError(f"Unsupported type: {type_}")

    def to_dict(self) -> Dict:
        return {
            attr: (
                getattr(self, attr).to_dict()
                if isinstance(getattr(self, attr), BaseObjectLike)
                else getattr(self, attr)
            )
            for attr in self._CACHABLE_ATTRIBUTES
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "BaseObjectLike":
        return cls.from_cache(
            **{
                attr: (
                    cls.from_dict(data[attr])
                    if isinstance(data[attr], dict) and "object_hash" in data[attr]
                    else data[attr]
                )
                for attr in cls._CACHABLE_ATTRIBUTES
            }
        )

    def dumps(self):
        return dill.dumps(self.to_dict())

    @classmethod
    def loads(cls, data) -> "BaseObjectLike":
        return cls.from_dict(dill.loads(data))

    def to_cache(self) -> Dict:
        return self.to_dict()

    @staticmethod
    def from_cache(**kwargs) -> "BaseObjectLike":
        ObjectLike = BaseObjectLike._infer_object_like(kwargs.get("type"))
        return ObjectLike(**kwargs)

    @staticmethod
    def from_object(
        obj: Union[MethodType, FunctionType, type],
        parent: Optional["BaseObjectLike"] = None,
        root_obj: Union[MethodType, FunctionType, type] = None,
        from_global: Optional[bool] = False,
        ctx: "TranslatedContext" = None,
        is_root_obj: Optional[bool] = None,
        depth: Optional[int] = 0,
        target: Optional[str] = "",
        base_output_dir: str = "",
    ) -> BaseObjectLike:
        if isinstance(obj, property):
            obj = obj.fget

        obj_type = Types.get_type(obj)

        from ...utils.ast_utils import FileNameStrategy

        ObjectLike = BaseObjectLike._infer_object_like(obj_type)
        object_like = ObjectLike(
            obj=obj,
            parent=parent,
            root_obj=root_obj,
            from_global=from_global,
            ctx=ctx,
            depth=depth,
            is_root_obj=is_root_obj,
            target=target,
            base_output_dir=base_output_dir,
        )
        object_like.filename = FileNameStrategy.infer_filename_from_object_like(
            object_like=object_like, target="", base_output_dir=base_output_dir
        )
        if (
            any(
                is_submodule_of(object_like, cls)
                for cls in (
                    "torch.nn.modules.module.Module",
                    "ivy.stateful.module.Module",
                )
            )
            and target == "numpy"
        ):
            raise InvalidObjectException(
                f"Cannot transpile the object '{obj}' to `numpy` because it is an instance of a stateful class. ",
                propagate=True,
            )
        return object_like


class FuncObjectLike(BaseObjectLike):
    """Data Transfer Object (DTO) to carry information regarding a func-like object."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TypeObjectLike(BaseObjectLike):
    """Data Transfer Object (DTO) to carry information regarding a type-like object (e.g., a class)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
