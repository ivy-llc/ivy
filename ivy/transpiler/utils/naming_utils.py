# global
import gast
import collections
from typing import Any, Iterable, Optional
import builtins
import keyword
import re
import types
import inspect
import functools
from inspect import signature, Parameter

# local
from . import pickling_utils
from .profiling_utils import name_map
from .type_utils import Types


def unwrap_obj(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        new_args = [inspect.unwrap(arg) if arg is not None else arg for arg in args]
        return func(self, *new_args, **kwargs)

    return wrapper


class UniqueNameGenerator:
    """A class for associating names uniquely with objects.

    The following invariants are enforced:
    - Each object gets a single name.
    - Each name is unique within a given namespace.
    - Names generated do not shadow builtins, unless the object is indeed that builtin.
    """

    def __init__(self):
        self._obj_to_name = {}
        self._name_to_obj = {}  # Reverse mapping
        self._unassociated_names = set()
        self._used_names = set()
        self._base_count = collections.defaultdict(int)

        self._illegal_char_regex = re.compile("[^0-9a-zA-Z_]+")
        self._name_suffix_regex = re.compile(r"(.*)_(base_count_\d+)$")

        self.target = ""
        self._old_prefix = ""
        self._new_prefix = ""

    @property
    def old_prefix(self):
        return self._old_prefix

    @old_prefix.setter
    def old_prefix(self, value):
        self._old_prefix = value

    @property
    def new_prefix(self):
        return self._new_prefix

    @new_prefix.setter
    def new_prefix(self, value):
        self._new_prefix = value

    def set_prefixes(self, target):
        """Set the old and new prefixes based on the target framework."""
        if "frontend" in target:
            self.old_prefix = "Translated_"
            self.new_prefix = "Translated_"
        elif target == "ivy":
            self.old_prefix = "Translated_"
            self.new_prefix = "ivy_"
        else:
            self.old_prefix = "ivy_"
            self.new_prefix = f"{target}_"

        self.target = target

    def get_suffix(self, obj: Optional[Any]) -> str:
        """Get the suffix for the candidate name based on the object's module."""
        from ivy.transpiler.translations.data.object_like import (
            BaseObjectLike,
        )

        if isinstance(obj, BaseObjectLike):
            module = obj.module if obj.module else ""
        else:
            module = getattr(obj, "__module__", "")
        if "frontends.torch.tensor" in module:
            return "frnt_"  # for torch.Tensor methods
        elif "frontends" in module:
            return "frnt"  # for torch frontend functions
        elif "data_classes.array" in module:
            return "bknd_"  # for ivy.Array methods
        elif "functional.ivy" in module:
            return "bknd"  # for ivy functions
        else:
            return ""

    def get_candidate_name(self, obj: Optional[Any]):
        """Generate a candidate name for an object."""
        from ivy.transpiler.translations.data.object_like import (
            BaseObjectLike,
        )

        if isinstance(obj, BaseObjectLike) and obj.type in (
            Types.FunctionType,
            Types.ClassType,
        ):
            candidate = obj.qualname.split(".")[-1]

        elif isinstance(obj, (types.FunctionType, type)):
            candidate = obj.__qualname__.split(".")[-1]
        else:
            candidate = obj if isinstance(obj, str) else ""

        if candidate.startswith(self.old_prefix):
            candidate = candidate.replace(self.old_prefix, self.new_prefix)
        else:
            prefix = self.old_prefix if "frontend" in self.target else self.new_prefix
            candidate = prefix + candidate

        name_map[candidate] = obj

        return candidate

    @unwrap_obj
    def generate_name(self, obj: Optional[Any]):
        """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
        if obj is not None:
            object_hash = pickling_utils.get_object_hash(obj)
            if object_hash in self._obj_to_name:
                return self._obj_to_name[object_hash]

        # Generate the candidate name
        candidate = self.get_candidate_name(obj)

        # Append suffix based on the module of the object
        suffix = self.get_suffix(obj)
        if suffix and not candidate.endswith(suffix):
            candidate = f"{candidate}_{suffix}"

        # delete all characters that are illegal in a Python identifier
        candidate = self._illegal_char_regex.sub("_", candidate)

        if not candidate:
            candidate = "_unnamed"

        if candidate[0].isdigit():
            candidate = f"_{candidate}"

        match = self._name_suffix_regex.match(candidate)
        if match is None:
            base = candidate
            num = None
        else:
            base, num_str = match.group(1, 2)
            num = int(num_str.strip("base_count"))

        candidate = base if num is None else f"{base}_base_count_{num}"
        if not num:
            num = self._base_count[base]

        while candidate in self._used_names or self._is_illegal_name(candidate):
            num += 1
            candidate = f"{base}_base_count_{num}"

        self._used_names.add(candidate)
        self._base_count[base] = num
        if obj is None:
            self._unassociated_names.add(candidate)
        else:
            self._obj_to_name[pickling_utils.get_object_hash(obj)] = candidate
            self._name_to_obj[candidate] = obj
        return candidate

    @unwrap_obj
    def associate_name_with_obj(self, name, obj):
        """Associate a unique name with an object.

        Neither `name` nor `obj` should be associated already. This method is particularly
        useful when you generate a name without associating it with an object (by passing
        `None` to the `generate_name` method), and want to associate the name with an object
        at a later time.
        """
        assert obj not in self._obj_to_name
        assert name in self._unassociated_names
        self._obj_to_name[pickling_utils.get_object_hash(obj)] = name
        self._name_to_obj[name] = obj
        self._unassociated_names.remove(name)

    @unwrap_obj
    def get_object(self, name: str) -> Any:
        """Retrieve the object associated with a name."""
        return self._name_to_obj.get(name)

    @unwrap_obj
    def get_name(self, obj: Any) -> str:
        """Retrieve the name associated with an object."""
        return self._obj_to_name.get(pickling_utils.get_object_hash(obj))

    @unwrap_obj
    def set_object(self, name: str, obj: Any):
        """Set the object associated with a name."""
        self._name_to_obj[name] = obj
        self._obj_to_name[pickling_utils.get_object_hash(obj)] = name

    @unwrap_obj
    def set_name(self, obj: Any, name: str):
        """Set the name associated with an object."""
        self._obj_to_name[pickling_utils.get_object_hash(obj)] = name
        self._name_to_obj[name] = obj

    def reset_state(self):
        """Resets the internal mappings and variables."""
        self._obj_to_name = {}
        self._name_to_obj = {}  # Reverse mapping
        self._unassociated_names = set()
        self._used_names = set()
        self._base_count = collections.defaultdict(int)
        self.target = ""
        self._old_prefix = ""
        self._new_prefix = ""

    def _is_illegal_name(self, name):
        # 1. reserved keywords are never allowed as names.
        # 2. Can't shadow a builtin name, unless you *are* that builtin.
        return name in keyword.kwlist or name in builtins.__dict__


NAME_GENERATOR = UniqueNameGenerator()
