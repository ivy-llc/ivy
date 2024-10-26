# global
from __future__ import annotations
import re
import os
import tensorflow as tf
import keras
import functools
from tensorflow.python.util import nest
from typing import (
    NamedTuple,
    Callable,
    Any,
    Tuple,
    List,
    Set,
    Dict,
    Type,
    Iterator,
    Optional,
    Union,
    TYPE_CHECKING,
)
import itertools
import warnings
import typing
import inspect
from collections import OrderedDict
from packaging.version import parse

if TYPE_CHECKING:
    pass


if keras.__version__ >= "3.0.0":
    KerasVariable = keras.src.backend.Variable
else:
    KerasVariable = tf.Variable


def get_assignment_dict():
    # Traverse the call stack
    lhs = None
    for frame_info in inspect.stack():
        # Check if the code context is an assignment statement
        if frame_info.code_context and "=" in frame_info.code_context[0]:
            # Split the assignment and retrieve the LHS
            lhs = frame_info.code_context[0].split("=")[0].strip()
            if "self" not in lhs:
                continue
            break

    if not lhs:
        return None, ""

    # Replace indexing with attribute access
    lhs = re.sub(r"\[(\d+)\]", r".\1", lhs)

    # Split the LHS based on "." and get individual components
    components = lhs.split(".")

    # Initialize the dictionary
    assignment_dict = {}

    # Retrieve the live objects associated with each component
    for i in range(len(components)):
        # Construct the key
        key = ".".join(components[: i + 1])

        # Retrieve the value
        if i == 0:
            value = frame_info.frame.f_locals.get(components[i])
        else:
            value = getattr(assignment_dict[".".join(components[:i])], components[i])

        # Add the key-value pair to the dictionary
        assignment_dict[key] = value

    return assignment_dict, lhs


def store_frame_info(fn):
    @functools.wraps(fn)
    def frame_info_wrapper(self, *args, **kwargs):
        if self._previous_frame_info is None:
            # store the info about the calling frame.
            stack = inspect.stack()
            self._previous_frame_info = stack[1]
        res = fn(self, *args, **kwargs)
        # reset the frame-info
        self._previous_frame_info = None
        return res

    return frame_info_wrapper


# A NodeDef holds two callables:
# - flatten_fn should take the collection and return a flat list of values.
#   It can also return some context that is used in reconstructing the
#   collection.
# - unflatten_fn should take a flat list of values and some context
#   (returned by flatten_fn). It returns the collection by reconstructing
#   it from the list and the context.
Context = Any
PyTree = Any
FlattenFunc = Callable[[PyTree], Tuple[List, Context]]
UnflattenFunc = Callable[[List, Context], PyTree]


class NodeDef(NamedTuple):
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc


SUPPORTED_NODES: Dict[Type[Any], NodeDef] = {}


def _register_pytree_node(
    typ: Any, flatten_fn: FlattenFunc, unflatten_fn: UnflattenFunc
) -> None:
    SUPPORTED_NODES[typ] = NodeDef(flatten_fn, unflatten_fn)


def _dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())


def _dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return {key: value for key, value in zip(context, values)}


_register_pytree_node(dict, _dict_flatten, _dict_unflatten)

if parse(keras.__version__).major > 2:
    _register_pytree_node(
        keras.src.utils.tracking.TrackedDict, _dict_flatten, _dict_unflatten
    )


def _get_node_type(pytree: Any) -> Any:
    return type(pytree)


# A leaf is defined as anything that is not a Node.
def _is_leaf(pytree: PyTree) -> bool:
    return _get_node_type(pytree) not in SUPPORTED_NODES.keys()


# A TreeSpec represents the structure of a pytree. It holds:
# "type": the type of root Node of the pytree
# context: some context that is useful in unflattening the pytree
# children_specs: specs for each child of the root Node
# num_leaves: the number of leaves
class TreeSpec:
    def __init__(self, type, context, children_specs):
        self.type: Any = type
        self.context: Context = context
        self.children_specs: List["TreeSpec"] = children_specs
        self.num_leaves: int = sum([spec.num_leaves for spec in self.children_specs])

    def get_keychains(self, prefix="", sep="/"):
        keychains = []
        for key, child_spec in zip(self.context, self.children_specs):
            new_prefix = prefix + key + sep if prefix else key + sep
            if child_spec.children_specs:  # Non-leaf node
                keychains.extend(child_spec.get_keychains(new_prefix, sep))
            else:  # Leaf node
                keychains.append(new_prefix[: -len(sep)])
        return keychains

    def __repr__(self, indent: int = 0) -> str:
        repr_prefix: str = f"TreeSpec({self.type.__name__}, {self.context}, ["
        children_specs_str: str = ""
        if len(self.children_specs):
            indent += len(repr_prefix)
            children_specs_str += self.children_specs[0].__repr__(indent)
            children_specs_str += "," if len(self.children_specs) > 1 else ""
            children_specs_str += ",".join(
                [
                    "\n" + " " * indent + child.__repr__(indent)
                    for child in self.children_specs[1:]
                ]
            )
        repr_suffix: str = f"{children_specs_str}])"
        return repr_prefix + repr_suffix


class LeafSpec(TreeSpec):
    def __init__(self) -> None:
        super().__init__(None, None, [])
        self.num_leaves = 1

    def __repr__(self, indent: int = 0) -> str:
        return "*"


def tree_flatten(pytree: PyTree) -> Tuple[List[Any], TreeSpec]:
    """Flattens a pytree into a list of values and a TreeSpec that can be used
    to reconstruct the pytree."""
    if _is_leaf(pytree):
        return [pytree], LeafSpec()

    node_type = _get_node_type(pytree)
    flatten_fn = _dict_flatten
    child_pytrees, context = flatten_fn(pytree)

    # Recursively flatten the children
    result: List[Any] = []
    children_specs: List["TreeSpec"] = []
    for child in child_pytrees:
        flat, child_spec = tree_flatten(child)
        result += flat
        children_specs.append(child_spec)

    return result, TreeSpec(node_type, context, children_specs)


def tree_unflatten(values: List[Any], spec: TreeSpec) -> PyTree:
    """Given a list of values and a TreeSpec, builds a pytree.

    This is the inverse operation of `tree_flatten`.
    """
    if not isinstance(spec, TreeSpec):
        raise TypeError(
            f"tree_unflatten(values, spec): Expected `spec` to be instance of "
            f"TreeSpec but got item of type {type(spec)}."
        )
    if len(values) != spec.num_leaves:
        raise TypeError(
            f"tree_unflatten(values, spec): `values` has length {len(values)} "
            f"but the spec refers to a pytree that holds {spec.num_leaves} "
            f"items ({spec})."
        )
    if isinstance(spec, LeafSpec):
        return values[0]

    unflatten_fn = _dict_unflatten

    # Recursively unflatten the children
    start = 0
    end = 0
    child_pytrees = []
    for child_spec in spec.children_specs:
        end += child_spec.num_leaves
        child_pytrees.append(tree_unflatten(values[start:end], child_spec))
        start = end

    return unflatten_fn(child_pytrees, spec.context)


def serialize_obj(obj):
    if inspect.isclass(obj) or isinstance(obj, type):
        return {"cls_module": obj.__module__, "cls_name": obj.__name__}
    return obj


def recursive_serialize(d):
    if isinstance(d, dict):
        return {k: recursive_serialize(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [recursive_serialize(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(recursive_serialize(v) for v in d)
    else:
        return serialize_obj(d)


def deserialize_obj(serialized):
    if (
        isinstance(serialized, dict)
        and "cls_module" in serialized
        and "cls_name" in serialized
    ):
        module = __import__(serialized["cls_module"], fromlist=[serialized["cls_name"]])
        cls = getattr(module, serialized["cls_name"])
        return cls
    return serialized


def recursive_deserialize(d):
    if isinstance(d, dict) and "cls_module" not in d:
        return {k: recursive_deserialize(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [recursive_deserialize(v) for v in d]
    elif isinstance(d, tuple):
        return tuple(recursive_serialize(v) for v in d)
    else:
        return deserialize_obj(d)

class TorchModuleHelpers:

    def add_module(self, name: str, module: Optional["Model"]) -> None:
        if not isinstance(module, (Model, Layer, keras.Model,  keras.layers.Layer)) and module is not None:
            raise TypeError(f"{type(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {type(name)}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif "." in name:
            raise KeyError(f'module name can\'t contain ".", got: {name}')
        elif name == "":
            raise KeyError('module name can\'t be empty string ""')

        self._modules[name] = module

        super().__setattr__(name, module)

    def apply(self, fn: Callable[["Model"], None]):
        for module in self.children():
            if hasattr(module, "apply"):
                module.apply(fn)
            else:
                fn(module)
        fn(self)
        return self

    def _apply(self, fn, recurse=True):
        if recurse:
            if hasattr(self, "children"):
                for module in self.children():
                    if hasattr(module, "_apply"):
                        module._apply(fn)
        for key, param in self.v.items():
            if param is not None:
                self.v[key] = fn(param)
        for key, buf in self.buffers.items():
            if buf is not None:
                self.buffers[key] = fn(buf)
        return self

    def _named_members(
        self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
    ):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or id(v) in memo:
                    continue
                if remove_duplicate:
                    memo.add(id(v))
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v
                
    def register_module(self, name: str, module: Optional["Model"]) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Model":
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Model = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, (Model, Layer, keras.Model,  keras.layers.Layer)):
                raise TypeError("`" + item + "` is not a Module")

        return mod

    def get_parameter(self, target: str):
        target = target.replace(".", "/")
        return self.v[target]

    def parameters(self, recurse: bool = True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        gen = self._named_members(
            lambda module: module.v.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        gen = self._named_members(
            lambda module: module.buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Model"]:
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Model"]]:
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        memo = set()
        for name, module in self._module_dict.items():
            if module is not None and id(module) not in memo:
                memo.add(id(module))
                yield name, module

    def modules(self) -> Iterator["Model"]:
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["Model"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        if not getattr(self, "_built", False):
            self.build(
                *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
            )
        if memo is None:
            memo = set()
        if id(self) not in memo:
            if remove_duplicate:
                memo.add(id(self))
            yield prefix, self
            for name, module in self._module_dict.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                if not hasattr(module, "named_modules"):
                    yield submodule_prefix, self
                else:
                    yield from module.named_modules(
                        memo, submodule_prefix, remove_duplicate
                    )

    def _load_from_state_dict(
        self, state_dict, prefix, strict, missing_keys, unexpected_keys, error_msgs
    ):
        def _retrive_layer(model, key):
            if len(key.split(".")) == 1:
                return model, key

            module_path, weight_name = key.rsplit(".", 1)

            # Retrieve the layer using the module path
            layer = model
            for attr in module_path.split("."):
                layer = getattr(layer, attr)

            return layer, weight_name

        persistent_buffers = {k: v for k, v in self._buffers.items()}
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if not isinstance(input_param, tf.Tensor):
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        "expected ArrayLike object from checkpoint but "
                        f"received {type(input_param)}"
                    )
                    continue

                if not isinstance(input_param, KerasVariable):
                    input_param = KerasVariable(input_param)

                layer, weight_name = _retrive_layer(self, name)
                try:
                    setattr(layer, weight_name, input_param)
                except Exception as ex:
                    error_msgs.append(
                        f'While copying the parameter named "{key}", '
                        f"whose dimensions in the model are {param.shape} and "
                        f"whose dimensions in the checkpoint are {input_param.shape}, "
                        f"an exception occurred : {ex.args}."
                    )
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix) :].split(".", 1)
                    if len(input_name) > 1:
                        if input_name[0] not in self._modules:
                            unexpected_keys.append(key)
                    elif input_name[0] not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(
        self,
        state_dict: typing.Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        r"""Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

        If :attr:`strict` is ``True``, then
        the keys of :attr:`state_dict` must exactly match the keys returned
        by this module's :meth:`~Module.state_dict` function.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~Module.state_dict` function. Default: ``True``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing any keys that are expected
                    by this module but missing from the provided ``state_dict``.
                * **unexpected_keys** is a list of str containing the keys that are not
                    expected by this module but present in the provided ``state_dict``.
        """
        if not isinstance(state_dict, typing.Mapping):
            raise TypeError(
                f"Expected state_dict to be dict-like, got {type(state_dict)}."
            )

        missing_keys: List[str] = []
        unexpected_keys: List[str] = []
        error_msgs: List[str] = []

        state_dict = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x.numpy()),
            state_dict,
        )
        state_dict = OrderedDict(state_dict)

        def load(module, local_state_dict, prefix=""):
            module._load_from_state_dict(
                local_state_dict,
                prefix,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            # TODO: maybe we should implement this similar to PT
            # and make this recursive.

        load(self, state_dict)
        del load

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if strict:
            missing_keys = sorted(missing_keys)
            unexpected_keys = sorted(unexpected_keys)
            if len(missing_keys) > 0:
                warnings.warn(
                    "Missing key(s) in state_dict: {}\n".format(
                        ", ".join(f"'{k}'" for k in missing_keys)
                    )
                )
            if len(unexpected_keys) > 0:
                warnings.warn(
                    "Unexpected key(s) in state_dict: {}\n".format(
                        ", ".join(f"'{k}'" for k in unexpected_keys)
                    )
                )

    def requires_grad_(self, requires_grad: bool = True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def _get_name(self):
        return self.__class__.__name__
    
class ModelHelpers:
    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _get_first_array(*args, **kwargs):
        arr = None
        flattened_args = tf.nest.flatten((args, kwargs))
        arr_candidates = tf.nest.map_structure(
            lambda x: x if isinstance(x, (tf.Tensor, tf.Variable)) else False,
            flattened_args,
        )
        for arr_candidate in arr_candidates:
            if arr_candidate is not False:
                arr = arr_candidate
                break
        return arr

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _get_input_shapes(*args):
        input_shapes = []
        for x in args:
            if isinstance(x, (tf.Tensor, tf.Variable)):
                input_shapes.append(x.shape)
            else:
                try:
                    x = tf.convert_to_tensor(x)
                    input_shapes.append(x.shape)
                except Exception:
                    input_shapes.append(None)
        return input_shapes

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _extract_v(v, keychain_mappings: dict, orig_key_chain, /):
        if ModelHelpers._dict_has_key_chain(v, orig_key_chain):
            ret_cont = ModelHelpers._dict_at_key_chain(v, orig_key_chain)
        else:
            ret_cont = dict()
        for old_kc, new_kc in keychain_mappings.items():
            if orig_key_chain in old_kc:
                # Check if `v` contains `new_kc` before replacing in `ret_cont`
                if ModelHelpers._dict_has_key_chain(v, new_kc):
                    ret_cont = ModelHelpers._dict_set_at_key_chain(
                        ret_cont,
                        "/".join(old_kc.split("/")[1:]),
                        ModelHelpers._dict_at_key_chain(v, new_kc),
                    )
                else:
                    continue
        return ret_cont

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _remove_duplicate_variables(vs, created, /):
        created_ids = tf.nest.map_structure(lambda x: id(x), created)
        vs_ids = tf.nest.map_structure(lambda x: id(x), vs)
        ids = {}
        duplicate_keychains = []
        keychain_mappings = {}

        def unique_callback(x, kc):
            ids[x] = kc
            return x

        def found_dup_callback(x, kc):
            if ids[x] == kc:
                return x
            duplicate_keychains.append(kc)
            keychain_mappings[kc] = ids[x]
            return x

        created_ids = nest.map_structure_with_paths(
            lambda kc, x: unique_callback(x, kc), created_ids
        )
        vs_ids = nest.map_structure_with_paths(
            lambda kc, x: (
                unique_callback(x, kc) if x not in ids else found_dup_callback(x, kc)
            ),
            vs_ids,
        )
        for dup_kc in duplicate_keychains:
            vs = ModelHelpers._dict_prune_key_chain(vs, dup_kc)
        return vs, keychain_mappings

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _dict_set_at_key_chain(in_dict, key_chain, val, inplace=False):
        keys = re.split("[/.]", key_chain)
        if inplace:
            cont = in_dict
        else:
            cont = in_dict
        sub_cont = cont
        for key in keys[:-1]:
            if key not in sub_cont:
                sub_cont[key] = dict()
            sub_cont = sub_cont[key]
        sub_cont[keys[-1]] = val
        return cont

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _dict_at_key_chain(dict, key_chain, ignore_key_errors=False):
        keys = re.split("[/.]", key_chain)
        ret = dict
        for key in keys:
            try:
                ret = ret[key]
            except KeyError as e:
                if ignore_key_errors:
                    return
                raise Exception(repr(e))
        return ret

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _dict_has_key_chain(dict, key_chain):
        keys = re.split("[/.]", key_chain)
        ret = dict
        for key in keys:
            try:
                ret = ret[key]
            except KeyError:
                return False
        return True

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _dict_prune_key_chain(in_dict, key_chain):
        keys_in_chain = re.split("[/.]", key_chain)
        out_dict = {}
        for key, value in in_dict.items():
            if isinstance(value, dict):
                if key == keys_in_chain[0]:
                    if len(keys_in_chain) == 1:
                        new_val = []
                    else:
                        new_val = ModelHelpers._dict_prune_key_chain(
                            value,
                            "/".join(keys_in_chain[1:]),
                        )
                    if len(new_val) > 0:
                        out_dict[key] = new_val
                else:
                    if len(value) > 0:
                        out_dict[key] = value
            else:
                if len(keys_in_chain) != 1 or key != keys_in_chain[0]:
                    out_dict[key] = value
        return out_dict

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def _addindent(s_, numSpaces):
        s = s_.split("\n")
        # don't do anything for single-line stuff
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * " ") + line for line in s]
        s = "\n".join(s)
        s = first + "\n" + s
        return s


class Layer(tf.keras.layers.Layer, ModelHelpers, TorchModuleHelpers):
    _build_mode = None
    _with_partial_v = None
    _store_vars = True
    _built = False
    _v = None
    _buffers = None
    _module_dict = None
    _args = None
    _kwargs = None
    _module_graph = None
    _target = None
    _lazy_traced = False
    _training = None
    _dynamic_backend = None
    _device = None
    _dtype = None
    _previous_frame_info = None

    def __init__(
        self,
        /,
        *args,
        v=None,
        buffers=None,
        build_mode="on_init",
        store_vars=True,
        with_partial_v=False,
        dynamic_backend=None,
        training=True,
        dtype=None,
        device=None,
        module_dict=None,
        **kwargs,
    ):
        super(Layer, self).__init__(
            trainable=training,
            dtype=dtype,
        )
        if hasattr(self, 'forward'):
            self._call_signature = inspect.signature(self.forward)
        self._build_mode = build_mode
        self._with_partial_v = with_partial_v
        self._store_vars = store_vars
        self._built = False
        self._v_from_constructor = v if isinstance(v, dict) or v is None else dict(v)
        self._v = v if v is not None else dict()
        self._buffers = dict(buffers or {})
        self._module_dict = module_dict if module_dict is not None else dict()
        self._args = args
        self._kwargs = kwargs
        self._module_graph = None
        self._target = None
        self._lazy_traced = False
        self._training = training
        self._dynamic_backend = dynamic_backend
        self._device = device or "cpu"
        self._dtype = dtype or tf.float32
        if build_mode != "on_init":
            return
        self.build(*args, dynamic_backend=dynamic_backend, **kwargs)

    @tf.autograph.experimental.do_not_convert
    def _find_variables(
        self,
        /,
        *,
        obj=None,
        without_initialisation=False,
        _visited=None,
        trainable=True,
    ):
        _visited = _visited or {}
        vs = dict()
        if id(obj) in _visited:
            return vs
        _visited[id(obj)] = True
        if isinstance(obj, Layer) and obj is not self:
            fn = "_build_and_return_v" if trainable else "_build_and_return_buffers"
            if not obj._built and without_initialisation:
                return lambda: getattr(obj, fn)(
                    *obj._args, dynamic_backend=self._dynamic_backend, **obj._kwargs
                )

            return getattr(obj, fn)(
                *obj._args, dynamic_backend=obj._dynamic_backend, **obj._kwargs
            )
        elif isinstance(obj, tf.keras.layers.Layer) and obj is not self:
            return obj.v if trainable else obj.buffers

        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                    trainable=trainable,
                )
                if ret:
                    vs[f"v{str(i)}"] = ret
            return vs
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                    trainable=trainable,
                )
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
            return vs
        elif not hasattr(obj, "__dict__"):
            return vs
        for k, v in obj.__dict__.items():
            if (
                v is not None
                and k[0:2] != "__"
                and not k.startswith(
                    (
                        "_module_dict",
                        "_self_",
                        "_args",
                        "_kwargs",
                    )
                )
            ):
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                    trainable=trainable,
                )
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
        return vs

    @tf.autograph.experimental.do_not_convert
    def _find_buffers(self):
        if hasattr(self, "_module_dict"):
            for key, sub_module in self._module_dict.items():
                if len(sub_module._buffers) > 0:
                    self._buffers[key] = sub_module._buffers

    @tf.autograph.experimental.do_not_convert
    def _build_and_return_v(self, *args, **kwargs):
        if not self._built:
            self.build(*args, **kwargs)
        return self.v

    @tf.autograph.experimental.do_not_convert
    def _build_and_return_buffers(self, *args, **kwargs):
        if not self._built:
            self.build(*args, **kwargs)
        return self.buffers

    @tf.autograph.experimental.do_not_convert
    def _wrap_call_methods(
        self, keychain_mappings, /, *, key="", obj=None, _visited=None
    ):
        _visited = _visited or {}
        if id(obj) in _visited or not isinstance(key, str):
            return
        _visited[id(obj)] = True
        if isinstance(obj, Model) and obj is not self:
            orig_key_chain = key[1:] if key[0] == "_" else key

            obj.__call__ = self._fn_with_var_arg(
                obj.__call__, self._extract_v, keychain_mappings, orig_key_chain
            )
            return
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                self._wrap_call_methods(
                    keychain_mappings,
                    key=f"{key}/v{str(i)}",
                    obj=val,
                    _visited=_visited,
                )
            return
        elif isinstance(obj, dict):
            for k, val in obj.items():
                k = f"{key}/{k}" if key != "" and isinstance(k, str) else k
                self._wrap_call_methods(
                    keychain_mappings, key=k, obj=val, _visited=_visited
                )
            return
        for k, val in obj.module_dict.items():
            if k.startswith(("__", "_self_")):
                continue
            k = f"{key}/{k}" if key != "" else k
            if val is not None:
                self._wrap_call_methods(
                    keychain_mappings, key=k, obj=val, _visited=_visited
                )
        return

    @tf.autograph.experimental.do_not_convert
    def _compute_module_dict(self):
        self._module_dict = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, (Layer, tf.keras.layers.Layer)):
                if (
                    "stateful" in value.__module__
                    or hasattr(value, "_frontend_module")
                    or not hasattr(value, "_module_dict")
                ):
                    self._module_dict[key] = value
                else:
                    self._module_dict[key] = value._module_dict

    @tf.autograph.experimental.do_not_convert
    def _fn_with_var_arg_wrapper(
        self, *a, fn, v_fn, keychain_mappings, orig_key_chain, **kw
    ):
        if "v" in kw:
            del kw["v"]
        v = v_fn(self.v, keychain_mappings, orig_key_chain)
        return fn(*a, **kw, v=v)

    @tf.autograph.experimental.do_not_convert
    def _fn_with_var_arg(self, fn, v_fn, /, keychain_mappings, orig_key_chain):
        _fn_with_var_arg_wrapper = functools.partial(
            self._fn_with_var_arg_wrapper,
            fn=fn,
            v_fn=v_fn,
            keychain_mappings=keychain_mappings,
            orig_key_chain=orig_key_chain,
        )
        _fn_with_var_arg_wrapper.wrapped = True
        return _fn_with_var_arg_wrapper

    @tf.autograph.experimental.do_not_convert
    def _call(self, *args, v=None, buffers=None, **kwargs):
        if not self._built or not self.built:
            if not self._built:
                first_arr = self._get_first_array(*args, **kwargs)
                self.build(
                    *args,
                    **kwargs,
                    from_call=True,
                    dtype=first_arr.dtype if first_arr is not None else tf.float32,
                )

            if not self.built:
                # Don't use `keras` build method
                if os.environ.get("USE_KERAS_BUILD", "False").lower() == "false":
                    self.inputs = tf.nest.flatten(args)
                else:
                    input_shapes = self._get_input_shapes(*args)
                    if len(input_shapes) == 0:
                        input_shapes = tf.TensorShape(None)
                    elif len(input_shapes) == 1:
                        input_shapes = input_shapes[0]

                super(Layer, self).build(tf.TensorShape(None))  # noqa: UP008

        # If `v` was provided, replace with the module's v
        replace_v = False
        if v is not None:
            v_orig = self.v
            self._v = v
            replace_v = True

        # If `buffers` were provided, replace with the module's buffers
        replace_buffers = False
        if buffers is not None:
            buffers_orig = self.buffers
            self._buffers = buffers
            replace_buffers = True

        if replace_v or replace_buffers:
            # Call the forward pass
            ret = super(Layer, self).__call__(*args, **kwargs)  # noqa: UP008
            # Replace v, buffers if needed
            self._v = v_orig if replace_v else self._v
            self._buffers = buffers_orig if replace_buffers else self._buffers
            return ret
        elif hasattr(self.__call__, "wrapped"):
            return self.__call__(*args, **kwargs)

        # Get the signature of the forward method
        call_signature = inspect.signature(self.forward)

        # Convert all positional arguments to keyword arguments based on the signature
        new_kwargs = {}
        for idx, (param_name, param) in enumerate(call_signature.parameters.items()):
            if idx < len(args):
                new_kwargs[param_name] = args[idx]

        # Merge the existing kwargs
        new_kwargs.update(kwargs)
        return super(Layer, self).__call__(**new_kwargs)  # noqa: UP008

    @tf.autograph.experimental.do_not_convert
    def build(
        self,
        *args,
        from_call=False,
        device=None,
        dtype=None,
        dynamic_backend=None,
        **kwargs,
    ):
        self._built = True
        return

    def _lock_state(self):
        pass

    @tf.autograph.experimental.do_not_convert
    def register_buffer(self, name: str, value: Union[tf.Tensor, tf.Variable], persistent: bool = False):
        self._buffers.update({name: value})
        return value

    @tf.autograph.experimental.do_not_convert
    def register_parameter(self, name: str, value: Union[tf.Tensor, tf.Variable]):
        self._v.update({name: value})

    @tf.autograph.experimental.do_not_convert
    def train(self, mode: bool = True):
        self._training = mode
        for module in self.children():
            if isinstance(module, tf.keras.layers.Layer) and not hasattr(
                module, "train"
            ):
                module.trainable = mode
                continue
            module.train(mode)
        self.trainable = mode
        return self

    @tf.autograph.experimental.do_not_convert
    def eval(self):
        return self.train(mode=False)

    @tf.autograph.experimental.do_not_convert
    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_build_config(self):
        config = super().get_build_config()
        config = recursive_serialize(config)
        return config

    def build_from_config(self, config):
        config = recursive_deserialize(config)
        return super().build_from_config(config)

    def get_config(self):
        base_config = super().get_config()
        config = {}

        # Get the names and values of positional arguments in __init__
        init_signature = inspect.signature(self.__init__)
        arg_names = list(init_signature.parameters.keys())

        # Include the positional arguments in the config
        var_positional_arg_encountered = False
        var_positional_arg_name = None
        offset = 0
        for i, arg in enumerate(self._args):
            arg_name = arg_names[min(i, len(arg_names) - 1)]
            if var_positional_arg_encountered:
                config.update(
                    {
                        f"{var_positional_arg_name}_{i - offset}": arg,
                    }
                )
            elif (
                init_signature.parameters[arg_name].kind
                == inspect.Parameter.VAR_POSITIONAL
            ):
                var_positional_arg_encountered = True
                var_positional_arg_name = arg_name
                offset = i
                config.update(
                    {
                        f"{var_positional_arg_name}_{0}": arg,
                    }
                )
            else:
                config.update(
                    {
                        arg_name: arg,
                    }
                )

        # Include the keywords arguments in the config
        kwargs = self._kwargs.copy()
        kwargs.pop("devices", None)
        config.update(**kwargs)
        new_config = {**base_config, **config}
        new_config = recursive_serialize(new_config)
        return new_config

    @classmethod
    def from_config(cls, config):
        config = recursive_deserialize(config)
        # Get the signature of the __init__ method
        init_signature = inspect.signature(cls.__init__)
        arg_names = list(init_signature.parameters.keys())

        # Separate positional and keyword arguments based on the __init__ signature
        args = []
        pos_or_kw = OrderedDict()
        kwargs = {}
        var_positional_args = []
        for arg_name in arg_names:
            if (
                arg_name in config
                and init_signature.parameters[arg_name].kind
                == inspect.Parameter.KEYWORD_ONLY
            ):
                # Handle keyword arguments
                kwargs[arg_name] = config.pop(arg_name)
            elif (
                arg_name in config
                and init_signature.parameters[arg_name].kind
                == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                # Handle positional or keyword arguments
                pos_or_kw[arg_name] = config.pop(arg_name)
            elif any(re.match(rf"{arg_name}_\d+", key) for key in config.keys()):
                # Handle variable positional arguments
                var_positional_args.extend(
                    [
                        config.pop(key)
                        for key in sorted(config.keys())
                        if re.match(rf"{arg_name}_\d+", key)
                    ]
                )

        # Unpack positional arguments and the rest as keyword arguments
        config.pop("name", None)
        config.pop("trainable", None)
        config.pop("dtype", None)
        kwargs.update(config)

        # Determine the final args and kwargs
        if var_positional_args:
            args = list(pos_or_kw.values()) + var_positional_args
        else:
            kwargs.update(pos_or_kw)

        return cls(*args, **kwargs)

    # Methods to be Optionally Overridden #
    # -----------------------------------#

    @tf.autograph.experimental.do_not_convert
    def _create_variables(self, *, device=None, dtype=None):
        return {}

    @tf.autograph.experimental.do_not_convert
    def _build(self, *args, **kwargs) -> bool:
        return True

    @tf.autograph.experimental.do_not_convert
    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "When subclassing the `Module` class, you should "
            "implement a `_forward` method."
        )

    @tf.autograph.experimental.do_not_convert
    def _extra_repr(self) -> str:
        return ""

    # Properties #
    # -----------#

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def build_mode(self):
        return self._build_mode

    @property
    def training(self):
        return self._training

    @property
    def v(self):
        return self._v

    @property
    def buffers(self):
        return self._buffers

    @property
    def state_dict(self):
        return {**self.v, **self.buffers}

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def layers(self):
        return self._layers

    # Dunder Methods #
    # ---------------#
    @store_frame_info
    @tf.autograph.experimental.do_not_convert
    def __call__(
        self,
        *args,
        v=None,
        buffers=None,
        **kwargs,
    ):
        # TODO: Temp workaround to avoid `call`` from being transformed by AutoGraph
        if not hasattr(self.__class__.call, "autograph_info__"):
            setattr(self.__class__.call, "autograph_info__", True)
        ret = self._call(*args, v=v, buffers=buffers, **kwargs)
        return ret

    @tf.autograph.experimental.do_not_convert
    def __getattr__(self, name):
        if name == "v":
            if not super().__getattribute__("_v") and not getattr(  # noqa: E501
                self, "_built", False
            ):
                return self._build_and_return_v(
                    *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
                )

        _dict = super().__getattribute__("__dict__")
        if name in _dict:
            return _dict[name]

        elif "_v" in _dict and name in _dict["_v"]:
            return _dict["_v"][name]

        return super().__getattribute__(name)

    @tf.autograph.experimental.do_not_convert
    def __setattr__(self, name, value):
        if name in ["v", "buffers"]:
            name = "_" + name
        if isinstance(value, (Layer, tf.keras.layers.Layer)):
            _dict = getattr(self, "__dict__", None)
            if _dict:
                _dict[name] = value

            # compute the module dict
            self._compute_module_dict()

            obj_to_search = (
                None
                if not isinstance(value, (Layer, tf.keras.layers.Layer))
                else (
                    self._modules
                    if hasattr(self, "_modules") and self._modules
                    else self
                )
            )
            found_vars = self._find_variables(
                obj=obj_to_search,
                without_initialisation=(
                    True
                    if self._v_from_constructor and not self._with_partial_v
                    else False
                ),
            )
            flattened_v, v_spec = tree_flatten(found_vars)
            flattend_kc = v_spec.get_keychains()
            for kc, v in zip(flattend_kc, flattened_v):
                new_kc = kc.replace("/", ".")
                if new_kc not in self.v:
                    self.register_parameter(new_kc, v)

            # once all variables built, find and assign buffers
            found_buffers = self._find_variables(
                obj=obj_to_search,
                without_initialisation=(
                    True
                    if self._v_from_constructor and not self._with_partial_v
                    else False
                ),
                trainable=False,
            )
            flattened_buf, buf_spec = tree_flatten(found_buffers)
            flattend_kc = buf_spec.get_keychains()
            for kc, buf in zip(flattend_kc, flattened_buf):
                new_kc = kc.replace("/", ".")
                self.register_buffer(new_kc, buf)

            super().__setattr__(name, value)
            return
        elif isinstance(value, (tf.Variable, KerasVariable)) and not name.startswith("_"):
            _dict = getattr(self, "__dict__", None)
            if _dict:
                _dict[name] = value
            # Manual solution for cases where a `tf.int32` tensor
            # is placed on the GPU. TensorFlow doesn't have dedicated
            # kernels for placing `tf.int32` variables on the GPU and so
            # we manually cast them to `tf.int64` here otherwise due to
            # `tf.config.soft_device_placement(True)` by default,
            # TensorFlow puts the `tf.int32` variables on CPU which causes
            # unintended consequences downstream during tracing or
            # `tf.function` compilation e.g.
            # Ref: https://github.com/tensorflow/tensorflow/issues/9506
            # Ref: https://stackoverflow.com/questions/44813939/could-not-satisfy-explicit-device-specification-devicegpu0-because-no-devic
            dtype = (
                tf.int64
                if value.dtype == tf.int32 and "gpu:" in value.device.lower()
                else value.dtype
            )
            cast_dtype = dtype != value.dtype
            val = (
                value
                if not cast_dtype
                else KerasVariable(initial_value=tf.cast(value.value(), dtype), name=name)
            )
            self.register_parameter(name, val)
            super().__setattr__(name, val)
            return
        else:
            try:
                obj_to_search = getattr(self, name)
            except AttributeError:
                obj_to_search = None
            if isinstance(obj_to_search, Layer):
                # retrieve all hierarchical submodules
                assign_dict, kc = get_assignment_dict()

                # Iterate over all submods in assign_dict
                # updating their `v` and `buffers` with the
                # new value
                for key, submod in assign_dict.items():
                    # Get the subkey to match
                    subkey = kc[len(key) :].lstrip(".")

                    if hasattr(submod, "v"):
                        for v_key, v_value in submod.v.items():
                            if v_key.startswith(subkey):
                                submod.register_parameter(v_key, value)

                    # Repeat the same process for submod.buffers
                    if hasattr(submod, "buffers"):
                        for b_key, b_value in submod.buffers.items():
                            if b_key.startswith(subkey):
                                submod.register_buffer(b_key, value)

                # finally update the module dict
                self._module_dict[name] = value

            return super().__setattr__(name, value)

    @tf.autograph.experimental.do_not_convert
    def __delattr__(self, name):
        if hasattr(self, name):
            if isinstance(getattr(self, name), (Layer, tf.keras.layers.Layer)):
                super().__delattr__(name)
                return
        super().__delattr__(name)

    @tf.autograph.experimental.do_not_convert
    def __repr__(self):
        extra_lines = []
        extra_repr = self._extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key in self.v.keys():
            if isinstance(getattr(self, key, None), Layer):
                mod_str = repr(getattr(self, key))
                mod_str = self._addindent(mod_str, 2)
                child_lines.append(f"({key}): {mod_str}")
        lines = extra_lines + child_lines

        main_str = f"{self.__class__.__name__}("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Model(tf.keras.Model, ModelHelpers, TorchModuleHelpers):
    _build_mode = None
    _with_partial_v = None
    _store_vars = True
    _built = False
    _v = None
    _buffers = None
    _module_dict = None
    _args = None
    _kwargs = None
    _module_graph = None
    _target = None
    _lazy_traced = False
    _training = None
    _dynamic_backend = None
    _device = None
    _dtype = None
    _previous_frame_info = None

    def __init__(
        self,
        /,
        *args,
        v=None,
        buffers=None,
        build_mode="on_init",
        store_vars=True,
        with_partial_v=False,
        dynamic_backend=None,
        training=True,
        dtype=None,
        device=None,
        module_dict=None,
        **kwargs,
    ):
        super(Model, self).__init__(
            trainable=training,
            dtype=dtype,
        )
        if hasattr(self, 'forward'):
            self._call_signature = inspect.signature(self.forward)
        self._build_mode = build_mode
        self._with_partial_v = with_partial_v
        self._store_vars = store_vars
        self._built = False
        self._v_from_constructor = v if isinstance(v, dict) or v is None else dict(v)
        self._v = v if v is not None else dict()
        self._buffers = dict(buffers or {})
        self._module_dict = module_dict if module_dict is not None else dict()
        self._args = args
        self._kwargs = kwargs
        self._module_graph = None
        self._target = None
        self._lazy_traced = False
        self._training = training
        self._dynamic_backend = dynamic_backend
        self._device = device or "cpu"
        self._dtype = dtype or tf.float32
        if build_mode != "on_init":
            return
        self.build(*args, dynamic_backend=dynamic_backend, **kwargs)

    @tf.autograph.experimental.do_not_convert
    def _find_variables(
        self,
        /,
        *,
        obj=None,
        without_initialisation=False,
        _visited=None,
        trainable=True,
    ):
        _visited = _visited or {}
        vs = dict()
        if id(obj) in _visited:
            return vs
        _visited[id(obj)] = True
        if isinstance(obj, (Layer, Model)) and obj is not self:
            fn = "_build_and_return_v" if trainable else "_build_and_return_buffers"
            if not obj._built and without_initialisation:
                return lambda: getattr(obj, fn)(
                    *obj._args, dynamic_backend=self._dynamic_backend, **obj._kwargs
                )

            return getattr(obj, fn)(
                *obj._args, dynamic_backend=obj._dynamic_backend, **obj._kwargs
            )
        elif isinstance(obj, tf.keras.layers.Layer) and obj is not self:
            return obj.v if trainable else obj.buffers

        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                    trainable=trainable,
                )
                if ret:
                    vs[f"v{str(i)}"] = ret
            return vs
        elif isinstance(obj, dict):
            for k, v in obj.items():
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                    trainable=trainable,
                )
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
            return vs
        elif not hasattr(obj, "__dict__"):
            return vs
        for k, v in obj.__dict__.items():
            if (
                v is not None
                and k[0:2] != "__"
                and not k.startswith(
                    (
                        "_module_dict",
                        "_self_",
                        "_args",
                        "_kwargs",
                    )
                )
            ):
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
                    trainable=trainable,
                )
                if ret:
                    vs[k[1:] if k[0] == "_" else k] = ret
        return vs

    @tf.autograph.experimental.do_not_convert
    def _find_buffers(self):
        if hasattr(self, "_module_dict"):
            for key, sub_module in self._module_dict.items():
                if len(sub_module._buffers) > 0:
                    self._buffers[key] = sub_module._buffers

    @tf.autograph.experimental.do_not_convert
    def _build_and_return_v(self, *args, **kwargs):
        if not self._built:
            self.build(*args, **kwargs)
        return self.v

    @tf.autograph.experimental.do_not_convert
    def _build_and_return_buffers(self, *args, **kwargs):
        if not self._built:
            self.build(*args, **kwargs)
        return self.buffers

    @tf.autograph.experimental.do_not_convert
    def _wrap_call_methods(
        self, keychain_mappings, /, *, key="", obj=None, _visited=None
    ):
        _visited = _visited or {}
        if id(obj) in _visited or not isinstance(key, str):
            return
        _visited[id(obj)] = True
        if isinstance(obj, (Layer, Model)) and obj is not self:
            orig_key_chain = key[1:] if key[0] == "_" else key

            obj.__call__ = self._fn_with_var_arg(
                obj.__call__, self._extract_v, keychain_mappings, orig_key_chain
            )
            return
        elif isinstance(obj, (list, tuple)):
            for i, val in enumerate(obj):
                self._wrap_call_methods(
                    keychain_mappings,
                    key=f"{key}/v{str(i)}",
                    obj=val,
                    _visited=_visited,
                )
            return
        elif isinstance(obj, dict):
            for k, val in obj.items():
                k = f"{key}/{k}" if key != "" and isinstance(k, str) else k
                self._wrap_call_methods(
                    keychain_mappings, key=k, obj=val, _visited=_visited
                )
            return
        for k, val in obj.module_dict.items():
            if k.startswith(("__", "_self_")):
                continue
            k = f"{key}/{k}" if key != "" else k
            if val is not None:
                self._wrap_call_methods(
                    keychain_mappings, key=k, obj=val, _visited=_visited
                )
        return

    @tf.autograph.experimental.do_not_convert
    def _compute_module_dict(self):
        self._module_dict = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, (Layer, tf.keras.layers.Layer, Model, tf.keras.Model)):
                if (
                    "stateful" in value.__module__
                    or hasattr(value, "_frontend_module")
                    or not hasattr(value, "_module_dict")
                ):
                    self._module_dict[key] = value
                else:
                    self._module_dict[key] = value._module_dict

    @tf.autograph.experimental.do_not_convert
    def _fn_with_var_arg_wrapper(
        self, *a, fn, v_fn, keychain_mappings, orig_key_chain, **kw
    ):
        if "v" in kw:
            del kw["v"]
        v = v_fn(self.v, keychain_mappings, orig_key_chain)
        return fn(*a, **kw, v=v)

    @tf.autograph.experimental.do_not_convert
    def _fn_with_var_arg(self, fn, v_fn, /, keychain_mappings, orig_key_chain):
        _fn_with_var_arg_wrapper = functools.partial(
            self._fn_with_var_arg_wrapper,
            fn=fn,
            v_fn=v_fn,
            keychain_mappings=keychain_mappings,
            orig_key_chain=orig_key_chain,
        )
        _fn_with_var_arg_wrapper.wrapped = True
        return _fn_with_var_arg_wrapper

    @tf.autograph.experimental.do_not_convert
    def _call(self, *args, v=None, buffers=None, **kwargs):
        if not self._built or not self.built:
            if not self._built:
                first_arr = self._get_first_array(*args, **kwargs)
                self.build(
                    *args,
                    **kwargs,
                    from_call=True,
                    dtype=first_arr.dtype if first_arr is not None else tf.float32,
                )

            if not self.built:
                # Don't use `keras` build method
                if os.environ.get("USE_KERAS_BUILD", "False").lower() == "false":
                    self.inputs = tf.nest.flatten(args)
                else:
                    input_shapes = self._get_input_shapes(*args)
                    if len(input_shapes) == 0:
                        input_shapes = tf.TensorShape(None)
                    elif len(input_shapes) == 1:
                        input_shapes = input_shapes[0]

                super(Model, self).build(tf.TensorShape(None))  # noqa: UP008

        # If `v` was provided, replace with the module's v
        replace_v = False
        if v is not None:
            v_orig = self.v
            self._v = v
            replace_v = True

        # If `buffers` were provided, replace with the module's buffers
        replace_buffers = False
        if buffers is not None:
            buffers_orig = self.buffers
            self._buffers = buffers
            replace_buffers = True

        if replace_v or replace_buffers:
            # Call the forward pass
            ret = super(Model, self).__call__(*args, **kwargs)  # noqa: UP008
            # Replace v, buffers if needed
            self._v = v_orig if replace_v else self._v
            self._buffers = buffers_orig if replace_buffers else self._buffers
            return ret
        elif hasattr(self.__call__, "wrapped"):
            return self.__call__(*args, **kwargs)

        # Get the signature of the forward method
        call_signature = inspect.signature(self.forward)

        # Convert all positional arguments to keyword arguments based on the signature
        new_kwargs = {}
        for idx, (param_name, param) in enumerate(call_signature.parameters.items()):
            if idx < len(args):
                new_kwargs[param_name] = args[idx]

        # Merge the existing kwargs
        new_kwargs.update(kwargs)
        return super(Model, self).__call__(**new_kwargs)  # noqa: UP008

    @tf.autograph.experimental.do_not_convert
    def build(
        self,
        *args,
        from_call=False,
        device=None,
        dtype=None,
        dynamic_backend=None,
        **kwargs,
    ):
        self._built = True
        return

    def _lock_state(self):
        pass

    @tf.autograph.experimental.do_not_convert
    def register_buffer(self, name: str, value: Union[tf.Tensor, tf.Variable], persistent: bool = False):
        self._buffers.update({name: value})
        return value

    @tf.autograph.experimental.do_not_convert
    def register_parameter(self, name: str, value: Union[tf.Tensor, tf.Variable]):
        self._v.update({name: value})

    @tf.autograph.experimental.do_not_convert
    def train(self, mode: bool = True):
        self._training = mode
        for module in self.children():
            if isinstance(module, tf.keras.layers.Layer) and not hasattr(
                module, "train"
            ):
                module.trainable = mode
                continue
            module.train(mode)
        self.trainable = mode
        return self

    @tf.autograph.experimental.do_not_convert
    def eval(self):
        return self.train(mode=False)

    @tf.autograph.experimental.do_not_convert
    def call(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_build_config(self):
        config = super().get_build_config()
        config = recursive_serialize(config)
        return config

    def build_from_config(self, config):
        config = recursive_deserialize(config)
        return super().build_from_config(config)

    def get_config(self):
        base_config = super().get_config()
        config = {}

        # Get the names and values of positional arguments in __init__
        init_signature = inspect.signature(self.__init__)
        arg_names = list(init_signature.parameters.keys())

        # Include the positional arguments in the config
        var_positional_arg_encountered = False
        var_positional_arg_name = None
        offset = 0
        for i, arg in enumerate(self._args):
            arg_name = arg_names[min(i, len(arg_names) - 1)]
            if var_positional_arg_encountered:
                config.update(
                    {
                        f"{var_positional_arg_name}_{i - offset}": arg,
                    }
                )
            elif (
                init_signature.parameters[arg_name].kind
                == inspect.Parameter.VAR_POSITIONAL
            ):
                var_positional_arg_encountered = True
                var_positional_arg_name = arg_name
                offset = i
                config.update(
                    {
                        f"{var_positional_arg_name}_{0}": arg,
                    }
                )
            else:
                config.update(
                    {
                        arg_name: arg,
                    }
                )

        # Include the keywords arguments in the config
        kwargs = self._kwargs.copy()
        kwargs.pop("devices", None)
        config.update(**kwargs)
        new_config = {**base_config, **config}
        new_config = recursive_serialize(new_config)
        return new_config

    @classmethod
    def from_config(cls, config):
        config = recursive_deserialize(config)
        # Get the signature of the __init__ method
        init_signature = inspect.signature(cls.__init__)
        arg_names = list(init_signature.parameters.keys())

        # Separate positional and keyword arguments based on the __init__ signature
        args = []
        pos_or_kw = OrderedDict()
        kwargs = {}
        var_positional_args = []
        for arg_name in arg_names:
            if (
                arg_name in config
                and init_signature.parameters[arg_name].kind
                == inspect.Parameter.KEYWORD_ONLY
            ):
                # Handle keyword arguments
                kwargs[arg_name] = config.pop(arg_name)
            elif (
                arg_name in config
                and init_signature.parameters[arg_name].kind
                == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                # Handle positional or keyword arguments
                pos_or_kw[arg_name] = config.pop(arg_name)
            elif any(re.match(rf"{arg_name}_\d+", key) for key in config.keys()):
                # Handle variable positional arguments
                var_positional_args.extend(
                    [
                        config.pop(key)
                        for key in sorted(config.keys())
                        if re.match(rf"{arg_name}_\d+", key)
                    ]
                )

        # Unpack positional arguments and the rest as keyword arguments
        config.pop("name", None)
        config.pop("trainable", None)
        config.pop("dtype", None)
        kwargs.update(config)

        # Determine the final args and kwargs
        if var_positional_args:
            args = list(pos_or_kw.values()) + var_positional_args
        else:
            kwargs.update(pos_or_kw)

        return cls(*args, **kwargs)

    # Methods to be Optionally Overridden #
    # -----------------------------------#

    @tf.autograph.experimental.do_not_convert
    def _create_variables(self, *, device=None, dtype=None):
        return {}

    @tf.autograph.experimental.do_not_convert
    def _build(self, *args, **kwargs) -> bool:
        return True

    @tf.autograph.experimental.do_not_convert
    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "When subclassing the `Module` class, you should "
            "implement a `_forward` method."
        )

    @tf.autograph.experimental.do_not_convert
    def _extra_repr(self) -> str:
        return ""

    # Properties #
    # -----------#

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def build_mode(self):
        return self._build_mode

    @property
    def training(self):
        return self._training

    @property
    def v(self):
        return self._v

    @property
    def buffers(self):
        return self._buffers

    @property
    def state_dict(self):
        return {**self.v, **self.buffers}

    @property
    def module_dict(self):
        return self._module_dict

    # Dunder Methods #
    # ---------------#
    @store_frame_info
    @tf.autograph.experimental.do_not_convert
    def __call__(
        self,
        *args,
        v=None,
        buffers=None,
        **kwargs,
    ):
        # TODO: Temp workaround to avoid `call`` from being transformed by AutoGraph
        if not hasattr(self.__class__.call, "autograph_info__"):
            setattr(self.__class__.call, "autograph_info__", True)
        ret = self._call(*args, v=v, buffers=buffers, **kwargs)
        return ret

    @tf.autograph.experimental.do_not_convert
    def __getattr__(self, name):
        if name == "v":
            if not super().__getattribute__("_v") and not getattr(  # noqa: E501
                self, "_built", False
            ):
                return self._build_and_return_v(
                    *self._args, dynamic_backend=self._dynamic_backend, **self._kwargs
                )

        _dict = super().__getattribute__("__dict__")
        if name in _dict:
            return _dict[name]

        elif "_v" in _dict and name in _dict["_v"]:
            return _dict["_v"][name]

        return super().__getattribute__(name)

    @tf.autograph.experimental.do_not_convert
    def __setattr__(self, name, value):
        if name in ["v", "buffers"]:
            name = "_" + name
        if isinstance(value, (Layer, tf.keras.layers.Layer, Model, tf.keras.Model)):
            _dict = getattr(self, "__dict__", None)
            if _dict:
                _dict[name] = value

            # compute the module dict
            self._compute_module_dict()

            obj_to_search = (
                None
                if not isinstance(value, (tf.keras.layers.Layer, Layer, Model))
                else (
                    self._modules
                    if hasattr(self, "_modules") and self._modules
                    else self
                )
            )
            found_vars = self._find_variables(
                obj=obj_to_search,
                without_initialisation=(
                    True
                    if self._v_from_constructor and not self._with_partial_v
                    else False
                ),
            )
            flattened_v, v_spec = tree_flatten(found_vars)
            flattend_kc = v_spec.get_keychains()
            for kc, v in zip(flattend_kc, flattened_v):
                new_kc = kc.replace("/", ".")
                if new_kc not in self.v:
                    self.register_parameter(new_kc, v)

            # once all variables built, find and assign buffers
            found_buffers = self._find_variables(
                obj=obj_to_search,
                without_initialisation=(
                    True
                    if self._v_from_constructor and not self._with_partial_v
                    else False
                ),
                trainable=False,
            )
            flattened_buf, buf_spec = tree_flatten(found_buffers)
            flattend_kc = buf_spec.get_keychains()
            for kc, buf in zip(flattend_kc, flattened_buf):
                new_kc = kc.replace("/", ".")
                self.register_buffer(new_kc, buf)

            super().__setattr__(name, value)
            return
        elif isinstance(value, (tf.Variable, KerasVariable)) and not name.startswith("_"):
            _dict = getattr(self, "__dict__", None)
            if _dict:
                _dict[name] = value

            # Manual solution for cases where a `tf.int32` tensor
            # is placed on the GPU. TensorFlow doesn't have dedicated
            # kernels for placing `tf.int32` variables on the GPU and so
            # we manually cast them to `tf.int64` here otherwise due to
            # `tf.config.soft_device_placement(True)` by default,
            # TensorFlow puts the `tf.int32` variables on CPU which causes
            # unintended consequences downstream during tracing or
            # `tf.function` compilation e.g.
            # Ref: https://github.com/tensorflow/tensorflow/issues/9506
            # Ref: https://stackoverflow.com/questions/44813939/could-not-satisfy-explicit-device-specification-devicegpu0-because-no-devic
            dtype = (
                tf.int64
                if value.dtype == tf.int32 and "gpu:" in value.device.lower()
                else value.dtype
            )
            cast_dtype = dtype != value.dtype
            val = (
                value
                if not cast_dtype
                else KerasVariable(initial_value=tf.cast(value.value(), dtype), name=name)
            )
            self.register_parameter(name, val)
            super().__setattr__(name, val)
        else:
            try:
                obj_to_search = getattr(self, name)
            except AttributeError:
                obj_to_search = None
            if isinstance(obj_to_search, (Model, Layer)):
                # retrieve all hierarchical submodules
                assign_dict, kc = get_assignment_dict()

                # Iterate over all submods in assign_dict
                # updating their `v` and `buffers` with the
                # new value
                for key, submod in assign_dict.items():
                    # Get the subkey to match
                    subkey = kc[len(key) :].lstrip(".")

                    if hasattr(submod, "v"):
                        for v_key, v_value in submod.v.items():
                            if v_key.startswith(subkey):
                                submod.register_parameter(v_key, value)

                    # Repeat the same process for submod.buffers
                    if hasattr(submod, "buffers"):
                        for b_key, b_value in submod.buffers.items():
                            if b_key.startswith(subkey):
                                submod.register_buffer(b_key, value)

                # finally update the module dict
                self._module_dict[name] = value

            return super().__setattr__(name, value)

    @tf.autograph.experimental.do_not_convert
    def __delattr__(self, name):
        if hasattr(self, name):
            if isinstance(
                getattr(self, name),
                (Layer, tf.keras.layers.Layer, Model, tf.keras.Model),
            ):
                super().__delattr__(name)
                return
        super().__delattr__(name)

    @tf.autograph.experimental.do_not_convert
    def __repr__(self):
        extra_lines = []
        extra_repr = self._extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key in self.v.keys():
            if isinstance(getattr(self, key, None), (Layer, Model)):
                mod_str = repr(getattr(self, key))
                mod_str = self._addindent(mod_str, 2)
                child_lines.append(f"({key}): {mod_str}")
        lines = extra_lines + child_lines

        main_str = f"{self.__class__.__name__}("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str
