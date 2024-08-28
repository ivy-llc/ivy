# global
from __future__ import annotations
import re
import jax
from flax import nnx as nn
import jax.tree_util as tree
import jax.numpy as jnp
import functools
from typing import NamedTuple, Callable, Any, Tuple, List, Dict, Type
import inspect
from collections import OrderedDict


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


class ModelHelpers:
    @staticmethod
    def _get_first_array(*args, **kwargs):
        arr = None
        flattened_args, _ = jax.tree.flatten((args, kwargs))
        arr_candidates = jax.tree.map(
            lambda x: x if isinstance(x, (jax.Array)) else False,
            flattened_args,
        )
        for arr_candidate in arr_candidates:
            if arr_candidate is not False:
                arr = arr_candidate
                break
        return arr

    @staticmethod
    def _get_input_shapes(*args):
        input_shapes = []
        for x in args:
            if isinstance(x, (jax.Array)):
                input_shapes.append(x.shape)
            else:
                try:
                    x = jnp.asarray(x)
                    input_shapes.append(x.shape)
                except Exception:
                    input_shapes.append(None)
        return input_shapes

    @staticmethod
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
    def _remove_duplicate_variables(vs, created, /):
        created_ids = jax.tree.map(lambda x: id(x), created)
        vs_ids = jax.tree.map(lambda x: id(x), vs)
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

        created_ids = tree.tree_map_with_path(
            lambda kc, x: unique_callback(x, kc), created_ids
        )
        vs_ids = tree.tree_map_with_path(
            lambda kc, x: (
                unique_callback(x, kc) if x not in ids else found_dup_callback(x, kc)
            ),
            vs_ids,
        )
        for dup_kc in duplicate_keychains:
            vs = ModelHelpers._dict_prune_key_chain(vs, dup_kc)
        return vs, keychain_mappings

    @staticmethod
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


class Module(nn.Module, ModelHelpers):
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
        self._dtype = dtype or jnp.float32
        if build_mode != "on_init":
            return
        self.build(*args, dynamic_backend=dynamic_backend, **kwargs)

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

    def register_buffer(self, name: str, value: jax.Array):
        self._buffers.update({name: value})
        return value

    def register_parameter(self, name: str, value: jax.Array):
        self._v.update({name: value})

    def train(self, mode: bool = True):
        self._training = mode
        for module in self.children():
            if isinstance(module, nn.Module) and not hasattr(module, "train"):
                module.trainable = mode
                continue
            module.train(mode)
        self.trainable = mode
        return self

    def eval(self):
        return self.train(mode=False)

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError(
            "When subclassing the `Module` class, you should implement a `call` method."
        )

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
        for i, arg in enumerate(self._args[1:]):
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
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
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

    def _create_variables(self, *, device=None, dtype=None):
        return {}

    def _build(self, *args, **kwargs) -> bool:
        return True

    def _forward(self, *args, **kwargs):
        raise NotImplementedError(
            "When subclassing the `Module` class, you should "
            "implement a `_forward` method."
        )

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
    def __call__(
        self,
        *args,
        v=None,
        buffers=None,
        **kwargs,
    ):
        ret = self._call(v=v, *args, **kwargs)
        return ret

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

    def _compute_module_dict(self):
        self._module_dict = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, (Module, nn.Module)):
                if (
                    "stateful" in value.__module__
                    or hasattr(value, "_frontend_module")
                    or not hasattr(value, "_module_dict")
                ):
                    self._module_dict[key] = value
                else:
                    self._module_dict[key] = value._module_dict

    def __setattr__(self, name, value):
        if name in ["v", "buffers"]:
            name = "_" + name
        if isinstance(value, (Module, nn.Module)):
            _dict = getattr(self, "__dict__", None)
            if _dict:
                _dict[name] = value

            # compute the module dict
            self._compute_module_dict()

            obj_to_search = (
                None
                if not isinstance(value, (nn.Module, Module))
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
        else:
            try:
                obj_to_search = getattr(self, name)
            except AttributeError:
                obj_to_search = None
            if isinstance(obj_to_search, (nn.Module)):
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
        if isinstance(obj, (Module)) and obj is not self:
            fn = "_build_and_return_v" if trainable else "_build_and_return_buffers"
            if not obj._built and without_initialisation:
                return lambda: getattr(obj, fn)(
                    *obj._args, dynamic_backend=self._dynamic_backend, **obj._kwargs
                )
            return getattr(obj, fn)(
                *obj._args, dynamic_backend=obj._dynamic_backend, **obj._kwargs
            )
        elif isinstance(obj, nn.Module) and obj is not self:
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

    def _find_buffers(self):
        if hasattr(self, "_module_dict"):
            for key, sub_module in self._module_dict.items():
                if len(sub_module._buffers) > 0:
                    self._buffers[key] = sub_module._buffers

    def _build_and_return_v(self, *args, **kwargs):
        if not self._built:
            self.build(*args, **kwargs)
        return self.v

    def _build_and_return_buffers(self, *args, **kwargs):
        if not self._built:
            self.build(*args, **kwargs)
        return self.buffers

    def _wrap_call_methods(
        self, keychain_mappings, /, *, key="", obj=None, _visited=None
    ):
        _visited = _visited or {}
        if id(obj) in _visited or not isinstance(key, str):
            return
        _visited[id(obj)] = True
        if isinstance(obj, (Module)) and obj is not self:
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

    def _fn_with_var_arg_wrapper(
        self, *a, fn, v_fn, keychain_mappings, orig_key_chain, **kw
    ):
        if "v" in kw:
            del kw["v"]
        v = v_fn(self.v, keychain_mappings, orig_key_chain)
        return fn(*a, **kw, v=v)

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

    def _call(self, *args, v=None, buffers=None, **kwargs):
        if not self._built:
            first_arr = self._get_first_array(*args, **kwargs)
            self.build(
                *args,
                **kwargs,
                from_call=True,
                dtype=first_arr.dtype if first_arr is not None else jax.numpy.float32,
            )
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
            ret = super(Module, self).__call__(*args, **kwargs)  # noqa: UP008
            # Replace v, buffers if needed
            self._v = v_orig if replace_v else self._v
            self._buffers = buffers_orig if replace_buffers else self._buffers
            return ret
        elif hasattr(self.__call__, "wrapped"):
            return self.__call__(*args, **kwargs)
        return self.forward(*args, **kwargs)  # noqa: UP008

    def __delattr__(self, name):
        if hasattr(self, name):
            if isinstance(
                getattr(self, name),
                (Module, nn.Module),
            ):
                super().__delattr__(name)
                return
        super().__delattr__(name)

    def __repr__(self):
        extra_lines = []
        extra_repr = self._extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key in self.v.keys():
            if isinstance(getattr(self, key, None), (Module)):
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
