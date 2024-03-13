# global
from __future__ import annotations
import re
import os
import tensorflow as tf
import functools
import logging
from tensorflow.python.util import nest
from typing import NamedTuple, Callable, Any, Tuple, List, Dict, Type, Union
import inspect


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


class Model(tf.keras.Model, ModelHelpers):
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
        **kwargs,
    ):
        super(Model, self).__init__(
            trainable=training,
            dtype=dtype,
        )
        self._build_mode = build_mode
        self._with_partial_v = with_partial_v
        self._store_vars = store_vars
        self._built = False
        self._v_from_constructor = v if isinstance(v, dict) or v is None else dict(v)
        self._v = v if v is not None else dict()
        self._buffers = dict(buffers or {})
        self._module_dict = dict()
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
    ):
        _visited = _visited or {}
        vs = dict()
        if id(obj) in _visited:
            return vs
        _visited[id(obj)] = True
        if isinstance(obj, Model) and obj is not self:
            if not obj._built and without_initialisation:
                return lambda: obj._build_and_return_v(
                    *obj._args, dynamic_backend=self._dynamic_backend, **obj._kwargs
                )

            return obj._build_and_return_v(
                *obj._args, dynamic_backend=obj._dynamic_backend, **obj._kwargs
            )
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
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
                    )
                )
            ):
                ret = self._find_variables(
                    obj=v,
                    without_initialisation=without_initialisation,
                    _visited=_visited,
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
    def _assign_weights(self):
        model_weights = {}
        existing_ids = [id(w) for w in self.weights]

        # trainable weights
        flattened_v, v_spec = tree_flatten(self.v)
        flattened_kc = v_spec.get_keychains()
        new_weights = [None] * len(flattened_v)
        for i, (kc, x) in enumerate(zip(flattened_kc, flattened_v)):
            cast_dtype = False
            if x is not None:
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
                    if x.dtype == tf.int32 and "gpu:" in x.device.lower()
                    else x.dtype
                )
                cast_dtype = dtype != x.dtype
            new_weights[i] = (
                self.add_weight(name=kc, shape=x.shape, dtype=x.dtype, trainable=True)
                if x is not None and id(x) not in existing_ids
                else x
            )
            if isinstance(x, tf.Variable):
                val = x.value() if not cast_dtype else tf.cast(x.value(), dtype)
                new_weights[i].assign(val)
            if new_weights[i] is not None:
                model_weights[id(new_weights[i])] = new_weights[i].numpy()
        self.v = tree_unflatten(new_weights, v_spec)

        # non-trainable weights
        flattened_buf, buf_spec = tree_flatten(self.buffers)
        flattened_kc = buf_spec.get_keychains()
        new_buf = [None] * len(flattened_buf)
        for i, (kc, x) in enumerate(zip(flattened_kc, flattened_buf)):
            cast_dtype = False
            if x is not None:
                dtype = (
                    tf.int64
                    if x.dtype == tf.int32 and "gpu:" in x.device.lower()
                    else x.dtype
                )
                cast_dtype = dtype != x.dtype
            new_buf[i] = (
                self.add_weight(name=kc, shape=x.shape, dtype=x.dtype, trainable=False)
                if x is not None and id(x) not in existing_ids
                else x
            )
            if isinstance(x, tf.Variable):
                val = x.value() if not cast_dtype else tf.cast(x.value(), dtype)
                new_buf[i].assign(val)
            if new_buf[i] is not None:
                model_weights[id(new_buf[i])] = new_buf[i].numpy()
        self.buffers = tree_unflatten(new_buf, buf_spec)

        def _sort_weights(model_weights, weights):
            sorted_weights = []
            for weight in weights:
                sorted_weights.append(model_weights[id(weight)])
            return sorted_weights

        if model_weights:
            self.set_weights(_sort_weights(model_weights, self.weights))

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
        self._device = device or self._device
        self._dtype = dtype or self._dtype
        self._dynamic_backend = dynamic_backend or self._dynamic_backend
        # return False if not from_call but build_mode is on_call
        if not from_call and self._build_mode == "on_call":
            return self.v

        # build local Module, and any child modules flagged with "explicit" build mode
        # this gets the child modules initialised at best, their weights
        # remain un-generated
        built = self._build(*args, **kwargs) or True

        # this creates weights for this Module only
        created = self._create_variables(device=self._device, dtype=dtype)
        created = (
            created.cont_to_dict() if hasattr(created, "cont_to_dict") else created
        )

        # build variables based on locally built layers, if v not passed in constructor
        created_n_found = dict(
            **self._find_variables(
                obj=self,
                without_initialisation=(
                    True
                    if self._v_from_constructor and not self._with_partial_v
                    else False
                ),
            ),
            **created,
        )
        if self._v_from_constructor:
            # TODO: Add logic here for when `v` is passed in the constructor
            raise Exception("TODO: Implement this logic")
        else:
            self._v = created_n_found
        # remove duplicates
        self._v, keychain_mappings = self._remove_duplicate_variables(self._v, created)
        # build any child 'on_call' layers
        if not built and from_call:
            # TODO: Add logic for here
            raise Exception("TODO: Implement this logic")

        # flag built and remove local variables if specified
        self._built = bool(built)
        v_ret = self.v
        if not self._store_vars:
            # ToDo: verify variables in self.v are released once this method exits
            self._v = dict()

        # compute the module dict
        self._compute_module_dict()

        # once all variables built, find and assign buffers
        self._find_buffers()

        # also assign the keras model trainable and non-trainable weights now
        self._assign_weights()

        # wrap call methods if the model is fully built
        if built:
            self._wrap_call_methods(keychain_mappings, obj=self)

        return v_ret if bool(v_ret) or isinstance(built, bool) else built

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
        return super(Model, self).__call__(*args, **kwargs)  # noqa: UP008

    @tf.autograph.experimental.do_not_convert
    def _rebuild(self):
        logging.warning(
            "Building the module again as a trainable module was modified, "
            'please use the "explicit" or "on_call" build_modes instead '
            'of "on_init" to avoid repetitive building after each addition'
        )
        self._v = dict()
        self._built = False
        self.build(*self._args, **self._kwargs)

    @tf.autograph.experimental.do_not_convert
    def _compute_module_dict(self):
        self._module_dict = dict()
        for key, value in self.__dict__.items():
            if isinstance(value, Model):
                if "stateful" in value.__module__ or hasattr(value, "_frontend_module"):
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
    def register_buffer(self, name: str, value: Union[tf.Tensor, tf.Variable]):
        if value is not None:
            self._buffers.update({name: value})
        else:
            self.__setattr__(name, value)

    @tf.autograph.experimental.do_not_convert
    def register_parameter(self, name: str, value: Union[tf.Tensor, tf.Variable]):
        self._v.update({name: value})

    @tf.autograph.experimental.do_not_convert
    def train(self, mode: bool = True):
        self._training = mode
        for module in self.children():
            module.train(mode)
        self.trainable = mode
        return self

    @tf.autograph.experimental.do_not_convert
    def eval(self):
        return self.train(mode=False)

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError(
            "When subclassing the `Model` class, you should implement a `call` method."
        )

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
            "When subclassing the `Model` class, you should "
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
        if isinstance(value, Model):
            ret = super().__setattr__(name, value)
            if (
                hasattr(self, "_build_mode")
                and self.build_mode == "on_init"
                and getattr(self, "_built", False)
            ):
                self._rebuild()
            return ret
        elif isinstance(value, tf.Variable) and not name.startswith("_"):
            ret = self.register_parameter(name, value)
            if (
                hasattr(self, "_build_mode")
                and self.build_mode == "on_init"
                and getattr(self, "_built", False)
            ):
                self._rebuild()
            return ret
        return super().__setattr__(name, value)

    @tf.autograph.experimental.do_not_convert
    def __delattr__(self, name):
        if hasattr(self, name):
            if isinstance(getattr(self, name), Model):
                super().__delattr__(name)
                if self.build_mode == "on_init":
                    self._rebuild()
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
            if isinstance(getattr(self, key, None), Model):
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
