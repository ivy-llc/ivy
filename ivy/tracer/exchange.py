import importlib
import builtins
from typing import Callable, Mapping
import copy
import inspect
from functools import lru_cache

import ivy
import tracer.tracked_var_proxy as tvp
from tracer.conversion import to_native
from tracer.graph import Graph
from tracer.param import Param
from tracer.wrapping import Node


@lru_cache(maxsize=None)
def _get_backend_fn_from_path(fn_path: str):
    split_path = fn_path.split(".")
    if hasattr(tvp, split_path[0]):
        backend_fn = getattr(tvp, split_path[0])
    elif hasattr(builtins, split_path[0]):
        backend_fn = getattr(builtins, split_path[0])
    elif fn_path == "vmap":
        backend_fn = Node()
        backend_fn.__name__ = "vmap"
    elif "TensorShape" in split_path[0]:
        from tensorflow import TensorShape

        backend_fn = TensorShape
    else:
        backend_fn = importlib.import_module(split_path[0])
    for p in split_path[1:]:
        backend_fn = getattr(backend_fn, p)
    return backend_fn


def _convert_wrapped_function_to_dict(wrapped_fn: Node, func_to_id: Mapping):
    func_dict = wrapped_fn.__dict__.copy()
    if hasattr(wrapped_fn, "path"):
        func_dict["is_native_fn"] = True
    else:
        func_dict["is_native_fn"] = False
        if wrapped_fn.__name__ == "vmap":
            func_dict["path"] = "vmap"
        else:
            if hasattr(wrapped_fn.backend_fn, "__qualname__"):
                func_dict["path"] = wrapped_fn.backend_fn.__qualname__
            else:
                fn_name = getattr(wrapped_fn.backend_fn, "__name__", "")
                fn_module = getattr(wrapped_fn.backend_fn, "__module__", "")
                if fn_module and fn_name:
                    func_dict["path"] = fn_module + "." + fn_name
                if fn_name:
                    func_dict["path"] = fn_name
                else:
                    raise ValueError("Cannot find path for function")

    # TODO: The TrackedVarProxy classes are sometimes modified leading to pickle failures
    # We replace the TrackedVarProxy with the original class, this might cause future issues
    # If you face any such issues we need to track where the TrackedVarProxy is being modified
    output_param_types = []
    for param_type in func_dict["output_param_types"]:
        type_name = getattr(param_type, "__name__")
        if type_name.startswith("Tracked") and getattr(tvp, type_name, None) is not None:
            output_param_types.append(getattr(tvp, type_name))
    func_dict["output_param_types"] = output_param_types
    arg_param_types = []
    for param_type in func_dict["arg_param_types"]:
        type_name = getattr(param_type, "__name__")
        if type_name.startswith("Tracked") and getattr(tvp, type_name, None) is not None:
            arg_param_types.append(getattr(tvp, type_name))
    func_dict["arg_param_types"] = arg_param_types
    kwarg_param_types = []
    for param_type in func_dict["arg_param_types"]:
        type_name = getattr(param_type, "__name__")
        if type_name.startswith("Tracked") and getattr(tvp, type_name, None) is not None:
            kwarg_param_types.append(getattr(tvp, type_name))
    func_dict["kwarg_param_types"] = kwarg_param_types

    fns_in_id_list = []
    for fn in wrapped_fn.fns_in:
        fns_in_id_list.append(func_to_id[fn])
    func_dict["fns_id_in"] = fns_in_id_list

    fns_out_id_list = []
    for fn in wrapped_fn.fns_out:
        fns_out_id_list.append(func_to_id[fn])
    func_dict["fns_id_out"] = fns_out_id_list

    func_dict.pop("__repr__", None)
    func_dict.pop("backend_fn", None)
    func_dict.pop("fns_out", None)
    func_dict.pop("fns_in", None)
    func_dict.pop("prev_fn", None)

    return func_dict


def _wrap_backend_fn(backend_fn: Callable, func_dict: Mapping):
    wrapped_fn = Node()
    wrapped_fn.__dict__.update(func_dict)
    if not func_dict["is_native_fn"]:
        del wrapped_fn.path
    wrapped_fn.backend_fn = backend_fn

    wrapped_fn.__name__ = getattr(backend_fn, "__name__", None)
    if wrapped_fn.__name__ is None:
        if wrapped_fn.__qualname__:
            wrapped_fn.__name__ = wrapped_fn.__qualname__
        else:
            wrapped_fn.__name__ = getattr(func_dict, "fn_name", "<no name>")
    return wrapped_fn

def _convert_graph_to_dict(graph: Graph, is_subgraph: bool = False):
    """
    Serializes a Graph object to a simple dict.

    NOTE:
    1. currently memorisation of array objects is not supported.
    2. stateful objects are not supported

    Parameters
    ----------
    graph

    Returns
    -------
    ret
        a dict containing all the essential details to construct/transpile a graph
    """
    graph_dict = dict()

    graph_dict["_to_ivy"] = graph._to_ivy
    graph_dict["_with_numpy"] = graph._with_numpy
    graph_dict["backend"] = graph.backend
    graph_dict["_transpiling"] = graph._transpiling
    graph_dict["_is_trainable_module"] = graph._is_trainable_module
    graph_dict["_traced_train_modes"] = graph._traced_train_modes
    fn_signatures = {}
    for arg, sig in graph._fn_signature.items():
        if isinstance(sig, inspect.Parameter):
            non_annotated_sig = copy.copy(sig)
            non_annotated_sig._annotation = inspect.Parameter.empty()
            fn_signatures[arg] = non_annotated_sig
        else:
            fn_signatures[arg] = sig
    graph_dict["_fn_signature"] = fn_signatures

    args = ivy.nested_map(to_native, graph._args, include_derived=True)
    graph_dict["_args"] = args
    graph_dict["_arg_tracked_idxs"] = graph._arg_tracked_idxs
    graph_dict["_arg_param_ids"] = graph._arg_param_ids

    kwargs = ivy.nested_map(to_native, graph._kwargs, include_derived=True)
    graph_dict["_kwargs"] = kwargs
    graph_dict["_kwarg_tracked_idxs"] = graph._kwarg_tracked_idxs
    graph_dict["_kwarg_param_ids"] = graph._kwarg_param_ids

    graph_dict["_output_tracked_idxs"] = graph._output_tracked_idxs
    graph_dict["_output_param_ids"] = graph._output_param_ids

    graph_dict["parameters"] = dict()
    for id, param in graph._id_to_parameter.items():
        if param.ptype is not None:
            param_ptype = param.ptype
            ptype_name = getattr(param_ptype, "__name__")
            if ptype_name.startswith("Tracked") and getattr(tvp, ptype_name, None) is not None:
                param_ptype = getattr(tvp, ptype_name)
            graph_dict["parameters"][id] = param_ptype
        else:
            graph_dict["parameters"][id] = None

    # set tracked outputs to be None
    graph_dict["_output"] = dict()
    for mode, out in graph._output.items():
        if graph._output and graph._output_tracked_idxs:
            graph_dict["_output"][mode] = ivy.map_nest_at_indices(
                out,
                graph._output_tracked_idxs[mode],
                lambda x: None,
            )
        else:
            graph_dict["_output"][mode] = out

    if not is_subgraph:
        graph_dict["constants"] = copy.copy(graph.constants)
        if graph.backend == "jax":
            from ivy.functional.backends.jax.random import RNGWrapper

            for k, v in graph.constants.items():
                if isinstance(v, RNGWrapper):
                    graph_dict["constants"][k] = "rng_wrapper"
        graph_dict["_backend_compile"] = graph._backend_compile
        graph_dict["_static_argnums"] = graph._static_argnums
        graph_dict["_static_argnames"] = graph._static_argnames
        graph_dict["_compile_mode"] = graph._compile_mode
    # if graph._sub_graphs:
    graph_dict["_sub_graphs"] = {
        k: _convert_graph_to_dict(v, True) for k, v in graph._sub_graphs.items()
    }

    func_to_id = dict()
    func_to_id['train'] = {fn: id_ for id_, fn in graph._id_to_function["train"].items()}
    func_to_id['eval'] = {fn: id_ for id_, fn in graph._id_to_function["eval"].items()}

    func_obj_to_func_dict = dict()
    _all_function_ids = {"train": list(), "eval": list()}
    _id_to_function = {"train": dict(), "eval": dict()}

    for mode in ["train", "eval"]:
        id_to_func_dict = dict() 
        for key, wrapped_fn in graph._id_to_function[mode].items():
            func_dict = _convert_wrapped_function_to_dict(wrapped_fn, func_to_id[mode])
            func_obj_to_func_dict[wrapped_fn] = func_dict
            id_to_func_dict[key] = func_dict
            _id_to_function[mode] = id_to_func_dict

        function_ids = list()
        for wrapped_fn in graph._functions_dict[mode]:
            func_id = func_to_id[mode][wrapped_fn]
            function_ids.append(func_id)
        _all_function_ids[mode] = function_ids

    graph_dict['_id_to_function'] = _id_to_function
    graph_dict["_all_function_ids"] = _all_function_ids
    return graph_dict


# TODO: support Stateful and cached arrays
def _convert_dict_to_graph(
    graph_dict: Mapping, is_subgraph_dict: bool = False, graph: Graph = None
):
    # create an empty graph object
    if graph is None:
        graph = Graph.empty()

    graph._outer_connected = {"train": True, "eval": True}
    graph._to_ivy = graph_dict["_to_ivy"]
    graph._with_numpy = graph_dict["_with_numpy"]
    graph._args = graph_dict["_args"]
    graph._kwargs = graph_dict["_kwargs"]
    graph.backend = graph_dict["backend"]
    graph._transpiling = graph_dict["_transpiling"]
    graph._is_trainable_module = graph_dict["_is_trainable_module"]
    graph._traced_train_modes = graph_dict["_traced_train_modes"]
    fn_signatures = {}
    graph._fn_signature = graph_dict["_fn_signature"]

    if graph_dict["_sub_graphs"]:
        graph._sub_graphs = {
            k: _convert_dict_to_graph(v, True)
            for k, v in graph_dict["_sub_graphs"].items()
        }

    # TODO: infer ptype as well
    for id_, ptype in graph_dict["parameters"].items():
        graph._id_to_parameter[id_] = Param(ptype=ptype)

    graph._arg_tracked_idxs = graph_dict["_arg_tracked_idxs"]
    graph._arg_param_ids = graph_dict["_arg_param_ids"]

    graph._kwarg_tracked_idxs = graph_dict["_kwarg_tracked_idxs"]
    graph._kwarg_param_ids = graph_dict["_kwarg_param_ids"]

    graph._output_tracked_idxs = graph_dict["_output_tracked_idxs"]
    graph._output_param_ids = graph_dict["_output_param_ids"]

    graph._id_to_function = {"train": dict(), "eval": dict()}
    for mode, mode_dict in graph_dict["_id_to_function"].items():
        wrapped_fn_dict = dict()
        for id_, func_dict in mode_dict.items():
            backend_fn_path = func_dict["path"]
            backend_fn = _get_backend_fn_from_path(backend_fn_path)
            wrapped_fn = _wrap_backend_fn(backend_fn, func_dict)
            wrapped_fn_dict[id_] = wrapped_fn
        graph._id_to_function[mode] = wrapped_fn_dict

    graph._functions_dict = {"train": list(), "eval": list()}
    for mode, mode_list in graph_dict["_all_function_ids"].items():
        wrapped_fn_list = list()
        for func_id in mode_list:
            wrapped_fn = graph._id_to_function[mode][func_id]
            wrapped_fn_list.append(wrapped_fn)
        graph._functions_dict[mode] = wrapped_fn_list

    for mode in ["train", "eval"]:
        for func in graph._id_to_function[mode].values():
            fns_in = []
            fns_out = []
            for id in func.fns_id_in:
                fns_in.append(graph._id_to_function[mode][id])
            func.fns_in = fns_in
            for id in func.fns_id_out:
                fns_out.append(graph._id_to_function[mode][id])
            func.fns_out = fns_out

    graph._functions = graph._functions_dict["eval"]
    graph._output = graph_dict["_output"]

    if is_subgraph_dict:
        return graph

    graph.constants = graph_dict["constants"]
    if graph.backend == "jax":
        from ivy.functional.backends.jax.random import RNGWrapper

        for k, v in graph.constants.items():
            if isinstance(v, str) and v == "rng_wrapper":
                graph.constants[k] = RNGWrapper()
    graph._backend_compile = graph_dict["_backend_compile"]
    graph._static_argnums = graph_dict["_static_argnums"]
    graph._static_argnames = graph_dict["_static_argnames"]
    graph._compile_mode = graph_dict["_compile_mode"]
    graph.traced()
    from tracer.tracer import _backend_compile

    graph = _backend_compile(
        graph,
        graph._backend_compile,
        graph._static_argnums,
        graph._static_argnames,
        graph._compile_mode,
        *graph._args,
        **graph._kwargs
    )
    return graph
