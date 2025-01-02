# global
from typing import Callable, List, Optional, Any
from types import FunctionType
import os
import sys
import json
import pickle
import inspect
import collections
import numpy as np
import warnings

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None

# local
import ivy
from ivy.data_classes.array.conversions import _to_native
from . import tracked_var_proxy as tvp
from . import globals as glob
from . import source_gen as sg  # infuse
from .param import (
    Param,
    _get_unique_id,
    is_parameter,
    record_parameters_info,
    get_ids,
    get_var_flags,
    get_shapes,
    get_types,
)
from .helpers import (
    _is_tracked_variable,
    _find_missing_frontends,
    _format_missing_frontends_msg,
    NoParametersError,
)
from .conversion import (
    is_frontend_array,
    to_custom_numpy_type,
    to_native,
    nest_array_to_new_backend,
)
from .visualisation import (
    _args_str_from_fn,
    _output_str_from_fn,
    _param_to_label,
    _copy_func,
)


# Helpers #
# ------- #


def _is_trainable_module(fn, transpiling, subgraph):
    """Determines whether a given `fn` is a trainable module"""

    if subgraph:
        return False
    elif isinstance(fn, Graph) and fn._is_trainable_module:
        return transpiling
    elif hasattr(fn, "__self__") and (
        hasattr(fn.__self__, "_native_module") or isinstance(fn.__self__, ivy.Module)
    ):
        return True
    return False


def _trace_lazy_subgraphs(graph):
    """
    Traces any lazy subgraphs which remain uninstantiated after op logging of the
    main graph by attempting to infer the graph arguments from those passed to other
    callbacks of the higher-order function
    """
    for subgraph_list in graph._id_to_subgraphs[glob.current_trace_mode].values():
        for _, subgraph in subgraph_list:
            if not subgraph._initialized:
                inferred_args = None
                inferred_kwargs = None

                # try to infer the args from another callback of this higher order function
                for _, callback in subgraph_list:
                    if callback._initialized:
                        inferred_args = callback._eager_graph._args
                        inferred_kwargs = callback._eager_graph._kwargs
                        break

                if inferred_args is not None or inferred_kwargs is not None:
                    # initialize the lazy graph
                    subgraph._initialize(*inferred_args, **inferred_kwargs)
                else:
                    warnings.warn("a subgraph remains uninitialized")


def _graph_to_networkx(graph, ignore_skippable=False):
    """
    Create a networkx graph equivalent to the current graph, while also recursively converting any subgraphs to networkx
    """
    # TODO: support vmap

    networkx_graph = nx.DiGraph()
    subgraphs_dict = {}
    subgraphs = []

    for f in graph._functions:
        if (
            ignore_skippable
            and f.__name__ in glob.TRUNCATION_FNS_TO_IGNORE[ivy.current_backend_str()]
        ):
            continue
        if f.id_ in graph._id_to_subgraphs[glob.current_trace_mode]:
            subgraphs_list = graph._id_to_subgraphs[glob.current_trace_mode][f.id_]
            subgraphs_dict[f.id_] = []
            for _, subgraph in subgraphs_list:
                networkx_subgraph = _graph_to_networkx(
                    subgraph._eager_graph, ignore_skippable=ignore_skippable
                )
                subgraphs_dict[f.id_].append(networkx_subgraph)
                subgraphs.append(networkx_subgraph)

        args = []
        kwargs = {}
        i = 0

        # defines the type of args/kwargs in each position of the node (used for truncation node matching)
        for arg in f.args:
            if isinstance(arg, LazyGraph) and arg._initialized:
                args.append("subgraph")
            elif isinstance(arg, (list, tuple)):
                args.append("list")  # elements can be cached or tracked
            else:
                # TODO: there may be a more robust solution here, which doesn't risk having cached args in the wrong location
                if len(f.fns_in) > i:
                    args.append(f.fns_in[i].__name__)  # fn the arg originates from
                    i += 1
                else:
                    args.append(
                        "cached/initial"
                    )  # the arg is either cached, or this is the first node in the graph/subgraph that receives this arg

        for key, kwarg in f.kwargs.items():
            if isinstance(kwarg, LazyGraph) and kwarg._initialized:
                kwargs[key] = "subgraph"
            elif isinstance(kwarg, (list, tuple)):
                kwargs[key] = "list"
            else:
                if len(f.fns_in) > i:
                    kwargs[key] = f.fns_in[i].__name__
                    i += 1
                else:
                    kwargs[key] = "cached/initial"

        networkx_graph.add_node(
            f,
            name=f.__name__,
            args=args,
            tracked_kwargs=f.kwarg_tracked_idxs,
            subgraphs=subgraphs,
        )

    # add edges to the graph
    for f in graph._functions:
        if (
            ignore_skippable
            and f.__name__ in glob.TRUNCATION_FNS_TO_IGNORE[ivy.current_backend_str()]
        ):
            continue
        for out_fn in f.fns_out:
            networkx_graph.add_edge(
                f, out_fn, origin=f.__name__, target=out_fn.__name__
            )

    networkx_graph.graph["subgraphs"] = subgraphs_dict
    return networkx_graph


# Main #
# ---- #


class Graph:
    def __init__(
        self,
        fn: Callable,
        *args: Any,
        stateful: Optional[List] = None,
        arg_stateful_idxs: Optional[List] = None,
        kwarg_stateful_idxs: Optional[List] = None,
        include_generators: bool = True,
        array_caching: bool = True,
        with_numpy: bool = False,
        modes_to_trace: str = "all",
        transpiling: bool = False,
        to_ivy: bool = False,
        empty: bool = False,
        **kwargs: Any,
    ):
        # config
        self._is_subgraph = isinstance(self, SubGraph)
        self._is_trainable_module = _is_trainable_module(
            fn, transpiling, self._is_subgraph
        )
        self._include_generators = include_generators
        self._array_caching = array_caching
        self._with_numpy = with_numpy
        self._transpiling = transpiling
        self._orig_recursion_limit = sys.getrecursionlimit()
        self.backend = "ivy" if to_ivy else ivy.current_backend_str()
        self._to_ivy = to_ivy
        self._traced_train_modes = modes_to_trace
        self._to_ignore = tvp.get_types_to_ignore()
        self.contains_truncations = False
        self._array_mode = True
        self._container_mode = True

        # stateful
        self._stateful = ivy.default(stateful, [])
        self._stateful_classes = get_types(self._stateful)
        self._stateful_param_ids = get_ids(self._stateful, to_ivy)
        self._stateful_param_var_flags = get_var_flags(self._stateful)
        self._stateful_param_shapes = get_shapes(self._stateful)
        arg_stateful_idxs = ivy.default(arg_stateful_idxs, [])
        kwarg_stateful_idxs = ivy.default(kwarg_stateful_idxs, [])
        stateful_args = ivy.multi_index_nest(args, arg_stateful_idxs)
        stateful_kwargs = ivy.multi_index_nest(kwargs, kwarg_stateful_idxs)
        self._all_stateful = self._stateful + stateful_args + stateful_kwargs
        self._all_stateful_param_ids = [id(x) for x in self._all_stateful]
        self._stateful_clone_id_dict = dict(
            zip(self._all_stateful_param_ids, self._all_stateful_param_ids)
        )

        self._all_stateful_classes = get_types(self._all_stateful)

        # function being traced into a graph
        self._fn = fn
        if isinstance(fn, FunctionType):
            self.__name__ = fn.__name__
        elif isinstance(fn, object):
            self.__name__ = type(fn).__name__

        try:
            self._fn_signature = dict(inspect.signature(self._fn).parameters)
        except:
            self._fn_signature = {}

        # positional args
        args = to_custom_numpy_type(args, with_numpy=self._with_numpy)
        self._args = args
        args = to_native(args, cont_inplace=True, to_ignore=self._to_ignore)
        (
            self._arg_tracked_idxs,
            _,
            self._arg_param_ids,
            self._arg_param_types,
            self._arg_param_var_flags,
            self._arg_param_shapes,
        ) = record_parameters_info(
            args, to_ivy, with_numpy, stateful_idxs=arg_stateful_idxs
        )

        # key-word args
        kwargs = to_custom_numpy_type(kwargs, with_numpy=self._with_numpy)
        self._kwargs = kwargs
        kwargs = to_native(kwargs, cont_inplace=False, to_ignore=self._to_ignore)
        (
            self._kwarg_tracked_idxs,
            _,
            self._kwarg_param_ids,
            self._kwarg_param_types,
            self._kwarg_param_var_flags,
            self._kwarg_param_shapes,
        ) = record_parameters_info(
            kwargs, to_ivy, with_numpy, stateful_idxs=kwarg_stateful_idxs
        )

        if self._is_subgraph:
            [glob.subgraph_dependent_ids.add(id_) for id_ in self._arg_param_ids]
            [glob.subgraph_dependent_ids.add(id_) for id_ in self._kwarg_param_ids]
        else:
            [glob.dependent_ids["train"].add(id_) for id_ in self._arg_param_ids]
            [glob.dependent_ids["train"].add(id_) for id_ in self._kwarg_param_ids]
            [glob.dependent_ids["eval"].add(id_) for id_ in self._arg_param_ids]
            [glob.dependent_ids["eval"].add(id_) for id_ in self._kwarg_param_ids]

        if not self._is_subgraph:
            assert (
                empty
                or len(self._arg_tracked_idxs) + len(self._kwarg_tracked_idxs) != 0
            ), "No parameters detected in the inputs."

        # tracing storage
        self._id_to_function = {"train": dict(), "eval": dict()}
        self._id_to_parameter = dict()
        self._id_to_subgraphs = {"train": dict(), "eval": dict()}

        # add tracked inputs to graph
        ids = self._arg_param_ids + self._kwarg_param_ids + self._stateful_param_ids
        types = self._arg_param_types + self._kwarg_param_types + self._stateful_classes
        var_flags = (
            self._arg_param_var_flags
            + self._kwarg_param_var_flags
            + self._stateful_param_var_flags
        )
        shapes = (
            self._arg_param_shapes
            + self._kwarg_param_shapes
            + self._stateful_param_shapes
        )
        self._add_parameters(ids, types, var_flags, shapes)

        # output param ids
        self._output = {"train": [], "eval": []}
        self._output_tracked_idxs = {"train": [], "eval": []}
        self._output_param_ids = {"train": [], "eval": []}
        self.constants_in_output = {}
        if isinstance(fn, Graph):
            for k in fn.constants_in_output.keys():
                fn.constants[k] = to_native(fn.constants[k])
                self.constants_in_output[f"c{id(fn.constants[k])}"] = fn.constants[k]

        # graph connection
        self._outer_grouped = False
        self._all_grouped = False
        self._outer_connected = {"train": False, "eval": False}
        self._all_connected = False

        # functions in graph
        self._functions = list()
        self._functions_dict = {"train": [], "eval": []}
        self._terminal_ids = list()
        self._all_terminal_fns = {"train": [], "eval": []}
        self._all_grouped_functions = list()
        self._grouped_functions_by_height = collections.defaultdict(list)
        self._functions_sorted_by_time = {"train": False, "eval": False}
        self._functions_sorted_by_height = {"train": False, "eval": False}

        # create additional graph attributes for handling vmap nodes/subgraphs in general
        self._tmp_subgraph_id_to_function = [{}]
        self._subgraph_id_to_function = dict()
        self._sub_graphs = dict()
        self.vmap_node_ids = list()

        # misc graph attributes
        self._max_tree_height = -1
        self._contains_builtin_callables = False
        self._node_expansion = {}  # {original_node: [transpiled_node(s)]}

        # graph formatting
        self._inter_node_color = "#00CC00"  # same -> (0.0, 0.8, 0.0)
        self._stateful_node_color = "#E6B233"  # prev -> (0.9, 0.7, 0.2)
        self._io_node_color = "#8075FF"  # prev -> (0.4, 0.4, 1.0)
        self._var_node_color = "#FF6699"  # prev -> (1.0, 0.4, 0.6)
        self._node_size = 20
        self._input_functions = dict()

        self.training = False
        self.traced_fns = {"train": None, "eval": None}
        self.constants = dict()
        self._train_kwarg_name = self._get_train_kwarg_name()

    def to_device(self, device):
        ivy.set_backend(self.backend)
        self.constants = ivy.nested_map(
            lambda x: (
                ivy.as_native_dev(device)
                if isinstance(x, ivy.NativeDevice)
                else (ivy.to_native(ivy.to_device(x, device)) if ivy.is_array(x) else x)
            ),
            self.constants,
        )
        ivy.previous_backend()

    def _to_networkx(self, ignore_skippable=False):
        return _graph_to_networkx(self, ignore_skippable=ignore_skippable)

    def to_networkx(self):
        return self._to_networkx(ignore_skippable=False)

    @classmethod
    def empty(cls):
        "Initialize an empty Graph instance"
        return cls(fn=None, empty=True)

    # Properties #
    # ---------- #

    @property
    def _max_graph_height(self):
        return len(self._all_grouped_functions)

    @property
    def _callback_required_ids(self):
        """
        The ids of functions from the main graph which connect to any callback
        """
        required_ids = []
        for subgraph_list in self._id_to_subgraphs[glob.current_trace_mode].values():
            for _, subgraph in subgraph_list:
                for id_ in subgraph._eager_graph._subgraph_required_ids:
                    required_ids.append(id_)
        return required_ids

    @property
    def node_expansion(self):
        return self._node_expansion

    @node_expansion.setter
    def node_expansion(self, node_expansion: dict):
        self._node_expansion.update(node_expansion)

    # Getters and Setters #
    # ------------------- #

    @node_expansion.setter
    def node_expansion(self, node_expansion: dict):
        self._node_expansion.update(node_expansion)

    def _get_terminal_ids(self):
        if self._terminal_ids and not self._is_trainable_module:
            return self._terminal_ids

        self._terminal_ids = self._output_param_ids[glob.current_trace_mode]
        for id_, fn in self._id_to_function[glob.current_trace_mode].items():
            # append fn to terminal fns to use later
            # when connecting all nodes heightwise
            if fn.terminal:
                self._all_terminal_fns[glob.current_trace_mode].append(fn)
            # append id_ to terminal ids if the cond satisfies
            if (
                (fn.terminal and id_ in self._stateful_clone_id_dict)
                or (
                    fn.inplace_fn
                    and fn.from_tracked_var
                    and not fn.from_tracked_var_iterators
                )
                or fn.is_inplace_w_side_effects
            ) and fn.mode == glob.current_trace_mode:
                self._terminal_ids.append(id_)
        if len(self._terminal_ids) == 0:
            raise NoParametersError("No parameters detected in the outputs.")
        return self._terminal_ids

    def _sort_functions(self, time_chronological: bool = True):
        if not time_chronological:
            if not self._functions_sorted_by_height[glob.current_trace_mode]:
                self._functions_dict[glob.current_trace_mode] = [
                    fn for fns in self._all_grouped_functions for fn in fns
                ]
                self._functions_sorted_by_height[glob.current_trace_mode] = True
                self._functions_sorted_by_time[glob.current_trace_mode] = False
                self._functions = self._functions_dict[glob.current_trace_mode]
            return

        # store reference counts and populate self._functions
        # as well to avoid reiterating afterwards.
        if not self._functions_sorted_by_time[glob.current_trace_mode]:
            self._functions_dict[glob.current_trace_mode] = sorted(
                self._functions, key=lambda fn: fn.timestamp
            )
            self._functions_sorted_by_time[glob.current_trace_mode] = True
            self._functions_sorted_by_height[glob.current_trace_mode] = False
            self._functions = self._functions_dict[glob.current_trace_mode]

    def _add_parameters(self, ids, types, var_flags, shapes):
        for id_, type_, is_var, shape in zip(ids, types, var_flags, shapes):
            self._id_to_parameter[id_] = Param(type_, is_var, shape)

    def add_fn_to_dict(self, id_: int, fn: Callable):
        self._id_to_function[glob.current_trace_mode][id_] = fn
        if eval(os.getenv("CHECK_TRANSPILER_OVERHEAD", "False")) and self._transpiling:
            glob.node_expansion[glob.current_frontend].append(fn)

    def _train(self, update_glob: bool = True):
        if update_glob:
            glob.current_trace_mode = "train"
        self.training = True
        self._functions = self._functions_dict["train"]
        if self.traced_fns["train"] is not None:
            self._scripted_call = self.traced_fns["train"]

    def _eval(self, update_glob: bool = True):
        if update_glob:
            glob.current_trace_mode = "eval"
        self.training = False
        self._functions = self._functions_dict["eval"]
        if self.traced_fns["eval"] is not None:
            self._scripted_call = self.traced_fns["eval"]

    def train(self):
        """
        Sets the mode of the graph to train - changing the behaviour
        of certain functions like dropout and batch norm.
        """
        if not self._is_trainable_module:
            warnings.warn(
                "Attempted to set train mode of a graph which only has one mode"
            )
        self._train(update_glob=False)

    def eval(self):
        """
        Sets the mode of the graph to eval - changing the behaviour
        of certain functions like dropout and batch norm.
        """
        if not self._is_trainable_module:
            warnings.warn(
                "Attempted to set eval mode of a graph which only has one mode"
            )
        self._eval(update_glob=False)

    def replace_nodes(self, nodes_to_replace, replacement_node):
        """
        Replace given nodes with a single new node
        """
        ids_to_replace = []
        first_idx = 0

        for i, node in enumerate(nodes_to_replace):
            if i == 0:
                first_idx = self._functions.index(node)
            self._functions.remove(node)
            ids_to_replace.append(node.id_)

        self._functions.insert(first_idx, replacement_node)

    # Forward with Op Logging #
    # ----------------------- #

    def _compute_return(self):
        """Runs the forward pass and returns the final output.

        Example
        -------
        >>> import ivy
        >>> from tracer.tracer import _create_graph
        >>> ivy.set_backend("torch")
        >>> x = ivy.array([0., 32.])

        >>> def function(x):
        ...     y = ivy.mean(x)
        ...     z = ivy.sqrt(y)
        ...     return z

        >>> graph = _create_graph(function, x)
        >>> print(graph._compute_return())
        (ivy.array(4., dtype=float32),)
        """
        glob.tracing_paused = False
        ret = self._fn(*self._args, **self._kwargs)
        glob.tracing_paused = True

        if self._is_subgraph:
            self._id_to_function[glob.current_trace_mode] = glob.subgraph_id_to_fn
            glob.subgraph_id_to_fn = dict()

        return [ret]

    def _register_output(self, ret):
        """Record information about the final output `ret` of the forward pass."""
        self._output[glob.current_trace_mode] = ret
        self._output_tracked_idxs[glob.current_trace_mode] = ivy.nested_argwhere(
            ret,
            lambda x: is_parameter(x, with_numpy=self._with_numpy)
            or id(x) in self._all_stateful_param_ids,
            to_ignore=self._to_ignore,
        )
        self._output_param_ids[glob.current_trace_mode] = [
            _get_unique_id(to_native(x, to_ignore=self._to_ignore))
            for x in ivy.multi_index_nest(
                ret, self._output_tracked_idxs[glob.current_trace_mode]
            )
        ]

        # find any inputs which were fed directly to the output, and update id_ and add identity function
        for i, id_ in enumerate(self._output_param_ids[glob.current_trace_mode]):
            if id_ in self._arg_param_ids + self._kwarg_param_ids:

                def input_to_output(a, _):
                    return a

                # this is here to avoid circular imports
                from tracer.wrapping import _wrap_function_for_op_logging

                if id_ in self._arg_param_ids:
                    index = self._arg_param_ids.index(id_)
                    arg = ivy.index_nest(self._args, self._arg_tracked_idxs[index])
                else:
                    index = self._kwarg_param_ids.index(id_)
                    arg = ivy.index_nest(self._kwargs, self._kwarg_tracked_idxs[index])
                if is_frontend_array(arg):
                    arg = arg.ivy_array
                from_tracked_var = True if _is_tracked_variable(arg) else False
                input_to_output = _wrap_function_for_op_logging(
                    input_to_output, self, from_tracked_var=from_tracked_var
                )
                glob.tracing_paused = False
                ret = input_to_output(arg, None)
                glob.tracing_paused = True
                self._output_param_ids[glob.current_trace_mode][i] = _get_unique_id(ret)

    def _log_ops_train(self, is_training=None):
        if self._transpiling:
            self._fn._train()
        else:
            is_training = self._set_module_to_training()
        self._train()
        out = self._compute_return()
        self._register_output(out)
        # attempt to trace any remaining lazy subgraphs before we reload the sourcecode
        _trace_lazy_subgraphs(self)
        return is_training

    def _log_ops_eval(self):
        if self._transpiling:
            self._fn._eval()
        else:
            self._set_module_to_eval()
        self._eval()
        out = self._compute_return()
        self._register_output(out)
        _trace_lazy_subgraphs(self)

    def log_all_ops(self):
        """Run a forward pass with operation logging turned on,
        so that we can keep track of all the functions executed
        in the forward pass.
        """
        if self._is_trainable_module:
            is_training = None

            # log all ops for the branch for each mode to be traced
            if self._traced_train_modes == "all":
                is_training = self._log_ops_train()
                self._log_ops_eval()
            elif self._traced_train_modes == "train":
                is_training = self._log_ops_train()
            elif self._traced_train_modes == "eval":
                self._log_ops_eval()

            if is_training is not None:
                self._reset_module_mode(is_training)
        else:
            out = self._compute_return()
            self._register_output(out)
            _trace_lazy_subgraphs(self)

    def _reset_module_mode(self, is_training):
        """
        Resets the mode of the native module after operation logging has taken place.
        Only torch and paddle have the modes set through a method (rather than through
        an argument given during the module call), so we only reset the modes of torch
        or paddle modules here.
        """
        if self.backend == "torch":
            self._fn.__self__._native_module.train(is_training)
        if self.backend == "paddle":
            mod = self._fn.__self__._native_module
            mod.train() if is_training else mod.eval()

    def _set_fn_str(self, fn_str):
        self.__fn_str = fn_str

    def _set_module_to_training(self):
        """Sets the mode of the module being traced/transpiled to train"""
        if hasattr(self._fn.__self__, "_native_module"):
            if self.backend == "torch":
                is_training = self._fn.__self__._native_module.training
                self._fn.__self__._native_module.train()
                return is_training
            elif self.backend == "tensorflow":
                self._kwargs.update({"training": True})
            elif self.backend == "jax":
                if self._train_kwarg_name is not None:
                    self._kwargs.update({self._train_kwarg_name: True})
            elif self.backend == "paddle":
                is_training = self._fn.__self__._native_module.training
                self._fn.__self__._native_module.train()
                return is_training
        else:
            self._fn.__self__.train()

    def _set_module_to_eval(self):
        """Sets the mode of the native module being traced/transpiled to eval"""
        if hasattr(self._fn.__self__, "_native_module"):
            if self.backend == "torch":
                self._fn.__self__._native_module.eval()
            elif self.backend == "tensorflow":
                self._kwargs.update({"training": False})
            elif self.backend == "jax":
                if self._train_kwarg_name is not None:
                    self._kwargs.update({self._train_kwarg_name: False})
            elif self.backend == "paddle":
                self._fn.__self__._native_module.eval()
        else:
            self._fn.__self__.eval()

    def _get_train_kwarg_name(self):
        # get the name of the arg that defines the train mode for flax/haiku
        if self.backend == "jax":
            if self._kwargs is not None:
                # only transpile flax/haiku module with train and eval
                # branches if it has a kwarg for setting the train mode
                # NOTE: this only looks in the kwargs (not the args)
                for train_kwarg in glob.TRAIN_KWARGS:
                    if train_kwarg in self._kwargs.keys():
                        return train_kwarg
        return None

    # Graph creation #
    # -------------- #

    def _stop_tracking_constant_fns_output(self, receiving_fn, id_):
        if not ivy.exists(receiving_fn):
            idx = self._output_param_ids[glob.current_trace_mode].index(id_)
            del self._output_tracked_idxs[glob.current_trace_mode][idx]
            del self._output_param_ids[glob.current_trace_mode][idx]
            return
        receiving_fn.untrack_cached_args([id_])
        receiving_fn.untrack_cached_kwargs([id_])

    def _get_param_recursive(
        self,
        id_: int,
        receiving_fn: Optional[Callable] = None,
        from_subgraph: Optional[bool] = False,
    ):
        """This function is first called on the final output, and traverses backwards through
        the graph until we reach the inputs, keeping track of any parameters in _id_to_parameter
        and any functions called in the list _tmp_sub_functions.

        Parameters
        ----------
        id_
            parameter id
        receiving_fn
            the function which the parameter is inputted to. On the first call it is None, since
            of course the final outputs of the graph won't be inputted into any function.
        """
        # return if the parameter is already in the dict or if we reach the graph inputs (as the inputs
        # are already in _id_to_parameter)
        if id_ in self._id_to_parameter or (
            self._is_subgraph and id_ in self._greater_scope_ids
        ):
            return

        # obtain the function which generated the given output associated with `id_`
        if id_ in self._id_to_function[glob.current_trace_mode]:
            fn = self._id_to_function[glob.current_trace_mode][id_]
            if not self._include_generators and fn.is_generator:
                # raise exception if we try to delete the output of a gen fn which has
                # tracked vars in inputs when include_generators is False
                if fn.arg_tracked_idxs or fn.kwarg_tracked_idxs:
                    raise Exception(
                        f"including generator functions is not permitted, but func: {fn.__name__}"
                        f" contains tracked vars in inputs."
                    )
                self._stop_tracking_constant_fns_output(receiving_fn, id_)
                return
        elif self._array_caching:
            if from_subgraph:
                if ivy.exists(receiving_fn):
                    receiving_fn.untrack_cached_args([id_])
                    receiving_fn.untrack_cached_kwargs([id_])
            else:
                self._stop_tracking_constant_fns_output(receiving_fn, id_)
            return
        else:
            raise Exception(
                "array caching is not permitted, but id {} was not found in _id_to_functions.".format(
                    id_
                )
            )
        # call function recursively on all inputs to the function unless the function
        # is from tracked_var_iterators since __next__ function calls have the
        # iterator object in the arguments which is the output of __iter__ (whereas
        # we want to recurse back to the previous __next__ function call to correctly
        # chain __iter__ and consecutive __next__ nodes)
        if fn.from_iterator_chain and fn.prev_fn:
            [
                self._get_param_recursive(input_id, fn, from_subgraph=from_subgraph)
                for input_id in fn.prev_fn.output_param_ids
            ]
        else:
            [
                self._get_param_recursive(input_id, fn, from_subgraph=from_subgraph)
                for input_id in fn.arg_param_ids + fn.kwarg_param_ids
            ]
        # for any constant function when array caching is on, we delete its output as
        # an argument to any subsequent function calls (unless the next function in the
        # graph operates inplace- causing cached args to be changed on each call)
        next_fn_inplace = (
            ivy.exists(receiving_fn) and receiving_fn.is_inplace_w_side_effects
        )
        if fn.is_constant and id_ in self._terminal_ids and fn.terminal:
            self.constants_in_output.update(
                {f"c{id_}": fn.backend_fn(*fn.args, **fn.kwargs)}
            )
        elif (
            self._array_caching
            and not next_fn_inplace
            and (fn.is_constant or (not self._include_generators and fn.is_generator))
            and not (from_subgraph and fn.__name__ == "__iter__")
            # ^^ when recursing from a subgraph required id, do not cache __iter__,
            # as doing so can cause __next__ calls which are required by different fns
            # in the subgraph to not connect correctly to the iterator
            and (not from_subgraph or receiving_fn is not None)
            # ^^ we have the final condition here because we don't want to cache a function
            # which directly connects to a subgraph, as we cannot correctly untrack the
            # arguments from the receiving function(s) within the subgraph from here
        ):
            for receiving_fn in fn.fns_out:
                receiving_fn.untrack_cached_args(fn.output_param_ids)
                receiving_fn.untrack_cached_kwargs(fn.output_param_ids)
            if len(fn.fns_out) == 1:
                fn.args = None
                fn.kwargs = None
            fn.output_tracked_idxs.clear()
            fn.output_param_ids.clear()
            fn.output_param_types.clear()
            fn.output_param_shapes.clear()
        elif (
            fn.is_generator and not self._include_generators and not self._array_caching
        ):
            raise Exception(
                "Generator function {} detected, but include_generators and array_caching "
                "are both False".format(fn.__name__)
            )
        else:
            if self._is_subgraph:
                # store the ids required by the subgraph which are not provided by the args
                [
                    self._subgraph_required_ids.append(param_id)
                    for param_id in fn.arg_param_ids + fn.kwarg_param_ids
                ]
                [
                    self._subgraph_required_ids.remove(param_id)
                    for param_id in fn.output_param_ids
                    if param_id in self._subgraph_required_ids
                ]

            # keep track of the parameter and the function it came from
            self._functions_dict[glob.current_trace_mode].append(fn)
            self._add_parameters(
                fn.output_param_ids,
                fn.output_param_types,
                fn.output_param_var_flags,
                fn.output_param_shapes,
            )
        return

    def _group_functions(self, terminal_ids: List):
        """We try to figure out the most efficient order of execution
        for operations such that in a multiprocessing context, if we are
        at height h-1 we can start executing functions at height h asap.
        We then group the operations according to their heights in the graph.

        Parameters
        ----------
        terminal_ids
            ids of the final outputs
        """
        if not terminal_ids:
            return

        # for storing function heights
        def store_fn_heights(fn: Callable) -> int:
            if hasattr(fn, "tree_height"):
                return fn.tree_height
            heights_in = [
                store_fn_heights(fn_in)
                for fn_in in fn.fns_in
                if fn_in in self._functions_dict[glob.current_trace_mode]
            ]
            if heights_in:
                _height = max(heights_in) + 1
            else:
                _height = 0
            fn.tree_height = _height
            # maybe also store current height as the max tree height
            self._max_tree_height = max(self._max_tree_height, _height)
            # store the current node against its height
            self._grouped_functions_by_height[_height].append(fn)
            return _height

        # store function heights
        [
            store_fn_heights(self._id_to_function[glob.current_trace_mode][id_])
            for id_ in terminal_ids
        ]
        # group the functions based on their height in the tree from the starting leaf nodes
        grouped_functions = list()
        for height in range(0, self._max_tree_height + 1):
            fns = self._grouped_functions_by_height[height]
            # for functions at height 0, we want to execute the ones with more `fns_out` first
            # i.e. the function with the most functions at the next height which depend on it
            # should be executed first (this is only useful in a multiprocessing context)
            if height == 0:
                fns = sorted(
                    fns, key=lambda x: -len(x.fns_out) if hasattr(x, "fns_out") else 0
                )
            # at other heights, we want the order to be such that we call functions earlier if
            # they depend on a function at the height below which is called earlier. This is so
            # in a multiprocessing context we can make a start on functions at the next height asap
            else:
                fns_hm1 = grouped_functions[-1]
                leftmost_idxs = [
                    max(
                        enumerate(
                            [
                                fn in fn_hm1.fns_out
                                for fn_hm1 in fns_hm1
                                if hasattr(fn_hm1, "fns_out")
                            ]
                        ),
                        key=lambda x: x[1],
                    )[0]
                    for fn in fns
                ]
                fns = [
                    fn for fn, _ in sorted(zip(fns, leftmost_idxs), key=lambda x: x[1])
                ]
            grouped_functions.append(fns)
            # group the collected functions against their height
            if len(self._all_grouped_functions) == height:
                self._all_grouped_functions.append(list())
            self._all_grouped_functions[height].extend(fns)

    def _chain_functions(self, terminal_ids: List):
        """We recurse back from the outputs to the inputs,
        keeping track of all relevant parameters and
        functions in the graph.

        Parameters
        ----------
        terminal_ids
            ids of the final outputs
        """
        # recursively chain the graph via backward traversal from the outputs
        [self._get_param_recursive(id_) for id_ in terminal_ids.copy()]

        # chain from all the subgraph inputs, otherwise necessary parts of the main graph may not be included
        for subgraph_list in self._id_to_subgraphs[glob.current_trace_mode].values():
            for key, subgraph in subgraph_list:
                # if the lazy subgraph has been traced, recurse back from the ids that
                # need to be provided to the subgraph from the main graph
                if subgraph._initialized:
                    subgraph_required_ids = subgraph._eager_graph._subgraph_required_ids
                    [
                        self._get_param_recursive(id_, from_subgraph=True)
                        for id_ in subgraph_required_ids
                    ]

    def connect(
        self, output_connected_only: bool = True, time_chronological: bool = True
    ):
        """Connects functions together into the final graph.

        Example
        -------
        >>> import ivy
        >>> from tracer.tracer import _create_graph
        >>> ivy.set_backend("torch")
        >>> x = ivy.array([1.])

        >>> def toy_function(x):
        ...    y = ivy.sum(x)
        ...    z = ivy.prod(x)
        ...    a = ivy.sin(y)
        ...    b = ivy.cos(z)
        ...    c = ivy.tan(z)
        ...    i = ivy.round(a)
        ...    j = ivy.floor(b)
        ...    k = ivy.ceil(c)
        ...    return i, j, k

        Let us call connect() and view the resulting order in which the
        functions will be executed:

        >>> graph = _create_graph(toy_function, x)
        >>> graph.connect()
        >>> print([fn.__name__ for fn in graph._functions])
        ['prod', 'sum', 'cos', 'tan', 'sin', 'floor', 'ceil', 'round']

        We can see that the order of functions isn't the same as in `toy_function`,
        as they are in the order determined in `_chain_functions` (optimised for a
        multiprocessing context).
        """
        # python's max recursion depth of 1000 may not be sufficient when using deeper networks
        sys.setrecursionlimit(
            max(2 * len(self._id_to_function[glob.current_trace_mode]), 1000)
        )

        # get the terminal ids from which we want to recurse backwards to connect the nodes
        self._terminal_ids = self._get_terminal_ids()

        # recursively chain functions from the outputs to the inputs
        if not self._outer_connected[glob.current_trace_mode]:
            self._chain_functions(self._terminal_ids)
            self._outer_connected[glob.current_trace_mode] = True

        self._functions = self._functions_dict[glob.current_trace_mode]

        # check whether we are attempting to trace an empty graph
        if not self._functions and not self.constants_in_output:
            if not self._id_to_function:
                # this is an empty graph, so create a constant function
                def constant_fn(*args, **kwargs):
                    return self._output[True]

                self._scripted_call = constant_fn
                return
            else:
                assert False, "tried to trace a disconnected graph"

        if not output_connected_only:
            if not self._all_connected:
                # if we're here, we need to group all the outer functions
                # by height first
                if not self._outer_grouped:
                    self._group_functions(self._terminal_ids)
                    self._outer_grouped = True

                # now chain and group all the terminal functions
                for fn in self._all_terminal_fns[glob.current_trace_mode]:
                    self._chain_functions(fn.output_param_ids)
                    self._group_functions(fn.output_param_ids)
                self._all_connected = True
                self._all_grouped = True

            sys.setrecursionlimit(self._orig_recursion_limit)

            # sort the list of functions appearing in the graph height-wise
            # and return early
            self._sort_functions(time_chronological=time_chronological)
            return

        if not time_chronological and not self._outer_grouped:
            self._group_functions(self._terminal_ids)
            self._outer_grouped = True

        sys.setrecursionlimit(self._orig_recursion_limit)

        # sort the list of functions appearing in the graph chronologically or height-wise.
        self._sort_functions(time_chronological=time_chronological)

    # Traced Function #
    # ----------------- #

    def __call__(self, *args, **kwargs):
        if self._to_ivy:
            args = nest_array_to_new_backend(
                args, to_ignore=self._to_ignore, native=False
            )
            kwargs = nest_array_to_new_backend(
                kwargs, to_ignore=self._to_ignore, native=False
            )
            self.constants = nest_array_to_new_backend(
                self.constants, to_ignore=self._to_ignore, native=False
            )
        elif self._array_mode:
            args = ivy.nested_map(lambda x: _to_native(x, inplace=True), args)
            kwargs = ivy.nested_map(lambda x: _to_native(x, inplace=True), kwargs)
        if self._backend_compile and self._container_mode:
            args = ivy.nested_map(
                lambda x: x.cont_to_dict() if isinstance(x, ivy.Container) else x,
                args,
            )
            kwargs = ivy.nested_map(
                lambda x: x.cont_to_dict() if isinstance(x, ivy.Container) else x,
                kwargs,
            )
            self.constants = ivy.nested_map(
                lambda x: x.cont_to_dict() if isinstance(x, ivy.Container) else x,
                self.constants,
            )
        return self.graph_call(*args, **kwargs)

    def graph_call(self, *args, **kwargs):
        """Runs when calls to the traced graph are made.

        Examples
        --------
        >>> import ivy, time
        >>> from tracer.tracer import trace_graph
        >>> ivy.set_backend("torch")
        >>> x = ivy.array([1., 2., 3.])

        >>> def fn(x):
        ...     y = ivy.sum(x)
        ...     z = ivy.multiply(x, y)
        ...     return z

        >>> comp_fn, graph = trace_graph(fn, x, return_graph=True)

        The traced function `_call` runs quicker than the original `fn`:

        >>> start = time.time()
        >>> normal_ret = fn(x)
        >>> print(time.time() - start)
        0.0005664825439453125

        >>> start = time.time()
        >>> traced_ret = graph._call(x)
        >>> print(time.time() - start)
        0.00022029876708984375

        Of course they also give the same output:

        >>> print(normal_ret, traced_ret)
        ivy.array([ 6., 12., 18.], dtype=float32) ivy.array([ 6., 12., 18.], dtype=float32)
        """
        return self._scripted_call(*args, **kwargs, **self.constants)

    def traced(self, time_chronological: bool = True, frontend: str = None) -> Callable:
        """Returns the scripted function. If time_chronological is `True`, the order
        in which operations will be executed will be the same as the order of
        execution during the inital forward pass. If `False`, the order will be that
        which we constructed in `_chain_functions`.
        """
        # only connect graph if we haven't already
        if self._is_trainable_module and self._traced_train_modes in ["all", "train"]:
            # connect the train branch
            self._train()
            self.connect(time_chronological=time_chronological)
            self.reload_sourcecode(frontend=frontend, mode="train")
        if not self._is_trainable_module or self._traced_train_modes != "train":
            # connect the eval branch
            if not self._is_subgraph:
                self._eval()
            self.connect(time_chronological=time_chronological)
            self.reload_sourcecode(frontend=frontend)
        # return the handle of the call function
        return self.__call__

    def reload_sourcecode(self, frontend=None, mode="eval"):
        # generate source code
        source_generator = sg.SourceGenerator(self)
        fn_str = source_generator.generate_source(graph=self, frontend=frontend)
        constants = source_generator.get_constants()
        # define function
        if os.getenv("IVY_DEBUG_SOURCE", "False").lower() == "true":
            traced_fn = sg.load_fn_from_file(fn_str)
        else:
            traced_fn = sg.load_fn_from_str(fn_str)
        self._scripted_call = traced_fn
        self.__fn_str = fn_str
        self.constants.update(constants)
        self.traced_fns[mode] = traced_fn

    def obtain_sourcecode(self):
        return self.__fn_str, self.constants

    def initialize_from_cache(self, traced_fn, constants):
        assert traced_fn is not None, "traced_fn must be specified."
        assert constants is not None, "constants must be specified."
        self._scripted_call = traced_fn
        self.constants = constants

    def save(self, save_path: str) -> bool:
        """
        Serializes the traced function and saves it to a file.

        Parameters
        ----------
        save_path : str
            path to save the traced function to.

        Returns
        -------
        success : bool
            Indicates whether the save operation was successful or not.
        """
        with open(save_path, "wb") as f:
            pickle.dump(self, f)
        return True

    @classmethod
    def load(cls, load_path: str):
        """
        Loads the serialized graph dict from a file name
        Returns a new class intialized as this graph.

        Parameters
        ----------
        load_path : str
            path to load the traced function from.

        Returns
        -------
        new_graph : Graph
        """
        with open(load_path, "rb") as f:
            graph = pickle.load(f)
        return graph

    # Helpers #
    # ------- #

    def list_function_frequencies(
        self,
        return_raw: bool = False,
        return_str: bool = False,
        save_json: str = None,
    ) -> None:
        """Logs a list of the functions used in the graph.

        Parameters
        ----------
        return_raw : bool, optional
            if set to True, the list of function objects will be returned,
            default is False
        return_str : bool, optional
            if set to True, the message will be returned as a string instead of printed,
            default is False
        save_json : str, optional
            if specified, path of the JSON file where the used functions
            will be logged, default is None
        """
        backend_fns = [
            f.backend_fn for f in self._functions_dict[glob.current_trace_mode]
        ]
        if return_raw:
            return backend_fns
        paths = [
            fn.path
            for fn in self._functions_dict[glob.current_trace_mode]
            if hasattr(fn, "path")
        ]
        frequency = collections.Counter(paths).most_common()
        msg = "The functions being used are <(number of calls) function_path> : \n-> {}".format(
            "\n-> ".join(
                [" (" + str(freq[1]) + ") \t" + str(freq[0]) for freq in frequency]
            )
        )
        if save_json:
            with open(save_json, "w") as fp:
                data = {freq[0]: {"count": freq[1]} for freq in frequency}
                json.dump(data, fp, indent=4)
        if return_str:
            return msg
        else:
            print(msg)

    def list_missing_frontends(self, save_json: str = None) -> None:
        """Logs a list of the functions used in the graph that are currently missing
        a corresponding frontend function.

        Parameters
        ----------
        save_json : str, optional
            if specified, path of the JSON file where the missing functions
            will be logged, default is None
        """
        frequency = _find_missing_frontends(self)
        msg = _format_missing_frontends_msg(frequency)
        if save_json:
            with open(save_json, "w") as fp:
                data = {freq[0]: {"count": freq[1]} for freq in frequency}
                json.dump(data, fp, indent=4)
        else:
            print(msg)

    # Graph Visualization #
    # --------------------#

    def _is_stateful(self, f):
        if hasattr(f, "args"):
            for a in f.arg_param_types:
                if a in self._stateful_classes:
                    return True
        if hasattr(f, "kwargs"):
            for kwa in f.kwarg_param_types:
                if kwa in self._stateful_classes:
                    return True
        return False

    def _add_edge(
        self,
        g,
        func,
        id_in,
        idx,
        inp,
        num_inputs,
        with_edge_labels,
        with_arg_labels,
        with_output_labels,
    ):
        start_color = self._io_node_color
        start_title = ""
        if id_in in self._id_to_function[glob.current_trace_mode]:
            fn_in = self._id_to_function[glob.current_trace_mode][id_in]
            fn_id = fn_in.output_param_ids[0]
            start_color = self._inter_node_color
            start_title = f"{_args_str_from_fn(fn_in)}\n" if with_arg_labels else ""
            start_title = (
                start_title + _output_str_from_fn(fn_in)
                if with_output_labels
                else start_title
            )
        elif id_in in self._input_functions:
            fn_in = self._input_functions[id_in]
            fn_id = id_in
        else:
            fn_in = _copy_func(inp)
            idx0 = idx[0]
            sig = list(self._fn_signature.keys())
            if isinstance(idx0, str):
                arg_name = idx0
            elif "args" not in sig and isinstance(idx0, int) and idx0 < len(sig):
                arg_name = sig[idx0]
            else:
                arg_name = str(idx0)
            fnc_name = "input: " + arg_name
            idx1on = idx[1:]
            if idx1on:
                fnc_name += ", {}".format(idx1on)
            fn_in.__name__ = fnc_name
            fn_id = id_in
            self._input_functions[id_in] = fn_in
            num_inputs += 1
        # change color if is var
        start_color = (
            self._var_node_color
            if fn_id in self._id_to_parameter and self._id_to_parameter[fn_id].is_var
            else start_color
        )
        # add start node
        g.add_node(
            fn_id,
            label=fn_in.__name__,
            size=self._node_size,
            color=start_color,
        )
        if start_title != "":
            g.nodes[fn_id]["title"] = start_title
        # add end node
        end_title = f"{_args_str_from_fn(func)}\n" if with_arg_labels else ""
        end_title = (
            end_title + _output_str_from_fn(func) if with_output_labels else end_title
        )
        # change color if is var
        end_color = (
            self._var_node_color
            if func.output_param_ids[0] in self._id_to_parameter
            and self._id_to_parameter[func.output_param_ids[0]].is_var
            else self._inter_node_color
        )
        g.add_node(
            func.output_param_ids[0],
            label=func.__name__,
            size=self._node_size,
            color=end_color,
        )
        if end_title != "":
            g.nodes[func.output_param_ids[0]]["title"] = end_title
        edge_label = (
            _param_to_label(self._id_to_parameter[id_in]) if with_edge_labels else ""
        )
        g.add_edge(
            fn_id,
            func.output_param_ids[0],
            label=edge_label,
            arrowStrikethrough=False,
        )
        return num_inputs

    def _position_nodes(self, g, num_inputs, num_outputs, all_nodes, randomness_factor):
        pos_dict = dict()
        assert 0 <= randomness_factor <= 1

        # select position based on width and height of graph
        for height, nodes in enumerate(all_nodes):
            width = len(nodes)
            for w, n in enumerate(nodes):
                pos = np.array(
                    [
                        (height + 1) / (self._max_graph_height + 1),
                        0.5 if width == 1 else w / (width - 1),
                    ]
                )
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                h_delta = 0.5 / self._max_graph_height
                h_rand = np.random.uniform(-h_delta, h_delta)
                w_delta = 0.5 if width == 1 else 0.5 / (width - 1)
                w_delta_low = 0 if (w == 0 and width != 1) else -w_delta
                w_delta_high = 0 if (w == (width - 1) and width != 1) else w_delta
                w_rand = np.random.uniform(w_delta_low, w_delta_high)
                pos += np.array([h_rand, w_rand]) * randomness_factor
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                pos_dict[n[0]] = pos

        # add inputs
        if num_inputs > 0:
            input_idx = 0
            input_nodes = [
                n
                for n in g.nodes
                if n not in pos_dict and g.nodes[n]["label"][:5] == "input"
            ]
            min_output_y_coords = [
                min([pos_dict[e[1]][1] for e in g.edges if n in e]) for n in input_nodes
            ]
            input_nodes = [n for _, n in sorted(zip(min_output_y_coords, input_nodes))]
            for n in input_nodes:
                pos = np.array(
                    [0.0, 0.5 if num_inputs == 1 else input_idx / (num_inputs - 1)]
                )
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                h_delta = 0.5 / self._max_graph_height
                h_rand = np.random.uniform(0, h_delta)
                w_delta = 0.5 if num_inputs == 1 else 0.5 / (num_inputs - 1)
                w_delta_low = 0 if input_idx == 0 else -w_delta
                w_delta_high = 0 if input_idx == (num_inputs - 1) else w_delta
                w_rand = np.random.uniform(w_delta_low, w_delta_high)
                pos += np.array([h_rand, w_rand]) * randomness_factor
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                pos_dict[n] = pos
                input_idx += 1

        # add outputs
        if num_outputs > 0:
            output_idx = 0
            output_nodes = [
                n
                for n in g.nodes
                if n not in pos_dict and g.nodes[n]["label"][:6] == "output"
            ]
            min_input_y_coords = [
                min([pos_dict[e[0]][1] for e in g.edges if n in e])
                for n in output_nodes
            ]
            output_nodes = [n for _, n in sorted(zip(min_input_y_coords, output_nodes))]
            for n in output_nodes:
                pos = np.array(
                    [1.0, 0.5 if num_outputs == 1 else output_idx / (num_outputs - 1)]
                )
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                h_delta = 0.5 / self._max_graph_height
                h_rand = np.random.uniform(-h_delta, 0)
                w_delta = 0.5 if num_outputs == 1 else 0.5 / (num_outputs - 1)
                w_delta_low = 0 if output_idx == 0 else -w_delta
                w_delta_high = 0 if output_idx == (num_outputs - 1) else w_delta
                w_rand = np.random.uniform(w_delta_low, w_delta_high)
                pos += np.array([h_rand, w_rand]) * randomness_factor
                assert np.logical_and((0 <= pos), (pos <= 1)).all()
                pos_dict[n] = pos
                output_idx += 1

        return pos_dict

    def _populate_graph(
        self,
        g,
        functions,
        with_edge_labels,
        with_arg_labels,
        with_output_labels,
        output_connected_only,
        randomness_factor,
        pos=None,
    ):
        # config
        node_sep_x = (
            5 * self._node_size if not with_edge_labels else 10 * self._node_size
        )
        node_sep_y = 4 * self._node_size

        num_inputs = 0
        num_outputs = 0

        # add intermediate nodes
        def inp():
            pass

        for func in self._id_to_function[glob.current_trace_mode].values():
            if func not in functions and output_connected_only:
                continue
            for id_in, idx in zip(func.arg_param_ids, func.arg_tracked_idxs):
                num_inputs = self._add_edge(
                    g,
                    func,
                    id_in,
                    idx,
                    inp,
                    num_inputs,
                    with_edge_labels,
                    with_arg_labels,
                    with_output_labels,
                )
            for id_in, idx in zip(func.kwarg_param_ids, func.kwarg_tracked_idxs):
                num_inputs = self._add_edge(
                    g,
                    func,
                    id_in,
                    idx,
                    inp,
                    num_inputs,
                    with_edge_labels,
                    with_arg_labels,
                    with_output_labels,
                )

        # add output nodes and edges
        for id_ in self._output_param_ids[glob.current_trace_mode]:
            # change color if is var
            color = (
                self._var_node_color
                if id_ in self._id_to_parameter and self._id_to_parameter[id_].is_var
                else self._io_node_color
            )
            g.add_node(
                f"{id_}_output",
                label=f"output\n"
                + _output_str_from_fn(
                    self._id_to_function[glob.current_trace_mode][id_]
                ),
                size=self._node_size,
                color=color,
            )
            edge_label = (
                _param_to_label(self._id_to_parameter[id_]) if with_edge_labels else ""
            )
            g.add_edge(
                id_,
                f"{id_}_output",
                label=edge_label,
            )
            num_outputs += 1

        # calculate node positions
        all_nodes = list()
        max_graph_width = 0
        for fns in self._all_grouped_functions:
            nodes = [
                (f.output_param_ids[0], f, _args_str_from_fn(f), _output_str_from_fn(f))
                for f in fns
            ]
            seen = set()
            nodes = [n for n in nodes if not (n in seen or seen.add(n))]
            max_graph_width = max(max_graph_width, len(nodes))
            all_nodes.append(nodes)
        pos = ivy.default(
            pos,
            self._position_nodes(
                g, num_inputs, num_outputs, all_nodes, randomness_factor
            ),
        )
        # scale pos
        _x = lambda x: int(x * node_sep_x * (len(g.nodes)))
        _y = lambda y: int(y * node_sep_y * max_graph_width)
        pos = {n: [_x(p[0]), _y(p[1])] for n, p in pos.items() if n in g.nodes}

        # assert all positions are accounted for, if provided
        if ivy.exists(pos):
            assert min([n in pos for n in g.nodes])

        # Add position to nodes
        for id_ in g.nodes():
            g.nodes[id_]["x"] = pos[id_][0]
            g.nodes[id_]["y"] = pos[id_][1]

        # change color for stateful nodes
        stateful_nodes = [
            n
            for n in g.nodes
            if (
                n in self._id_to_function[glob.current_trace_mode]
                and self._is_stateful(self._id_to_function[glob.current_trace_mode][n])
            )
        ]
        for sn in stateful_nodes:
            g.nodes[sn]["color"] = self._stateful_node_color

        # draw
        # ToDo: plt.draw_if_interactive() # check if it works in a notebook

        return

    def show(
        self,
        save_to_disk=False,
        notebook=False,
        with_edge_labels=True,
        with_arg_labels=True,
        with_output_labels=True,
        output_connected_only=True,
        randomness_factor=0.1,
        highlight_subgraph=None,
        fname=None,
    ):
        # ToDo: deal with highlight subgraph behaviour
        # ToDo: remove input_to_output_link from the graph
        # ToDo: add support to icons in small graphs maybe?
        # ToDo: add color based on the backend

        # TODO: add good visualisation for tf.while_loop, tf.cond, etc - which we can now trace

        # assert that required visualization packages are installed
        if not ivy.exists(nx):
            raise Exception(
                "networkx python package must be installed in order to visualize computation graphs."
            )
        if not ivy.exists(Network):
            raise Exception(
                "pyvis python package must be installed in order to visualize computation graphs."
            )

        # ensure graph is connected
        if not self._outer_grouped or (
            not output_connected_only and not self._all_grouped
        ):
            self.connect(
                output_connected_only=output_connected_only, time_chronological=False
            )

        # create directed networkX graph
        g = nx.DiGraph()
        subgraph_g = list()

        # create a pyvis Network
        notebook_kwargs = {}
        if notebook:
            notebook_kwargs = {
                "height": "300px",
                "notebook": True,
                "cdn_resources": "in_line",
            }
        nt = Network(directed=True, **notebook_kwargs)

        # build the graph
        all_functions = self._functions
        self._populate_graph(
            g,
            all_functions,
            with_edge_labels,
            with_arg_labels,
            with_output_labels,
            output_connected_only,
            randomness_factor,
        )

        # add nodes and edges from the main graph
        for node, node_attrs in g.nodes(data=True):
            nt.add_node(node, **node_attrs)

        for source, dest, edge_attrs in g.edges(data=True):
            nt.add_edge(source, dest, **edge_attrs)

        # add nodes and edges from the subgraph
        if self._sub_graphs:
            from tracer.special_ops.vmap_helpers import (
                _handle_vmap_nodes,
                _remove_input_subgraph_nodes,
                _get_all_graph_functions,
            )

            # reorder the subgraphs
            all_functions = _get_all_graph_functions(self)
            ordered_subgraphs = [
                self._sub_graphs[id(fn)]
                for fn in all_functions
                if fn.__name__ == "vmap"
            ]

            for subgraph in ordered_subgraphs:
                subg = nx.DiGraph()
                subgraph_g.append(subg)

                subgraph._populate_graph(
                    subg,
                    subgraph._functions,
                    with_edge_labels,
                    with_arg_labels,
                    with_output_labels,
                    output_connected_only,
                    randomness_factor,
                )
                subgraph_nodes = _remove_input_subgraph_nodes(subg.nodes(data=True))
                for node, node_attrs in subgraph_nodes:
                    if not "output" in node_attrs["label"]:
                        node_attrs["color"] = "#b6f1f1"
                        node_attrs["borderWidth"] = 2
                        node_attrs["borderWidthSelected"] = 4

                    nt.add_node(node, **node_attrs)

                subgraph_edges = [
                    (source, dest, attrs)
                    for source, dest, attrs in subg.edges(data=True)
                    if not any(
                        "input" in subg.nodes[node]["label"] for node in (source, dest)
                    )
                ]
                for source, dest, edge_attrs in subgraph_edges:
                    nt.add_edge(source, dest, **edge_attrs)

            vmap_tuples = _handle_vmap_nodes(g, subgraph_g, all_functions)

            for vmap_node in vmap_tuples:
                nt.add_edge(
                    vmap_node[1],
                    vmap_node[0],
                    dashes=True,
                )
                nt.add_edge(
                    vmap_node[2],
                    vmap_node[1],
                    dashes=True,
                )

        # maybe highlight sub-graph (ToDo) self._functions?
        # if isinstance(highlight_subgraph, int):
        #     # set node color as red
        #     self._inter_node_color = (0.8, 0.0, 0.0)
        #     self._stateful_node_color = (0.8, 0.0, 0.0)
        #     self._io_node_color = (0.8, 0.0, 0.0)
        #     self._edge_color = (0.4, 0.0, 0.0)
        #
        #     # show highlighted sub-graph
        #     subgraph_id = list(self._functions.keys())[highlight_subgraph]
        #     self._show_for_functions(
        #         ax,
        #         self._functions[subgraph_id],
        #         with_edge_labels,
        #         with_arg_labels,
        #         with_output_labels,
        #         True,
        #         randomness_factor,
        #         False,
        #         cv2_labels,
        #         pos=pos,
        #     )

        # create a pyvis Network
        notebook_kwargs = {}
        if notebook:
            notebook_kwargs = {
                "height": "300px",
                "notebook": True,
                "cdn_resources": "in_line",
            }
        nt = Network(directed=True, **notebook_kwargs)
        # populates the nodes and edges data structures from the networkx graph
        nt.from_nx(g)
        nt.set_options(
            """
        const options = { "edges" : { "color": { "inherit": "both" }, 
                                      "smooth": false},
                          "physics": {"enabled": false}}
            """
        )

        # maybe save to disk -> nt.shows saves the file by default,
        # so to visualize the graph, save_to_disk must be set. This should
        # maybe be revisited
        if save_to_disk or fname:
            fname = ivy.default(
                fname,
                "graph_{}.html".format(
                    "".join(
                        [f.__name__.replace("_", "")[0] for f in self._functions][0:20]
                    )
                ),
            )
            if fname[-5:] != ".html":
                if "." in fname:
                    fname = ".".join(fname.split(".")[:-1])
                fname += ".html"
            if notebook:
                return nt.show(fname)
            nt.save_graph(fname)

    def __getstate__(self):
        from tracer.exchange import _convert_graph_to_dict

        state = _convert_graph_to_dict(self, is_subgraph=isinstance(self, SubGraph))
        return state

    def __setstate__(self, state):
        from tracer.exchange import _convert_dict_to_graph

        self.__init__(fn=None, empty=True)
        _convert_dict_to_graph(state, graph=self)


class SubGraph(Graph):
    """
    Subclasses Graph to override methods that require different behaviour in the context of subgraphs
    """

    def __init__(self, *args, **kwargs):

        # for keeping track of fn ids that are defined within the broader scope of the higher order fn
        self._greater_scope_ids: list = []

        # default this to false for subgraphs
        self._backend_compile: bool = False

        # a list of ids required by the subgraph which are not provided through the args
        self._subgraph_required_ids: list = []

        # store the source_gen id_name_db here so the main graph can avoid variable name duplication
        self._id_name_db: dict = {}

        super().__init__(*args, **kwargs)

    def graph_call(self, *args, **kwargs):
        """Override
        When the subgraph has already been traced, we call the native function for the graph call,
        as the generated source code of the subgraph may not work independently of the main graph.
        """
        return self._fn(*args, **kwargs)

    def reload_sourcecode(self, frontend=None, mode="eval"):
        """Override
        Only generate the source and update the constants - do not create the traced_fn
        """
        source_generator = sg.SourceGenerator(self)
        fn_str = source_generator.generate_source(graph=self, frontend=frontend)
        self._set_fn_str(fn_str)
        constants = source_generator.get_constants()
        self.constants.update(constants)


class LazyGraph:
    def __init__(self, obj, initializer, *args, **kwargs):
        self._eager_graph = None
        self._initial_obj = obj
        self._initial_args = args
        self._initializer = initializer
        self._initial_kwargs = kwargs
        self._initialized = False
        self._to_ignore = tvp.get_types_to_ignore()

        if isinstance(obj, FunctionType):
            self.__name__ = obj.__name__
        elif isinstance(obj, object):
            self.__name__ = type(obj).__name__

    def _initialize(self, *args, **kwargs):
        if not self._initialized:
            if "source" in self._initial_kwargs:
                ivy.set_backend(self._initial_kwargs["source"])
            self._to_ignore = tvp.get_types_to_ignore()
            self._initial_args = nest_array_to_new_backend(
                self._initial_args, to_ignore=self._to_ignore
            )
            self._initial_kwargs = nest_array_to_new_backend(
                self._initial_kwargs, to_ignore=self._to_ignore
            )
            self._eager_graph = self._initializer(
                self._initial_obj,
                *self._initial_args,
                args=args,
                kwargs=kwargs,
                **self._initial_kwargs,
            )
            self._initialized = True
            if "source" in self._initial_kwargs:
                ivy.previous_backend()

    def _check_if_initialized(self):
        if not self._initialized:
            raise ValueError(
                "A LazyGraph instance must be initialized before calling a Graph method."
            )

    def __call__(self, *args, **kwargs):
        if not self._initialized:
            self._initialize(*args, **kwargs)
            to = self._initial_kwargs["to"] if "to" in self._initial_kwargs else None
            if to not in [None, "ivy"]:
                ivy.set_backend(to)
            args = nest_array_to_new_backend(args, to_ignore=self._to_ignore)
            kwargs = nest_array_to_new_backend(kwargs, to_ignore=self._to_ignore)
            if to not in [None, "ivy"]:
                ivy.previous_backend()
        return self._eager_graph(*args, **kwargs)

    def __repr__(self):
        not_init_str = 'not ' if not self._initialized else ''
        return f"{object.__repr__(self)} ({not_init_str}initialized)"

    def list_missing_frontends(self, *args, **kwargs):
        self._check_if_initialized()
        return self._eager_graph.list_missing_frontends(*args, **kwargs)

    list_missing_frontends.__doc__ = Graph.list_missing_frontends.__doc__

    def list_function_frequencies(self, *args, **kwargs):
        self._check_if_initialized()
        return self._eager_graph.list_function_frequencies(*args, **kwargs)

    list_function_frequencies.__doc__ = Graph.list_function_frequencies.__doc__

    def show(self, *args, **kwargs):
        self._check_if_initialized()
        return self._eager_graph.show(*args, **kwargs)

    show.__doc__ = Graph.show.__doc__


class TracedGraph:

    def __init__(
        self,
        fn_str,
        constants,
        to_ivy: bool = False,
        with_numpy: bool = False,
        backed_during_tracing: str = "",
    ):
        self.__fn_str = fn_str
        self.constants = constants
        self.backend = "ivy" if to_ivy else backed_during_tracing
        self._to_ivy = to_ivy
        self._with_numpy = with_numpy
        self._array_mode = True
        self._container_mode = True
        self._backend_compile = False
        self._to_ignore = tvp.get_types_to_ignore()

        if os.getenv("IVY_DEBUG_SOURCE", "False").lower() == "true":
            self._scripted_call = sg.load_fn_from_file(self.__fn_str)
        else:
            self._scripted_call = sg.load_fn_from_str(self.__fn_str)

    def __call__(self, *args, **kwargs):
        if self._to_ivy:
            args = nest_array_to_new_backend(
                args, to_ignore=self._to_ignore, native=False
            )
            kwargs = nest_array_to_new_backend(
                kwargs, to_ignore=self._to_ignore, native=False
            )
            self.constants = nest_array_to_new_backend(
                self.constants, to_ignore=self._to_ignore, native=False
            )
        elif self._array_mode:
            args = ivy.nested_map(lambda x: _to_native(x, inplace=True), args)
            kwargs = ivy.nested_map(lambda x: _to_native(x, inplace=True), kwargs)
        if self._backend_compile and self._container_mode:
            args = ivy.nested_map(
                lambda x: x.cont_to_dict() if isinstance(x, ivy.Container) else x,
                args,
            )
            kwargs = ivy.nested_map(
                lambda x: x.cont_to_dict() if isinstance(x, ivy.Container) else x,
                kwargs,
            )
            self.constants = ivy.nested_map(
                lambda x: x.cont_to_dict() if isinstance(x, ivy.Container) else x,
                self.constants,
            )
        return self._scripted_call(*args, **kwargs, **self.constants)
    
    def show(
        self,
        save_to_disk=False,
        notebook=False,
        with_edge_labels=True,
        with_arg_labels=True,
        with_output_labels=True,
        output_connected_only=True,
        randomness_factor=0.1,
        highlight_subgraph=None,
        fname=None,
    ):
        pass
