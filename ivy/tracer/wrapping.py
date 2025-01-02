# global
from typing import Callable, Iterable, Optional, List
from types import ModuleType
import os
import time
import inspect
import importlib
import numpy as np
from types import FunctionType, BuiltinFunctionType
import functools

# local
import ivy
from .conversion import (
    nest_native_array_to_new_frontend,
    to_custom_numpy_type,
    nest_native_array_to_new_frontend,
    to_native,
    remove_batch_dim,
    native_array_to_frontend,
    _batched_tracer_to_array,
    _convert_to_ivy_dtype,
    _custom_to_numpy,
    _to_ivy_device,
    track,
    untrack,
)
from . import globals as glob
from .graph import Graph, LazyGraph
from . import tracked_var_proxy as tvp
from .numpy_proxy import custom_np_classes, custom_np_class_names
from .param import (
    store_unique_id,
    _get_unique_id,
    delete_parameter,
    record_parameters_info,
)
from .helpers import (
    MethodDescriptor,
    _convert_callbacks_to_lazy_graphs,
    _is_untracked_enum,
    _wraps,
    copy_dict,
)
from .visualisation import _get_argument_reprs, _get_output_reprs
from .special_ops.vmap_helpers import (
    process_vmap_fn,
    process_scalar_fn,
    process_vectorized_fn,
    add_incoming_subgraph_fns,
    add_subgraph_fns_to_dict,
)


class Node:
    """Represents a node in the graph i.e. some native/ivy function."""

    def __repr__(self):
        return (
            f"<Node `{self.__name__}` with id {self.id_}>"
            if hasattr(self, "id_")
            else super().__repr__()
        )

    @property
    def is_constant(self):
        """Returns True if the node has no tracked arguments, and
        it isn't a generator to be included in the graph."""
        has_no_tracked_args = len(self.arg_param_ids + self.kwarg_param_ids) == 0
        _is_constant = has_no_tracked_args and not self.is_generator_to_include
        # we shouldn't consider arguments as constant if we inplace update them
        _is_constant = _is_constant and self.is_inplace_w_side_effects
        return _is_constant

    def is_inplace(self, fn, args, kwargs, ret, from_tracked_var):
        """Check if the function is inplace or not. We need to track such functions
        in torch and numpy since they are the only frameworks which have the behaviour
        of inplace updates sometimes affecting other tensors. For example

        >>> x = torch.tensor([[1., 2.]])
        >>> y = x[0]
        >>> y[0] = 3.
        >>> print(x)
        tensor([[3., 2.]])

        So we need to track such nodes and include them in the graph to ensure
        the same behaviour in the traced graph.
        """
        ret_id = id(ret)
        arg_ids = [id(arg) for arg in args]
        kwarg_ids = [id(kwarg) for kwarg in kwargs.values()]
        ret_is_in_args = ret_id in arg_ids + kwarg_ids
        is_inplace_node = (
            ret_is_in_args
            and not from_tracked_var
            and fn.__name__ not in ["to", "type", "as_tensor"]
        )
        # to, type, as_tensor sometimes just return their inputs- they dont edit inplace
        self.is_inplace_w_side_effects = (
            ivy.current_backend_str() in ["torch", "numpy"] and is_inplace_node
        )
        if ivy.current_backend_str() == "tensorflow" and fn.__name__ in (
            "assign",
            "assign_sub",
            "assign_add",
        ):
            self.is_inplace_w_side_effects = True

    def untrack_cached_args(self, ids: List[int]):
        "The object associated with `id_` is now a constant, so we stop tracking it"
        indices = [i for i, x in enumerate(self.arg_param_ids) if x in ids]
        for idx in reversed(indices):
            del self.arg_tracked_idxs[idx]
            del self.arg_param_ids[idx]
            del self.arg_param_types[idx]

    def untrack_cached_kwargs(self, ids: List[int]):
        indices = [i for i, x in enumerate(self.kwarg_param_ids) if x in ids]
        for idx in reversed(indices):
            del self.kwarg_tracked_idxs[idx]
            del self.kwarg_param_ids[idx]
            del self.kwarg_param_types[idx]


def _cache_constant_args(args, kwargs, node, to_ivy):
    args, kwargs = _convert_to_ivy_dtype(args, kwargs, to_ivy)
    if to_ivy:
        args = ivy.to_ivy(args, nested=True)
        kwargs = ivy.to_ivy(kwargs, nested=True)
        args = ivy.nested_map(_to_ivy_device, args)
        kwargs = ivy.nested_map(_to_ivy_device, kwargs)
    args, kwargs = _custom_to_numpy(args, kwargs)
    node.args = args
    node.kwargs = kwargs
    return args, kwargs


def _wrap_numpy_ufuncs(wrapped, original, graph):
    """NumPy ufuncs (eg np.add) aren't functions, but instances of a class.
    Hence functools.wraps won't properly handle copying over the attributes to
    the wrapped function. This function does that manually.
    Also some attributes (eg np.add.reduce) could also be in the graph, so we
    wrap these methods before copying them over.
    """
    if isinstance(original, np.ufunc):
        wrapped.nin = original.nin
        wrapped.nout = original.nout
        wrapped.nargs = original.nargs
        wrapped.ntypes = original.ntypes
        wrapped.types = original.types
        wrapped.ntypes = original.ntypes
        wrapped.signature = original.signature
        wrapped.identity = original.identity
        wrapped.reduce = _wrap_function_for_op_logging(original.reduce, graph)
        wrapped.accumulate = _wrap_function_for_op_logging(original.accumulate, graph)
        wrapped.reduceat = _wrap_function_for_op_logging(original.reduceat, graph)
        wrapped.outer = _wrap_function_for_op_logging(original.outer, graph)
        wrapped.at = _wrap_function_for_op_logging(original.at, graph)

        FUNC_TO_PATH[original.reduce] = "numpy." + original.__name__ + ".reduce"
        FUNC_TO_PATH[original.accumulate] = "numpy." + original.__name__ + ".accumulate"
        FUNC_TO_PATH[original.reduceat] = "numpy." + original.__name__ + ".reduceat"
        FUNC_TO_PATH[original.outer] = "numpy." + original.__name__ + ".outer"
        FUNC_TO_PATH[original.at] = "numpy." + original.__name__ + ".at"


def _unwrap_numpy_ufuncs(wrapped, original):
    """Since called attributes of NumPy ufuncs aren't exposed through the normal paths,
    we need to look inside the attributes of wrapped functions
    during unwrapping to find and unwrap these.
    """
    if isinstance(original, np.ufunc):
        wrapped.reduce = _unwrap_function_from_op_logging(wrapped.reduce)
        wrapped.accumulate = _unwrap_function_from_op_logging(wrapped.accumulate)
        wrapped.reduceat = _unwrap_function_from_op_logging(wrapped.reduceat)
        wrapped.outer = _unwrap_function_from_op_logging(wrapped.outer)
        wrapped.at = _unwrap_function_from_op_logging(wrapped.at)


def _ongoing_iterator_chain(node) -> bool:
    """Check if the current node is part of an ongoing iterator chain or not."""
    if node.__name__ == "__iter__":
        # In case of a __iter__ call, the key is the output iterator object itself
        key = node.output_param_ids[0]
        ongoing_iterator_chain = key in glob.iterator_chains
    else:
        # In case of a __next__ call, the key is contained in the args
        ongoing_iterator_chain = True

    return ongoing_iterator_chain


def _get_incoming_iterator_fns(node, ongoing_iterator_chain) -> List:
    """Iterator methods form a chain but the output of previous node (__next__)
    is not the input of the next node (__next__) so need to account for that.
    """
    key = node.arg_param_ids[0] if ongoing_iterator_chain else node.output_param_ids[0]
    # Fetch the last method from the current iterator chain
    return [glob.iterator_chains[key][-1]]


def _append_to_iterator_chain(node, from_ongoing_iterator_chain) -> bool:
    """Append the current node to the ongoing iterator chain if it exists, otherwise
    create a new iterator chain and append accordingly.
    """
    # In case of a __iter__ call, the key is the output iterator object itself
    # In case of a __next__ call, the key is contained in the args
    key = (
        node.output_param_ids[0]
        if node.__name__ == "__iter__"
        else node.arg_param_ids[0]
    )
    if not from_ongoing_iterator_chain:
        glob.iterator_chains[key] = list()
    glob.iterator_chains[key].append(node)


def _wrap_function_for_op_logging(
    fn: Callable,
    graph: Graph,
    limit_attributes: bool = True,
    from_tracked_var: bool = False,
    is_builtin_fn: bool = False,
    stateful_classes: Optional[Iterable] = (),
    to_ivy: bool = False,
) -> Callable:
    """Wraps all the functions of a class/module so that these functions can be
    logged/stored while doing the preliminary forward pass.

    Parameters
    ----------
    fn
        function/method to potentially wrap.
    graph
        graph instance for the function we are tracing.
    limit_attributes
        limit attributes being added to the graph.
    from_tracked_var
        flag that indicates if the method being wrapped is from the TrackedVarProxy class.
    is_builtin_fn
        flag that indicates if the function being wrapped is from the builtins module.
    stateful_classes
        stateful classes that we also want to wrap and be included in the graph.

    Returns
    -------
        the wrapped function which would be called instead of the native function during
        the initial forward pass through the function we are tracing.
    """
    stateful_classes = tuple(
        ivy.default(list(graph._all_stateful_classes) + list(stateful_classes), tuple())
    )
    is_already_wrapped = hasattr(fn, "wrapped_for_tracing")

    # do not wrap default __init__
    if fn is object.__init__:
        return fn

    # Do not wrap the function:
    # (a) if it's a special method but not in ARRAY_BUILTINS
    # (b) if it's already wrapped
    # (c) if we are wrapping TrackedVarProxy and fn is in TRACKED_VAR_NON_WRAPPED_METHODS or tvp.RAW_RET_METHODS
    if (
        (
            hasattr(fn, "__name__")
            and not from_tracked_var
            and (
                fn.__name__[0] == "_"
                and fn.__name__
                not in glob.ARRAY_BUILTINS + glob.PRIVATE_FUNCTIONS_TO_WRAP
            )
        )
        or is_already_wrapped
        or (
            from_tracked_var
            and fn.__name__ in tvp.NON_WRAPPED_METHODS + tvp.RAW_RET_METHODS
        )
    ):
        return fn

    if isinstance(fn, MethodDescriptor):
        fn = fn.method

    @_wraps(fn, from_tracked_var=from_tracked_var, is_builtin_fn=is_builtin_fn)
    def _tracing_function(*args, **kwargs):
        """
        This is the function that will be called instead of the native function e.g.
        `torch.add` when doing the operation logging forward pass. This wrapped
        function of course executes the original native function but also records a
        variety of information such as the ids of the outputs and the native function
        which produced them. This is to enable the tracer to connect functions
        together in the correct order, during the later construction of the graph.
        """
        # if logging is paused (as it is before the op logging forward pass begins or during
        # execution of some helpers etc), just execute original native function
        if glob.tracing_paused:
            return fn(*args, **kwargs)

        if isinstance(fn, functools.partial):
            return fn(*args, **kwargs)

        target_framework = "ivy" if to_ivy else ivy.current_backend_str()

        # in the middle of tracing another function
        if glob.tracing_stack and not glob.tracing_subgraph:
            # if the function is part of vmap, or we are inside the scalar function,
            # continue logging.
            if (
                hasattr(fn, "__name__")
                and fn.__name__ in ("scalar_fn", "vectorized_fn", "vmap")
                or glob.tracing_stack[-1] == "scalar_fn"
            ):
                pass
            else:
                return fn(*args, **kwargs)

        # attributes to ignore
        att_name = None
        from_class_instance = False

        if fn.__name__ in ["__getattr__", "__setattr__", "__getattribute__"]:
            att_name = args[1]
            # return if the attribute being retrieved is another built-in method
            if att_name[0:2] == "__":
                return fn(*args, **kwargs)
            # if the attribute is not recognized as one which can form part of the graph, then return
            if (
                limit_attributes
                and att_name
                not in glob.GRAPH_ATTRIBUTES[target_framework] + tvp.ATTRIBUTES
            ):
                return fn(*args, **kwargs)

            # if we're here, this means we want to track the attr setter/getter so
            # we check first to see if the object it is called on
            if (
                fn.__name__ == "__setattr__"
                and hasattr(args[0], "__module__")
                and args[0].__module__ == "tracer.helpers"
            ):
                from_class_instance = True

        # if none of the above exceptions apply, then we log the
        # function and add it to the stack to indicate this
        glob.tracing_stack.append(fn.__name__)
        glob.tracing_paused = True

        _tracked_var_backend_fn = fn

        # Store the wrapped method in case it's a valid method from TrackedVarProxy
        # The only exceptions are methods from TrackedVarProxyMeta classes since they
        # will not contain any backend var in them, only the TrackedVarProxy class instance itself
        if (
            from_tracked_var
            and fn.__name__ != "input_to_output"
            and fn.__name__[3:] not in tvp.AVAILABLE_RAW_RET
            and not (
                hasattr(fn, "__qualname__")
                and any(
                    [
                        metacls in fn.__qualname__
                        for metacls in tvp.TRACKED_VAR_PROXY_META_CLASSES
                    ]
                )
            )
        ):
            # Need to support tracking enums in the args since they are subclasses of int
            # and use the same tracked methods
            _arg = args[0]
            _arg = (
                track(
                    _arg,
                    with_numpy=graph._with_numpy,
                    stateful_classes=stateful_classes,
                )
                if _is_untracked_enum(_arg)
                else _arg
            )

            # There is no _tracked_var_backend_fn for iterator proxies
            if fn.__name__ not in tvp.ITERATOR_METHODS:
                try:
                    _arg_var = _arg.get_var()
                except AttributeError:
                    raise AttributeError(
                        f"{type(_arg).__name__} is not a valid tracked proxy"
                    )
                else:
                    try:
                        _tracked_var_backend_fn = getattr(
                            _arg_var.__class__, fn.__name__
                        )
                    except AttributeError:
                        _tracked_var_backend_fn = getattr(_arg.__class__, fn.__name__)

        is_init = fn.__name__ == "__init__"
        is_wrapped_builtin_callable = fn.__name__ == "wrapped_builtin_callable"
        is_builtin_method = is_wrapped_builtin_callable and kwargs.pop(
            "is_builtin_method", False
        )
        is_builtin_callable = is_builtin_fn
        is_ivy_fn = (
            hasattr(fn, "__module__")
            and fn.__module__ is not None
            and "ivy" in fn.__module__
        )
        is_higher_order_fn = (
            fn.__name__
            in glob.HIGHER_ORDER_FNS_TO_TRACE[
                "ivy" if is_ivy_fn else ivy.current_backend_str()
            ]
        )
        effective_fn = fn

        # If the function is __init__ or a builtin callable we're tracking,
        # then strip the first argument. For __init__, the first argument
        # is the instance. For a builtin callable, the first argument is
        # the builtin fn to call
        if is_init or is_wrapped_builtin_callable:
            effective_args = args[1:]
            f_arg = args[0]
            effective_fn = fn if is_init else f_arg
            is_builtin_callable = is_builtin_callable if is_init else True
            args = effective_args if is_builtin_callable else args

        # store information about this vmap function which will later be
        # used in reconstructing vmap
        if fn.__name__ == "vmap":
            args, kwargs = process_vmap_fn(graph, fn, args, kwargs)

        if is_higher_order_fn:
            # if this is a higher order function, convert the callback
            # arguments to lazy graphs so they can be traced into subgraphs
            subgraphs, args, kwargs = _convert_callbacks_to_lazy_graphs(
                graph, fn, args, kwargs
            )
        else:
            subgraphs = []

        # Flag to determine if the function is from the iterator classes for TVPs
        from_tracked_var_iterators = (
            any(
                [
                    itercls in fn.__qualname__
                    for itercls in tvp.tracked_var_proxy_iter_classes
                ]
            )
            if hasattr(fn, "__qualname__")
            else False
        )

        # Flag to determine if the function forms a part of an ongoing iterator
        # chain i.e. __iter__ ---> __next__ ---> fn
        from_iterator_chain = from_tracked_var_iterators or (
            not from_tracked_var_iterators
            and from_tracked_var
            and fn.__name__ in tvp.BUILTIN_ITERATOR_METHODS
        )

        _to_ignore = tvp.get_types_to_ignore()

        # args and kwargs to native arrays
        input_args, input_kwargs = args, kwargs

        # Need to separate the call to to_native since it also
        # handles frontend arrays and we need that when transpiling
        # a builtin callable on a frontend array
        if from_class_instance or is_builtin_callable:
            args = to_native(args, cont_inplace=True, to_ignore=_to_ignore)
            kwargs = to_native(kwargs, cont_inplace=True, to_ignore=_to_ignore)
        elif not to_ivy:
            args = ivy.to_native(args, True, cont_inplace=True, to_ignore=_to_ignore)
            kwargs = ivy.to_native(
                kwargs, True, cont_inplace=True, to_ignore=_to_ignore
            )

        args = ivy.nested_map(_batched_tracer_to_array, args, to_ignore=_to_ignore)
        kwargs = ivy.nested_map(_batched_tracer_to_array, kwargs, to_ignore=_to_ignore)

        if fn.__name__ == "scalar_fn" and graph._to_ivy:
            args = ivy.nested_map(native_array_to_frontend, args, to_ignore=_to_ignore)
            kwargs = ivy.nested_map(
                native_array_to_frontend, kwargs, to_ignore=_to_ignore
            )

        # check if there are slices with TrackedVars inside
        arg_tracked_slices_idxs = ivy.nested_argwhere(
            args, tvp.is_tracked_slice, to_ignore=_to_ignore
        )
        kwarg_tracked_slices_idxs = ivy.nested_argwhere(
            kwargs, tvp.is_tracked_slice, to_ignore=_to_ignore
        )
        # convert slices to slice-lists
        args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.slice_to_list)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_tracked_slices_idxs, tvp.slice_to_list
        )

        node = Node()
        (
            node.arg_tracked_idxs,
            arg_parameters,
            node.arg_param_ids,
            node.arg_param_types,
            _,
            _,
        ) = record_parameters_info(
            args,
            to_ivy,
            graph._with_numpy,
            stateful_classes,
        )

        (
            node.kwarg_tracked_idxs,
            _,
            node.kwarg_param_ids,
            node.kwarg_param_types,
            _,
            _,
        ) = record_parameters_info(
            kwargs,
            to_ivy,
            graph._with_numpy,
            stateful_classes,
        )

        if is_builtin_callable:
            # early return for sum etc if no input params: alleviates overhead of
            # tracing many irrelevant builtin calls
            if not node.arg_param_ids + node.kwarg_param_ids:
                glob.tracing_paused = False
                glob.tracing_stack.pop()
                return effective_fn(*args, **kwargs)
            # otherwise, flag that the graph contains builtin callables
            graph._contains_builtin_callables = True

        # set the backend function
        backend_fn = effective_fn

        if from_tracked_var:
            # Store the TrackedVarProxy to update its var instead of creating a new one
            tracked_var_instance = args[0]
            # If the function comes a TrackedVarProxy instance, the function we should store is the
            # corresponding function from the wrapped var. This way, once the function is traced,
            # the methods from the original variables are called
            backend_fn = _tracked_var_backend_fn

        # set the backend instance that the builtin method is bound to
        backend_self = backend_fn.__self__ if is_builtin_method else None
        backend_self_repr = repr(backend_self) if backend_self is not None else ""

        arg_types = [a.__class__.__name__ for a in args]

        # Need to untrack the args/kwargs here except when the function is one of the
        # iterator protocols for tracked proxies since these iterator protocols
        # take tracked proxies as inputs not the original backend vars
        untrack_post_ret = False
        if not (
            from_tracked_var_iterators
            or (
                not from_tracked_var_iterators
                and from_tracked_var
                and fn.__name__ in tvp.DICT_ITERATOR_METHODS
            )
            or any(  # do not untrack the args of any fns which take an iterator proxy
                [itercls in arg_types for itercls in tvp.tracked_var_proxy_iter_classes]
            )
        ):
            args = untrack(args)
            kwargs = untrack(kwargs)
        else:
            untrack_post_ret = True

        # convert slice-lists to slices
        args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.list_to_slice)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_tracked_slices_idxs, tvp.list_to_slice
        )

        # call the original native function. We pause logging here since we don't
        # want to add any functions called inside `backend_fn` to the graph as well
        if backend_fn.__name__ == "scalar_fn":
            # store the current fns we have traced
            graph._tmp_subgraph_id_to_function.insert(
                -1, graph._subgraph_id_to_function
            )

            # reset/initialize the subgraph data structures
            graph._subgraph_id_to_function = dict()

        glob.tracing_paused = backend_fn.__name__ not in ("scalar_fn", "vectorized_fn")

        try:
            if fn.__name__ == "vectorized_fn":
                # get the scalar fn
                scalar_fn = graph.scalar_fn
                effective_args, effective_kwargs = remove_batch_dim(args, kwargs)
                # trace into the scalar fn with logging paused
                scalar_fn(*effective_args, **effective_kwargs)
            else:
                ret = backend_fn(*args, **kwargs)

            if backend_fn.__name__ == "vmap":
                # change the name to track in the logging stack
                setattr(ret, "__name__", "vectorized_fn")

                # delete the wrapped_for_tracing attribute if exists
                if hasattr(ret, "wrapped_for_tracing"):
                    delattr(ret, "wrapped_for_tracing")

                # wrap the function to enable logging
                ret = _wrap_function_for_op_logging(ret, graph, to_ivy=graph._to_ivy)

                # store the scalar fn
                graph.scalar_fn = args[0]

            if backend_fn.__name__ == "vectorized_fn":
                # re-compute the output as tracing gave incorrect results
                glob.tracing_paused = True
                ret = backend_fn(*args, **kwargs)
        except Exception as e:
            glob.tracing_paused = False
            if glob.tracing_stack:
                glob.tracing_stack.pop()
            raise e
        glob.tracing_paused = True

        node.is_inplace(fn, args, kwargs, ret, from_tracked_var)

        # Need to untrack here for the dependent params to be correctly deleted
        # in case the function was one of the iterator protocols of the tracked proxies,
        # because we didn't untrack the args/kwargs above.
        args = untrack(args) if untrack_post_ret else args
        kwargs = untrack(kwargs) if untrack_post_ret else kwargs

        # convert slices to slices-lists
        args = ivy.map_nest_at_indices(args, arg_tracked_slices_idxs, tvp.slice_to_list)
        kwargs = ivy.map_nest_at_indices(
            kwargs, kwarg_tracked_slices_idxs, tvp.slice_to_list
        )

        # find ids for dependent paramaters in the args/kwargs so that we don't track ret
        # if the fn had no dependent parameters in the args/kwargs
        input_parameter_ids = node.arg_param_ids + node.kwarg_param_ids
        if glob.tracing_subgraph:
            with_dependent_parameters = any(
                [x in glob.subgraph_dependent_ids for x in input_parameter_ids]
            )
        else:
            with_dependent_parameters = any(
                [
                    x in glob.dependent_ids[glob.current_trace_mode]
                    for x in input_parameter_ids
                ]
            )

        # determine if the function is a numpy function
        fn_is_numpy = False
        if graph._with_numpy:
            if hasattr(fn, "__qualname__"):
                fn_is_numpy = "ndarray" in fn.__qualname__

            if not fn_is_numpy:
                # check for method
                if hasattr(fn, "__self__") and fn.__self__ is not None:
                    fn_is_numpy = "numpy" in str(fn.__self__.__class__)
                # check for function
                elif hasattr(fn, "__module__") and fn.__module__ is not None:
                    fn_is_numpy = (
                        "numpy" in fn.__module__ and "jax" not in fn.__module__
                    )

        # added so tensorflow inplace variable updates work properly (return is set
        # to first arg since this is the variable updated inplace)
        # provide return value for __setattr__ and similar functions
        inplace_fn = False
        if (
            (
                not from_tracked_var
                and fn.__name__
                in ["__setattr__"]
                + glob.INPLACE_METHODS_WITHOUT_RET[target_framework]
                + glob.INPLACE_FUNCTIONS_WITHOUT_RET[target_framework]
            )
            or (
                fn_is_numpy
                and fn.__name__
                in glob.INPLACE_METHODS_WITHOUT_RET["numpy"]
                + glob.INPLACE_FUNCTIONS_WITHOUT_RET["numpy"]
            )
            or (from_tracked_var and fn.__name__ in tvp.INPLACE_METHODS_WITHOUT_RET)
        ) and (ret is None or id(ret) == id(args[0])):
            ret = args[0]
            inplace_fn = True

        if fn.__name__ in glob.TENSOR_LIST_FNS_TO_TRACK[target_framework]:
            # track an output list of tensors
            ret = tvp.TrackedListProxy(ret)

        # track output if the fn is a TrackedVarProxy method or if ret is an instance of a class
        # that should always be tracked, however don't track ret if the inputs had no dependent params
        if with_dependent_parameters and (
            from_tracked_var
            or tvp.should_be_tracked(fn, att_name, ret, target_framework)
        ):
            if fn.__name__ not in tvp.RAW_RET_METHODS:
                if fn.__name__ in tvp.INPLACE_METHODS_WITHOUT_RET:
                    ret = tracked_var_instance
                else:
                    ret = track(
                        ret,
                        with_numpy=graph._with_numpy,
                        stateful_classes=stateful_classes,
                        _deepcopy=False,
                    )

        # prevent side effects from modifying dicts
        args = list(args)
        for i in range(len(args)):
            args[i] = copy_dict(args[i])
        args = tuple(args)

        for k in kwargs.keys():
            kwargs[k] = copy_dict(kwargs[k])

        # remove parameters from args and kwargs
        args = ivy.map_nest_at_indices(
            args,
            node.arg_tracked_idxs,
            lambda x: delete_parameter(x, graph),
        )
        kwargs = ivy.map_nest_at_indices(
            kwargs,
            node.kwarg_tracked_idxs,
            lambda x: delete_parameter(x, graph),
        )

        ret_list = [ret]

        # Since we don't have frontends for builtin fns
        # like min/max/sum etc, so need to handle that
        # conversion logic that is typically handled by decorators
        # in the frontends
        transpiling_builtin_fn = False
        transpiling_instance_method = False
        if not to_ivy and (is_builtin_fn or transpiling_instance_method):
            input_nest = input_args if input_args else input_kwargs
            (ret_list, temp_bool) = nest_native_array_to_new_frontend(
                ret_list, input_nest
            )
            transpiling_builtin_fn = temp_bool
            transpiling_instance_method = temp_bool

        new_ret = to_custom_numpy_type(ret_list, with_numpy=graph._with_numpy)

        # Python doesn't allow __init__ to return anything other than None
        new_ret = [None] if is_init else new_ret

        if fn.__name__ == "__getattribute__":
            ret_list = ivy.nested_map(
                lambda x: x.ivy_array if hasattr(x, "ivy_array") else x,
                ret_list,
                shallow=False,
            )
            ret_list = ivy.to_native(ret_list, nested=True)

        if to_ivy:
            ret_list = ivy.to_native(ret_list, nested=True, to_ignore=_to_ignore)

        (
            node.output_tracked_idxs,
            output_parameters,
            node.output_param_ids,
            node.output_param_types,
            node.output_param_var_flags,
            node.output_param_shapes,
        ) = record_parameters_info(
            (
                to_native(ret_list)
                if (transpiling_builtin_fn or transpiling_instance_method)
                else ret_list
            ),
            to_ivy,
            graph._with_numpy,
            stateful_classes,
        )

        # return if there are no tracked outputs
        if not node.output_tracked_idxs:
            if glob.tracing_stack:
                glob.tracing_stack.pop()
            glob.tracing_paused = False
            return new_ret[0]

        # store a unique id for the attribute, as future accesses of the same attribute
        # will have the same id, and we need to preserve uniqueness in the graph.
        if fn.__name__ in ["__getattr__", "__getattribute__"]:
            [store_unique_id(x, graph) for x in output_parameters]

        # find all those outputs which have the same id as one of
        # the inputs and find a new unique id for them
        duplicates = list()
        for i, ret_id in enumerate(node.output_param_ids):
            if ret_id in input_parameter_ids:
                duplicates.append(i)
        duplicate_tracked_idxs = [node.output_tracked_idxs[i] for i in duplicates]
        duplicate_params = ivy.multi_index_nest(ret_list, duplicate_tracked_idxs)
        [store_unique_id(x, graph) for x in duplicate_params]
        # update output param ids after obtaining new ids
        node.output_param_ids = [
            (
                _get_unique_id(to_native(x))
                if transpiling_instance_method
                else _get_unique_id(x)
            )
            for x in output_parameters
        ]

        gen_fns = glob.GENERATOR_FUNCTIONS[target_framework]
        if graph._with_numpy and target_framework != "numpy":
            gen_fns = gen_fns + glob.GENERATOR_FUNCTIONS["numpy"]

        node.is_generator = fn.__name__ in gen_fns
        node.is_generator_to_include = node.is_generator and graph._include_generators
        if node.is_generator_to_include or with_dependent_parameters:
            if glob.tracing_subgraph:
                [glob.subgraph_dependent_ids.add(id_) for id_ in node.output_param_ids]
            else:
                [
                    glob.dependent_ids[glob.current_trace_mode].add(id_)
                    for id_ in node.output_param_ids
                ]

        args, kwargs = _cache_constant_args(args, kwargs, node, to_ivy)

        # store info about this node
        node.backend_fn = backend_fn
        if backend_fn in FUNC_TO_PATH and not from_tracked_var:
            node.path = FUNC_TO_PATH[backend_fn]

        node.is_builtin_callable = is_builtin_callable
        node.from_tracked_var = from_tracked_var
        node.from_tracked_var_iterators = from_tracked_var_iterators
        node.from_iterator_chain = from_iterator_chain
        node.mode = glob.current_trace_mode
        node.is_higher_order = is_higher_order_fn
        node.is_ivy = False

        try:
            sig = inspect.signature(fn)
            sig_keys = list(sig.parameters.keys())
        except ValueError:
            sig_keys = list()
        node.arg_n_kwarg_reprs = _get_argument_reprs(sig_keys, args, kwargs)
        ret_placeholder = ivy.set_nest_at_indices(
            ret_list,
            node.output_tracked_idxs,
            "tracked",
            shallow=False,
        )
        node.output = ret_placeholder
        node.output_reprs = _get_output_reprs(ret_list)

        node.timestamp = time.perf_counter()
        node.terminal = True
        node.inplace_fn = inplace_fn
        node.with_tracked_slices = arg_tracked_slices_idxs + kwarg_tracked_slices_idxs

        # assign the same name to `node` as it is in the backend
        node.__repr__ = lambda: node.__name__
        if fn.__name__ == "vectorized_fn":
            node.__name__ = "vmap"
        elif fn.__name__ == "rep_method":
            # extract the name of the wrapped dunder method by inspecting its closure
            try:
                node.__name__ = fn.__closure__[0].cell_contents.__name__
            except:
                node.__name__ = backend_fn.__name__
        elif is_builtin_method:
            node.__name__ = f"{backend_self_repr}.{backend_fn.__name__}"
        else:
            node.__name__ = backend_fn.__name__

        # Add the current fn to the iterator chain to correctly chain consecutive
        # __next__ nodes in the graph
        from_ongoing_iterator_chain = (
            _ongoing_iterator_chain(node) if from_iterator_chain else False
        )

        if backend_fn.__name__ == "scalar_fn":
            ret = new_ret[0]
            ret = ret if isinstance(ret, tuple) else (ret,)
            process_scalar_fn(graph, backend_fn, arg_parameters, ret)

        if backend_fn.__name__ == "vectorized_fn":
            node = process_vectorized_fn(graph, node)

        prev_fn = False
        if len(glob.tracing_stack) > 1 and glob.tracing_stack[-2] == "scalar_fn":
            fns_in = add_incoming_subgraph_fns(
                graph,
                fn,
                input_parameter_ids,
            )
        elif from_iterator_chain and from_ongoing_iterator_chain:
            fns_in = _get_incoming_iterator_fns(node, from_ongoing_iterator_chain)
            prev_fn = True
        elif glob.tracing_subgraph:
            fns_in = [
                glob.subgraph_id_to_fn[id_]
                for id_ in input_parameter_ids
                if id_ in glob.subgraph_id_to_fn
            ]
        else:
            fns_in = [
                graph._id_to_function[glob.current_trace_mode][id_]
                for id_ in input_parameter_ids
                if id_ in graph._id_to_function[glob.current_trace_mode]
            ]

        # add this function as the outgoing function of the incoming functions
        if node.output_param_ids:
            for fn_in in fns_in:
                fn_in.terminal = False
                if node not in fn_in.fns_out:
                    fn_in.fns_out.append(node)

        node.fns_in = fns_in
        node.fns_out = list()
        node.id_ = id(node)
        if eval(os.getenv("CHECK_TRANSPILER_OVERHEAD", "False")):
            node.from_ = glob.current_frontend

        for sub in subgraphs:
            if node.id_ not in graph._id_to_subgraphs[glob.current_trace_mode].keys():
                graph._id_to_subgraphs[glob.current_trace_mode][node.id_] = []
            graph._id_to_subgraphs[glob.current_trace_mode][node.id_].append(sub)

        # For iterator chains s.t.  __iter__ ---> __next__ ---> __next__
        node.prev_fn = fns_in[0] if prev_fn else None

        # add this function to the graph for each output id
        if fn.__name__ not in (
            "safe_map",
            "scalar_fn",
        ):  # do not add these functions to the main graph
            if len(glob.tracing_stack) > 1 and glob.tracing_stack[-2] == "scalar_fn":
                add_subgraph_fns_to_dict(graph, fn, node, node.output_param_ids)
            else:
                for id_ in node.output_param_ids:
                    if glob.tracing_subgraph and id_ not in glob.subgraph_id_to_fn:
                        glob.subgraph_id_to_fn[id_] = node
                    elif id_ not in graph._id_to_function[glob.current_trace_mode]:
                        # use most recent fn with id, so higher order function will be
                        # used rather than anything within the callback
                        graph.add_fn_to_dict(id_, node)

        # Add the current fn to the iterator chain to correctly chain consecutive
        # __next__ nodes in the graph
        if from_iterator_chain:
            _append_to_iterator_chain(node, from_ongoing_iterator_chain)

        # remove function from stack, now logging has occurred
        if glob.tracing_stack:
            glob.tracing_stack.pop()
        glob.tracing_paused = False

        # return the function output
        return new_ret[0]

    _tracing_function.wrapped_for_tracing = id(graph)
    _wrap_numpy_ufuncs(_tracing_function, fn, graph)
    return _tracing_function


def _unwrap_function_from_op_logging(function_wrapped):
    if hasattr(function_wrapped, "wrapped_for_tracing"):
        _unwrap_numpy_ufuncs(function_wrapped, function_wrapped.__wrapped__)
        return function_wrapped.__wrapped__
    return function_wrapped


def _should_be_wrapped(obj):
    return (
        callable(obj)
        and not inspect.isclass(obj)
        and not (hasattr(obj, "__module__") and obj.__module__ == "typing")
    )


FUNC_TO_PATH = {}


def _wrap_or_unwrap_module(
    wrap_or_unwrap_fn,
    module,
    framework=None,
    to_ivy=False,
):
    framework = ivy.current_backend_str() if framework is None else framework
    framework = "ivy" if to_ivy else framework
    module_name = (
        module.__name__
        if module.__name__ in glob.BUILTIN_MODULES_TO_TRACK or not to_ivy
        else "ivy"
    )
    for k in dir(module):
        v = getattr(module, k)
        if (
            k in glob.FUNCTIONS_ATTRS_NOT_TO_WRAP[framework]
            or k[0] == "_"
            or not _should_be_wrapped(v)
        ):
            continue
        try:
            setattr(module, k, wrap_or_unwrap_fn(v))
            if not hasattr(v, "wrapped_for_tracing"):
                FUNC_TO_PATH[v] = module_name + "." + k
        except Exception:
            pass


def _wrap_or_unwrap_class(
    wrap_or_unwrap_fn, cls, cls_path=None, framework=None, to_ivy=False
):
    if cls is None:
        return
    framework = ivy.current_backend_str() if framework is None else framework
    framework = "ivy" if to_ivy else framework
    for k in dir(cls):
        attr = getattr(cls, k)
        if k in glob.FUNCTIONS_ATTRS_NOT_TO_WRAP[framework] or not _should_be_wrapped(
            attr
        ):
            continue

        # TODO: this is a temporary fix to avoid tracing ResourceVariable.__init__ which isn't working properly
        if k == "__init__" and cls.__name__ in ["Variable", "ResourceVariable"]:
            continue

        try:
            if hasattr(getattr(cls, k), "__name__"):
                if getattr(cls, k).__name__ != "":
                    setattr(cls, k, wrap_or_unwrap_fn(attr))
        except Exception as e:
            pass
        if cls_path is not None:
            if cls_path == "NewNDArray":
                FUNC_TO_PATH[attr] = "numpy.ndarray." + k
            elif cls_path in custom_np_class_names:
                FUNC_TO_PATH[attr] = k
            else:
                FUNC_TO_PATH[attr] = ".".join(cls_path) + "." + k


def _wrap_or_unwrap_intenum(wrap_or_unwrap_fn, int_enum_proxy):
    val_source = tvp.ATTRS_TO_WRAP_AND_MAP[int_enum_proxy.__name__]["mapping"]
    to_map = tvp.ATTRS_TO_WRAP_AND_MAP[int_enum_proxy.__name__].get("to_map")
    to_ignore = tvp.ATTRS_TO_WRAP_AND_MAP[int_enum_proxy.__name__].get("to_ignore")
    dir_source = [_m for _m in dir(val_source) if _m not in to_ignore]
    dir_source = (
        [_m for _m in dir(val_source) if _m not in to_ignore]
        if "*" in to_map
        else dir_source
    )
    dir_source = (
        [_m for _m in dir(val_source) if _m in to_map]
        if "*" in to_ignore
        else dir_source
    )
    for k in dir_source:
        v = getattr(val_source, k)
        if not _should_be_wrapped(v):
            continue
        try:
            setattr(int_enum_proxy, k, wrap_or_unwrap_fn(v))
        except Exception:
            pass


def _load_classes_from(ctw: List):
    classes = []
    for _ctw in ctw:
        try:
            classes.append(getattr(importlib.import_module(_ctw[0]), _ctw[1]))
        except AttributeError:
            classes.append(None)
    return classes


def _load_modules_from(mtw: List):
    modules = []
    for _mtw in mtw:
        try:
            modules.append(importlib.import_module(_mtw))
        except:
            pass
    return modules


def _wrap_functions_for_op_logging(
    graph,
    stateful_classes=None,
    to_ivy=False,
    with_numpy=False,
    trace_builtins=False,
):
    glob.wrapped_fns = {}
    target = "ivy" if to_ivy else ivy.current_backend_str()
    private_class_paths = glob.PRIVATE_CLASSES_TO_WRAP(target)
    private_classes = _load_classes_from(private_class_paths)
    for cls, path in zip(private_classes, private_class_paths):
        _wrap_or_unwrap_class(
            lambda fn: _wrap_function_for_op_logging(
                fn, graph, stateful_classes=private_classes, to_ivy=to_ivy
            ),
            cls,
            path,
            to_ivy=to_ivy,
        )
    class_paths = glob.CLASSES_TO_WRAP[target]
    classes = _load_classes_from(class_paths)
    for cls, path in zip(classes, class_paths):
        _wrap_or_unwrap_class(
            lambda fn: _wrap_function_for_op_logging(
                fn, graph, stateful_classes=private_classes, to_ivy=to_ivy
            ),
            cls,
            path,
            to_ivy=to_ivy,
        )
    if target == "tensorflow":
        import tensorflow as tf

        # these tf modules can't be imported from a string, so adding them manually
        modules_to_wrap = [
            tf.compat.v2.compat.v1.nn,
            tf.compat.v2.compat.v1.linalg,
            tf.compat.v2.compat.v1.math,
        ]
    elif target == "ivy":
        modules_to_wrap = [ivy.linalg]
    else:
        modules_to_wrap = []
    modules_to_wrap += _load_modules_from(glob.MODULES_TO_WRAP[target])
    for module in modules_to_wrap:
        _wrap_or_unwrap_module(
            lambda fn: _wrap_function_for_op_logging(
                fn, graph, stateful_classes=private_classes, to_ivy=to_ivy
            ),
            module,
            to_ivy=to_ivy,
        )

    # wrap numpy after wrapping modules of current backend. wrapping before causes
    # issues with modules like jax.scipy.optimise where they import like
    # `from numpy import asarray` which would then import the wrapped version of
    # numpy.asarray, and would not be unwrapped afterwards. this is only a problem
    # with modules in jax.scipy because they are not initialised upon `import jax`,
    # and so will be initialised when we import them to wrap.
    if with_numpy:
        for custom_class in custom_np_classes:
            _wrap_or_unwrap_class(
                lambda fn: _wrap_function_for_op_logging(fn, graph),
                custom_class,
                custom_class.__name__,
                framework="numpy",
                to_ivy=to_ivy,
            )
        for module in _load_modules_from(glob.MODULES_TO_WRAP["numpy"]):
            _wrap_or_unwrap_module(
                lambda fn: _wrap_function_for_op_logging(fn, graph),
                module,
                framework="numpy",
                to_ivy=to_ivy,
            )

    # wrap TrackedVarProxy classes
    for proxy_class in tvp.proxy_classes():
        if proxy_class.__name__ in tvp.ATTRS_TO_WRAP_AND_MAP:
            _wrap_or_unwrap_intenum(
                lambda fn: _wrap_function_for_op_logging(
                    fn, graph, from_tracked_var=True
                ),
                proxy_class,
            )
        _wrap_or_unwrap_class(
            lambda fn: _wrap_function_for_op_logging(fn, graph, from_tracked_var=True),
            proxy_class,
        )

    if graph._transpiling and ivy.current_backend_str() in ["torch"]:
        # wrap ivy.while_loop so it can be converted to a pythonic while loop in source_gen
        ivy.while_loop = _wrap_function_for_op_logging(
            ivy.while_loop,
            graph,
        )

    # wrap functorch.vmap
    if target == "torch":
        try:
            import functorch

            functorch.vmap = _wrap_function_for_op_logging(
                functorch.vmap,
                graph,
            )
        except:
            # do not wrap functorch.vmap if it is not installed,
            # which can occur when using torch versions < 1.13.0
            pass

    # wrap any native functions in the arguments
    graph._args = ivy.nested_map(
        lambda x: (
            _wrap_function_for_op_logging(x, graph)
            if isinstance(x, (FunctionType, BuiltinFunctionType)) and x in FUNC_TO_PATH
            else x
        ),
        graph._args,
    )
    graph._kwargs = ivy.nested_map(
        lambda x: (
            _wrap_function_for_op_logging(x, graph)
            if isinstance(x, (FunctionType, BuiltinFunctionType)) and x in FUNC_TO_PATH
            else x
        ),
        graph._kwargs,
    )

    # wrap builtins
    if trace_builtins:
        # i. for wrapping type castings and atomic fns like len.
        from tracer import helpers

        helpers.wrapped_builtin_callable = _wrap_function_for_op_logging(
            helpers.wrapped_builtin_callable,
            graph,
        )
        # ii. for wrapping other fns like min, max, sum etc.
        for fn in glob.BUILTIN_FNS_TO_TRACK:
            glob__builtins__ = globals()["__builtins__"]
            if isinstance(glob__builtins__, ModuleType):
                glob__builtins__ = glob__builtins__.__dict__
            glob__builtins__[fn] = _wrap_function_for_op_logging(
                glob__builtins__[fn],
                graph,
                is_builtin_fn=True,
            )

        # iii. for wrapping std libs such as math, itertools etc.
        builtin_modules_paths = glob.BUILTIN_MODULES_TO_TRACK
        builtin_modules_to_wrap = _load_modules_from(builtin_modules_paths)
        for module in builtin_modules_to_wrap:
            _wrap_or_unwrap_module(
                lambda fn: _wrap_function_for_op_logging(
                    fn,
                    graph,
                    stateful_classes=private_classes,
                    to_ivy=to_ivy,
                    is_builtin_fn=True,
                ),
                module,
                to_ivy=to_ivy,
            )

    # wrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, "__setattr__") and (
            hasattr(cls, "__getattr__") or hasattr(cls, "__getattribute__")
        )
        assert hasattr(cls, "__init__")
        cls.__init__ = _wrap_function_for_op_logging(
            cls.__init__,
            graph,
            limit_attributes=False,
            stateful_classes=stateful_classes,
        )
        cls.__setattr__ = _wrap_function_for_op_logging(
            cls.__setattr__,
            graph,
            limit_attributes=False,
            stateful_classes=stateful_classes,
        )
        if hasattr(cls, "__getattr__"):
            cls.__getattr__ = _wrap_function_for_op_logging(
                cls.__getattr__,
                graph,
                limit_attributes=False,
                stateful_classes=stateful_classes,
            )
        if hasattr(cls, "__getattribute__"):
            cls.__getattribute__ = _wrap_function_for_op_logging(
                cls.__getattribute__,
                graph,
                limit_attributes=False,
                stateful_classes=stateful_classes,
            )

    # wrap object base methods
    _wrap_function_for_op_logging(
        object.__setattr__,
        graph,
        limit_attributes=False,
        stateful_classes=stateful_classes,
    )


def _unwrap_functions_from_op_logging(
    stateful_classes=None,
    to_ivy=False,
    with_numpy=False,
    trace_builtins=False,
):
    wrapped_dict = glob.wrapped_fns
    glob.wrapped_fns = {}
    for _, v in wrapped_dict.items():
        if hasattr(v[1], "wrapped_for_tracing"):
            glob.wrapped_fns[id(v[1])] = (v[1], v[0])
    wrapped_dict = {}

    target = "ivy" if to_ivy else ivy.current_backend_str()
    if with_numpy:
        for custom_class in custom_np_classes:
            _wrap_or_unwrap_class(
                _unwrap_function_from_op_logging,
                custom_class,
                framework="numpy",
            )
        for module in _load_modules_from(glob.MODULES_TO_WRAP["numpy"]):
            _wrap_or_unwrap_module(
                _unwrap_function_from_op_logging,
                module,
                framework="numpy",
            )

    # unwrap proxy classes
    for proxy_class in tvp.proxy_classes():
        if proxy_class.__name__ in tvp.ATTRS_TO_WRAP_AND_MAP:
            _wrap_or_unwrap_intenum(
                _unwrap_function_from_op_logging,
                proxy_class,
            )
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            proxy_class,
        )
    modules_to_unwrap = _load_modules_from(glob.MODULES_TO_WRAP[target])
    if target == "tensorflow":
        import tensorflow as tf

        modules_to_unwrap += [
            tf.compat.v2.compat.v1.nn,
            tf.compat.v2.compat.v1.linalg,
            tf.compat.v2.compat.v1.math,
        ]
    elif target == "ivy":
        modules_to_unwrap += [ivy.linalg]
    for module in modules_to_unwrap:
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            module,
            to_ivy=to_ivy,
        )
    # unwrap backend classes
    ctu = glob.CLASSES_TO_WRAP[target]
    classes_to_unwrap = _load_classes_from(ctu) + stateful_classes
    for cls in classes_to_unwrap:
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            cls,
            to_ivy=to_ivy,
        )

    # unwrap private classes
    pctw = glob.PRIVATE_CLASSES_TO_WRAP(target)[::-1]
    priv_classes_to_wrap = _load_classes_from(pctw)
    for pctw in priv_classes_to_wrap:
        _wrap_or_unwrap_class(
            _unwrap_function_from_op_logging,
            pctw,
            to_ivy=to_ivy,
        )

    # unwrap ivy.while_loop
    ivy.while_loop = _unwrap_function_from_op_logging(
        ivy.while_loop,
    )

    # unwrap functorch.vmap
    if target == "torch":
        try:
            import functorch

            functorch.vmap = _unwrap_function_from_op_logging(
                functorch.vmap,
            )
        except:
            pass

    # unwrap builtins
    if trace_builtins:
        from tracer import helpers

        helpers.wrapped_builtin_callable = _unwrap_function_from_op_logging(
            helpers.wrapped_builtin_callable,
        )
        for fn in glob.BUILTIN_FNS_TO_TRACK:
            glob__builtins__ = globals()["__builtins__"]
            if isinstance(glob__builtins__, ModuleType):
                glob__builtins__ = glob__builtins__.__dict__
            glob__builtins__[fn] = _unwrap_function_from_op_logging(
                glob__builtins__[fn],
            )

        # unwrap std buildin libs (math, itertools) etc.
        builtin_modules_paths = glob.BUILTIN_MODULES_TO_TRACK
        builtin_modules_to_unwrap = _load_modules_from(builtin_modules_paths)
        for module in builtin_modules_to_unwrap:
            _wrap_or_unwrap_class(
                _unwrap_function_from_op_logging,
                module,
                to_ivy=to_ivy,
            )
    # unwrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, "__init__")
        cls.__init__ = _unwrap_function_from_op_logging(cls.__init__)
        assert hasattr(cls, "__setattr__") and (
            hasattr(cls, "__getattr__") or hasattr(cls, "__getattribute__")
        )
        cls.__setattr__ = _unwrap_function_from_op_logging(cls.__setattr__)
        if hasattr(cls, "__getattr__"):
            cls.__getattr__ = _unwrap_function_from_op_logging(cls.__getattr__)
        if hasattr(cls, "__getattribute__"):
            cls.__getattribute__ = _unwrap_function_from_op_logging(
                cls.__getattribute__
            )

    # unwrap object base methods
    _unwrap_function_from_op_logging(object.__setattr__)
