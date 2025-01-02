# global
from typing import (
    Callable,
    Optional,
    Union,
    Tuple,
    List,
    Any,
    Iterable,
    Mapping,
    Sequence,
)
from types import ModuleType
import copy
import os  # used for infuser
import warnings
import sys
from .. import module

# local
import ivy
# from tracing_caching.cacher import CACHER
# from tracing_caching.telemetry import get_func_stats, get_graph_repr
from ..utils import assertions
from .graph import Graph, LazyGraph, SubGraph
from . import globals as glob
from .wrapping import (
    _wrap_functions_for_op_logging,
    _unwrap_functions_from_op_logging,
    _wrap_function_for_op_logging,
    FUNC_TO_PATH,
)
from .helpers import (
    _apply_fn_to_class,
    _deepcopy,
    _apply_fn_to_module,
    _check_is_trainable_module,
)
from .reloader import apply_and_reload
from .conversion import nest_array_to_new_backend, track, to_native
from . import tracked_var_proxy as tvp
from .special_ops import builtin_helpers as bh


# Helpers #
# ------- #


def clear_graph_cache():
    """Clears the graph cache which gets populated if `graph_caching` is set
    to `True` in `ivy.trace_graph`, `ivy.transpile` or `ivy.unify`. Use this to
    reset or clear the graph cache if needed.

    Examples
    --------
    >>> import ivy
    >>> ivy.clear_graph_cache()
    """

    global CACHER
    CACHER.clear_graph_cache()


def delete_graph_cache():
    """Deletes the graph cache which gets populated if `graph_caching` is set
    to `True` in `ivy.trace_graph`, `ivy.transpile` or `ivy.unify`. Use this to
    drop/delete the graph cache incl. the cache table s.t. a new table gets created
    during the next trace/transpile run.
    """

    global CACHER
    CACHER.delete_graph_cache()


def _reset_globals(initial_globals):
    glob.tracing_paused = initial_globals[0]
    glob.use_reloader = initial_globals[1]
    glob.tracing_stack.clear()
    glob.transformed_callables.clear()
    glob.iterator_chains = dict()
    glob.raw_id_to_weakref = dict()
    glob.raw_id_to_unique_id["train"] = dict()
    glob.raw_id_to_unique_id["eval"] = dict()
    glob.dependent_ids["train"] = set()
    glob.dependent_ids["eval"] = set()
    glob.subgraph_dependent_ids = set()
    glob.wrapped_fns = dict()


def _disable_native_compiler():
    """Disable native compilers so our tracing works."""
    original_flag = None
    if ivy.current_backend_str() == "tensorflow":
        import tensorflow as tf

        original_flag = tf.config.functions_run_eagerly()
        tf.config.run_functions_eagerly(True)
    elif ivy.current_backend_str() == "jax":
        import jax

        original_flag = jax.config.jax_disable_jit
        jax.config.update("jax_disable_jit", True)
    elif ivy.current_backend_str() == "torch":
        try:
            import torch._dynamo

            original_flag = torch._dynamo.config.disable
            torch._dynamo.config.disable = True
        except ModuleNotFoundError:
            pass
    return original_flag


def _restore_native_compiler_flag(original_flag):
    if ivy.current_backend_str() == "tensorflow":
        import tensorflow as tf

        tf.config.run_functions_eagerly(original_flag)
    elif ivy.current_backend_str() == "jax":
        import jax

        jax.config.update("jax_disable_jit", original_flag)
    elif ivy.current_backend_str() == "torch":
        try:
            import torch._dynamo

            torch._dynamo.config.disable = original_flag
        except ModuleNotFoundError:
            pass


def _disable_einops_caching():
    if "einops" in sys.modules:
        import einops

        cached_reconstruct = einops.einops._reconstruct_from_shape
        if hasattr(cached_reconstruct, "__wrapped__"):
            einops.einops._reconstruct_from_shape = cached_reconstruct.__wrapped__
        return cached_reconstruct


def _enable_einops_caching(cached_reconstruct):
    if "einops" in sys.modules and cached_reconstruct is not None:
        import einops

        einops.einops._reconstruct_from_shape = cached_reconstruct


def _restore_transformed_callables():
    for callable, transformed_fn in glob.transformed_callables:
        setattr(callable, "__call__", transformed_fn.__wrapped__)


# Tracing #
# ------- #


def _create_graph(
    fn: Callable,
    *args: Any,
    initial_globals: Tuple[bool, bool],
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    subgraph: bool = False,
    to_ivy: bool = False,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = False,
    modes_to_trace: str = "all",
    **kwargs: Any,
) -> Graph:
    """

    Parameters
    ----------
    fn
        function to trace and create a graph of
    args
        positional arguments to `fn`
    stateful
        list of instances to be considered stateful during the graph tracing
    arg_stateful_idxs
        positional arguments to be considered stateful during the graph tracing
    kwarg_stateful_idxs
        keyword arguments to be considered stateful during the graph tracing
    include_generators
        include array creation/generation functions as part of the graph
    array_caching
        cache the constant arrays that appear as arguments to the functions in the graph;
        these arrays are not created using some tracked arrays, they are usually generated/created.
    kwargs
        keyword arguments to `fn`

    Returns
    -------
    graph
        returns the traced graph

    Example
    -------
    >>> import ivy
    >>> from tracer.tracer import _create_graph
    >>> ivy.set_backend("torch")
    >>> x = ivy.array([1.])

    >>> def fn(x):
    ...     a = ivy.sum(x)
    ...     b = ivy.prod(x)
    ...     c = ivy.mean(x)
    ...     return a, b, c

    >>> graph = _create_graph(fn, x)

    Our graph stores the ids of the outputs (we have 3 outputs in this case):

    >>> a_id, b_id, c_id = graph._output_param_ids
    >>> print(a_id, b_id, c_id)
    140334650901584 140334650904064 3993654529954946995

    The graph also stores which function produced any given parameter in
    `_id_to_function` (which is useful later in the tracing process
    when we recurse backwards from the output ids to the inputs):

    >>> print(graph._id_to_function["eval"][b_id].__name__)
    prod

    """
    assertions.assert_framework_set(ivy.current_backend_str())

    # transform builtin function calls
    try:
        transpiling = kwargs.get("transpiling", False)
        trace_builtins = (transpiling and fn._contains_builtin_callables) or (
            not transpiling and ivy.current_backend_str() not in ("tensorflow", "jax")
        )
        if trace_builtins:
            fn = bh.transform_builtins(fn, depth=0, transpiling=transpiling)
    except Exception:
        warnings.warn(
            "Any builtin callables in the source code will not be tracked properly "
            "and might get cached in the graph."
        )

    # extra stateful instances modified in the graph
    stateful = ivy.default(stateful, [])
    arg_stateful_idxs = ivy.default(arg_stateful_idxs, [])
    stateful_args = ivy.multi_index_nest(args, arg_stateful_idxs)
    kwarg_stateful_idxs = ivy.default(kwarg_stateful_idxs, [])
    stateful_kwargs = ivy.multi_index_nest(kwargs, kwarg_stateful_idxs)
    all_stateful = stateful + stateful_args + stateful_kwargs

    # copy stateful arguments to avoid modifying the originals during trace
    args_copied = ivy.map_nest_at_indices(args, arg_stateful_idxs, _deepcopy)
    kwargs_copied = ivy.map_nest_at_indices(kwargs, kwarg_stateful_idxs, _deepcopy)

    _to_ignore = tvp.get_types_to_ignore()

    # ensure that arguments are from the required framework
    # using 'native' argument to define whether a native array or ivy array should be returned
    args_copied = nest_array_to_new_backend(
        args_copied, native=not to_ivy, to_ignore=_to_ignore
    )
    kwargs_copied = nest_array_to_new_backend(
        kwargs_copied,
        native=not to_ivy,
        to_ignore=_to_ignore,
    )

    # extract the associated stateful classes
    all_stateful_classes = [s.__class__ for s in all_stateful]

    # copy the states for resetting after forward pass and tracing
    all_state_copies = list()
    for s in all_stateful:
        state_copy = _deepcopy(s).__dict__
        if isinstance(s, dict):
            state_copy = {**state_copy, **s}
        all_state_copies.append(state_copy)

    # track all non-array classes if available as arbitrary tracked variables
    args_copied = track(
        args_copied, with_numpy=with_numpy, stateful_classes=tuple(all_stateful_classes)
    )
    kwargs_copied = track(
        kwargs_copied,
        with_numpy=with_numpy,
        stateful_classes=tuple(all_stateful_classes),
    )

    # construct the graph
    if not subgraph:
        graph = Graph(
            fn,
            *args_copied,
            **kwargs_copied,
            stateful=stateful,
            arg_stateful_idxs=arg_stateful_idxs,
            kwarg_stateful_idxs=kwarg_stateful_idxs,
            include_generators=include_generators,
            array_caching=array_caching,
            with_numpy=with_numpy,
            modes_to_trace=modes_to_trace,
            to_ivy=to_ivy,
        )
    else:
        graph = SubGraph(
            fn,
            *args_copied,
            **kwargs_copied,
            stateful=stateful,
            arg_stateful_idxs=arg_stateful_idxs,
            kwarg_stateful_idxs=kwarg_stateful_idxs,
            include_generators=include_generators,
            array_caching=array_caching,
            with_numpy=with_numpy,
            modes_to_trace=glob.current_trace_mode,
            to_ivy=to_ivy,
        )
    original_flag = _disable_native_compiler()
    cached_einops = _disable_einops_caching()
    # wrap all functions for operation logging
    if subgraph:
        using_reloader = glob.use_reloader
        glob.use_reloader = False
    if not graph._is_subgraph:
        graph._fn = apply_and_reload(
            to_reload=graph._fn,
            to_apply=_wrap_functions_for_op_logging,
            args=(
                graph,
                all_stateful_classes,
            ),
            kwargs={
                "to_ivy": to_ivy,
                "with_numpy": with_numpy,
                "trace_builtins": trace_builtins,
            },
            stateful=[id(cls) for cls in all_stateful_classes],
        )
    if subgraph:
        glob.use_reloader = using_reloader
    if graph._fn in FUNC_TO_PATH:
        graph._fn = _wrap_function_for_op_logging(graph._fn, graph, to_ivy=to_ivy)
    try:
        graph.log_all_ops()
    except Exception as e:
        if not graph._is_subgraph:
            graph._fn = apply_and_reload(
                to_reload=graph._fn,
                to_apply=_unwrap_functions_from_op_logging,
                args=(all_stateful_classes,),
                kwargs={
                    "to_ivy": to_ivy,
                    "with_numpy": with_numpy,
                    "trace_builtins": trace_builtins,
                },
                stateful=[id(cls) for cls in all_stateful_classes],
            )
        _restore_transformed_callables()
        if not graph._is_subgraph:
            _reset_globals(initial_globals)
        _restore_native_compiler_flag(original_flag)
        _enable_einops_caching(cached_einops)
        raise e
    _restore_transformed_callables()
    _restore_native_compiler_flag(original_flag)
    _enable_einops_caching(cached_einops)
    if not graph._is_subgraph:
        # unwrap all functions now tracing is done, but only unwrap if
        # we are done tracing the main graph (rather than any subgraphs)
        graph._fn = apply_and_reload(
            to_reload=graph._fn,
            to_apply=_unwrap_functions_from_op_logging,
            args=(all_stateful_classes,),
            kwargs={
                "to_ivy": to_ivy,
                "with_numpy": with_numpy,
                "trace_builtins": trace_builtins,
            },
            stateful=[id(cls) for cls in all_stateful_classes],
        )

    # reset the stateful objects to their initial state, prior to tracing
    for s, sc in zip(all_stateful, all_state_copies):
        for k in list(s.__dict__.keys()):
            if k not in sc:
                del s.__dict__[k]
                continue
            s.__dict__[k] = sc[k]
        if isinstance(s, dict):
            for k in list(s.keys()):
                if k not in sc:
                    del s[k]
                    continue
                s[k] = sc[k]

    return graph


def trace_graph(
    *objs: Callable,
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    to: Optional[str] = None,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = True,
    modes_to_trace: str = "all",
    backend_compile: bool = False,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
    compile_mode: Optional[str] = None,
    graph_caching: bool = True,
    args: Optional[Sequence] = None,
    kwargs: Optional[Mapping] = None,
    params_v=None,
    v=None,
) -> Union[Graph, LazyGraph]:
    """Takes `fn` and traces it into a more efficient composition of backend operations.

    Parameters
    ----------
    objs
        callable(s) to trace and create a graph of
    stateful
        list of instances to be considered stateful during the graph tracing
    arg_stateful_idxs
        positional arguments to be considered stateful during the graph tracing
    kwarg_stateful_idxs
        keyword arguments to be considered stateful during the graph tracing
    include_generators
        include array creation/generation functions as part of the graph
    array_caching
        cache the constant arrays that appear as arguments to the functions in the graph
    modes_to_trace
        the module mode(s) which should be traced when tracing a trainable module
        can be either "all", "train" or "eval".
    backend_compile
        whether to apply the native compilers, i.e. tf.function, after ivy's tracing
    static_argnums
        for jax's jit compilation
    static_argnames
        for jax's jit compilation
    compile_mode
        mode for torch's compilation
    graph_caching
        whether to cache the traced graph
    args
        positional arguments for `obj`
    kwargs
        keyword arguments for `obj`

    Returns
    -------
    the traced `Graph` object.

    Examples
    --------
    >>> import ivy, time
    >>> from trace.tracer import trace_graph
    >>> ivy.set_backend("torch")
    >>> x = ivy.array([1.])

    >>> def fn(x):
    ...     y = ivy.sum(x)
    ...     z = ivy.prod(x)
    ...     a = ivy.sin(y)
    ...     b = ivy.cos(z)
    ...     c = ivy.tan(z)
    ...     i = ivy.round(a)
    ...     j = ivy.floor(b)
    ...     k = ivy.ceil(c)
    ...     return i, j, k


    >>> graph = trace_graph(fn, args=(x,))

    Notice how the time taken to execute the traced function is lower than
    the original function. A typical run:

    >>> start = time.time()
    >>> fn(x)
    >>> print(time.time() - start)
    0.0003559589385986328

    >>> start = time.time()
    >>> graph(x)
    >>> print(time.time() - start)
    0.0001785755157470703
    """
    assertions.assert_valid_to(to, transpiling=False, is_lazy=args is None and kwargs is None)
    assert modes_to_trace in [
        "all",
        "train",
        "eval",
    ], 'modes_to_trace needs to be one of "all", "train" or "eval"'

    if to is None and ivy.current_backend_str() != "":
        to = ivy.current_backend_str()

    infuser_injection = "TRACER SERVER VERIFY"  # needed for infuser

    _trace_kwargs = {
        "stateful": stateful,
        "arg_stateful_idxs": arg_stateful_idxs,
        "kwarg_stateful_idxs": kwarg_stateful_idxs,
        "to": to,
        "include_generators": include_generators,
        "array_caching": array_caching,
        "with_numpy": with_numpy,
        "modes_to_trace": modes_to_trace,
        "backend_compile": backend_compile,
        "static_argnums": static_argnums,
        "static_argnames": static_argnames,
        "compile_mode": compile_mode,
    }

    # this is being used as a decorator, only if there are no positional args
    if len(objs) == 0:

        def decorator(func):
            return trace_graph(
                func,
                args=args,
                kwargs=kwargs,
                **_trace_kwargs,
            )

        return decorator

    if len(objs) > 1:
        return tuple(
            trace_graph(
                o,
                args=args,
                kwargs=kwargs,
                **_trace_kwargs,
            )
            for o in objs
        )

    obj = objs[0]

    # check if obj is a module or a function
    if isinstance(obj, ModuleType):
        return _apply_fn_to_module(
            obj,
            fn=trace_graph,
            args=args,
            kwargs=kwargs,
            **_trace_kwargs,
        )

    # check if obj is a class
    if isinstance(obj, type):
        return _apply_fn_to_class(
            obj,
            trace_graph,
            fargs=args,
            fkwargs=kwargs,
            **_trace_kwargs,
        )

    is_trainable_module, to, to_mod = _check_is_trainable_module(
        obj, to if to else ivy.current_backend_str()
    )

    if is_trainable_module:
        traced_graph = None
        no_cache_exists = False
        # No caching for trainable modules
        graph_caching = False
        if graph_caching:
            _copied_trace_kwargs = copy.copy(_trace_kwargs)
            _copied_trace_kwargs["current_backend"] = (
                to if to is not None and to != "ivy" else ivy.current_backend_str()
            )
            CACHER.prepare_graph_cache()
            traced_data, cached_data = CACHER.get_graph_cached_traced_data(
                obj, args, kwargs, _copied_trace_kwargs
            )
            traced_graph = CACHER.get_matching_graph(traced_data, cached_data)

        if traced_graph is None:
            no_cache_exists = True

            traced_graph = module._trace_trainable_module(
                obj,
                to=to,
                to_mod=to_mod,
                args=args,
                kwargs=kwargs,
                with_numpy=with_numpy,
                graph_caching=False,
                modes_to_trace=modes_to_trace,
                backend_compile=backend_compile,
                params_v=params_v,
            )

        if graph_caching and no_cache_exists and (args is not None or kwargs is not None):
            if hasattr(traced_graph, "_ivy_module"):
                fn_str, constants = traced_graph._ivy_module._module_graph.obtain_sourcecode()
                CACHER.store_graph_cache(
                    traced_data,
                    fn_str,
                    constants,
                    source="",
                    to=_trace_kwargs["to"],
                )
            elif hasattr(traced_graph, "ivy_module"):
                # graph._ivy_module is not accessible in flax, use graph.ivy_module instead
                fn_str, constants = traced_graph.ivy_module._module_graph.obtain_sourcecode()
                CACHER.store_graph_cache(
                    traced_data,
                    fn_str,
                    constants,
                    source="",
                    to=_trace_kwargs["to"],
                )
            elif isinstance(traced_graph, ivy.Module):
                fn_str, constants = traced_graph._module_graph.obtain_sourcecode()
                CACHER.store_graph_cache(
                    traced_data,
                    fn_str,
                    constants,
                    source="",
                    to=_trace_kwargs["to"],
                )

        return traced_graph

    # return eager graph if args or kwargs are supplied
    if (args is not None) or (kwargs is not None):
        to = _trace_kwargs["to"]
        original_backend = ivy.current_backend_str()        

        if to and ivy.current_backend_str() != to and to != "ivy":
            ivy.set_backend(to)

        args = ivy.default(args, ())
        kwargs = ivy.default(kwargs, {})
        if ivy.exists(v):
            kwargs = copy.copy(kwargs)
            kwargs["v"] = v

        traced_graph = None
        no_cache_exists = False
        if graph_caching:
            _copied_trace_kwargs = copy.copy(_trace_kwargs)
            _copied_trace_kwargs["current_backend"] = (
                to if to is not None and to != "ivy" else ivy.current_backend_str()
            )
            CACHER.prepare_graph_cache()
            traced_data, cached_data = CACHER.get_graph_cached_traced_data(
                obj, args, kwargs, _copied_trace_kwargs
            )
            traced_graph = CACHER.get_matching_graph(traced_data, cached_data)

        if traced_graph is None:
            no_cache_exists = True
            traced_graph = _trace_graph(
                obj,
                *args,
                **kwargs,
                **_trace_kwargs,
            )

        if graph_caching and no_cache_exists:
            fn_str, constants = traced_graph.obtain_sourcecode()
            CACHER.store_graph_cache(
                traced_data,
                fn_str,
                constants,
                source="",
                to=_trace_kwargs["to"],
            )
        
        # if glob.connector is not None:
        #     if not (glob.is_transpile or glob.is_unify):
        #         ivy_func = "compile"
        #         flags = copy.copy(_trace_kwargs)
        #         if flags["stateful"] is not None:
        #             flags["stateful"] = [str(s) for s in flags["stateful"]]
        #         funcs, func_freq = get_func_stats(traced_graph)
        #         graph_dict = get_graph_repr(traced_graph, ivy_func)
        #         glob.connector.log_api_call(
        #             ivy_func,
        #             flags,
        #             funcs=funcs,
        #             func_freq=func_freq,
        #             graph=graph_dict,
        #         )

        if ivy.current_backend_str() != original_backend:
            ivy.previous_backend()

        infuser_injection = "TRACER TELEMETRY"  # needed for infuser

        return traced_graph

    # untraced
    awaiting = LazyGraph(
        obj,
        initializer=trace_graph,
        **_trace_kwargs,
    )

    return awaiting


def _trace_graph(
    fn: Callable,
    *args: Any,
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    to: Optional[str] = None,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = False,
    modes_to_trace: str = "all",
    backend_compile: bool = False,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
    compile_mode: Optional[str] = None,
    **kwargs: Any,
) -> Union[Graph, Callable]:
    if ivy.current_backend_str() == "numpy" or to == "ivy" and backend_compile:
        backend_compile = False
        warnings.warn(
            "Numpy or Ivy does not support tracing natively, `backend_compile` is set to False."
        )
    # todo: add feature: stateful native tracing
    if stateful or arg_stateful_idxs or kwarg_stateful_idxs and backend_compile:
        backend_compile = False
        warnings.warn(
            "Native tracing does not support stateful classes yet, `backend_compile` is set to False."
        )
    if (
        include_generators
        and not backend_compile
        and ivy.current_backend_str() == "jax"
    ):
        stateful = ivy.default(stateful, [])
        stateful += [ivy.functional.backends.jax.RNG]

    initial_globals = (glob.tracing_paused, glob.use_reloader)

    graph = _create_graph(
        fn,
        *args,
        stateful=stateful,
        arg_stateful_idxs=arg_stateful_idxs,
        kwarg_stateful_idxs=kwarg_stateful_idxs,
        to_ivy=to == "ivy",
        include_generators=include_generators,
        array_caching=array_caching,
        with_numpy=with_numpy,
        modes_to_trace=modes_to_trace,
        initial_globals=initial_globals,
        **kwargs,
    )
    graph._backend_compile = backend_compile
    graph._static_argnums = static_argnums
    graph._static_argnames = static_argnames
    graph._compile_mode = compile_mode

    # trace the graph forward pass into an executable function
    graph.traced()

    _reset_globals(initial_globals)
    graph = _backend_compile(
        graph,
        backend_compile,
        static_argnums,
        static_argnames,
        compile_mode,
        *args,
        **kwargs,
    )

    return graph


def _backend_compile(
    graph,
    backend_compile,
    static_argnums,
    static_argnames,
    compile_mode,
    *args,
    **kwargs,
):
    if not backend_compile:
        return graph
    args = to_native(args)
    kwargs = to_native(kwargs)
    if graph.backend == "jax":
        import jax

        backend_compiler = lambda x: jax.jit(
            x[0], static_argnums=x[1], static_argnames=x[2]
        )
    elif graph.backend == "torch":
        import torch

        if torch.__version__ >= "2.0":
            backend_compiler = lambda x: torch.compile(x[0], mode=x[3])
        else:
            return graph
        # ToDo: This breaks in previous versions
        # else:
        #     backend_compiler = lambda x: torch.jit.trace(x[0])
    elif graph.backend == "tensorflow":
        import tensorflow as tf

        backend_compiler = lambda x: tf.function(x[0], jit_compile=True)
    elif graph.backend == "paddle":
        import paddle

        if paddle.__version__ > "2.4.2":
            backend_compiler = lambda x: paddle.jit.api.dygraph_to_static_func(x[0])
        else:
            backend_compiler = (
                lambda x: paddle.fluid.dygraph.jit.dygraph_to_static_func(x[0])
            )

    graph.graph_call = backend_compiler(
        (
            graph.graph_call,
            static_argnums,
            static_argnames,
            compile_mode,
        )
    )
    _ = graph(*args, **kwargs)  # ToDo: Make this cleaner
    return graph


def trace_subgraph(
    fn: Callable,
    to: Optional[str] = None,
    id_to_fn: dict = {},
    id_to_param: dict = {},
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = True,
    args: Optional[Sequence] = None,
    kwargs: Optional[Mapping] = None,
) -> Union[Graph, LazyGraph]:

    glob.tracing_subgraph = True

    initial_globals = (
        glob.tracing_paused,
        glob.use_reloader,
    )

    subgraph = _create_graph(
        fn,
        *args,
        subgraph=True,
        to_ivy=to == "ivy",
        include_generators=include_generators,
        array_caching=array_caching,
        with_numpy=with_numpy,
        initial_globals=initial_globals,
        modes_to_trace=glob.current_trace_mode,
        **kwargs,
    )

    # load the existing main graphs fns and ids appropriately into the subgraph
    if glob.current_trace_mode == "train":
        for main_graph_id, main_graph_fn in id_to_fn["train"].items():
            if main_graph_id not in subgraph._id_to_function["train"]:
                subgraph._id_to_function["train"][main_graph_id] = main_graph_fn
                subgraph._greater_scope_ids.append(main_graph_id)
        for param_id in id_to_param.keys():
            if param_id not in subgraph._id_to_function["train"]:
                subgraph._greater_scope_ids.append(param_id)
    else:
        for main_graph_id, main_graph_fn in id_to_fn["eval"].items():
            if main_graph_id not in subgraph._id_to_function["eval"]:
                subgraph._id_to_function["eval"][main_graph_id] = main_graph_fn
                subgraph._greater_scope_ids.append(main_graph_id)
        for param_id in id_to_param.keys():
            if param_id not in subgraph._id_to_function["eval"]:
                subgraph._greater_scope_ids.append(param_id)

    # trace the subgraph
    subgraph.traced()

    # reset only the necessary globals, so as not to affect the main graph tracing
    glob.tracing_paused = initial_globals[0]
    glob.use_reloader = initial_globals[1]
    glob.subgraph_dependent_ids = set()
    glob.tracing_subgraph = False

    return subgraph
