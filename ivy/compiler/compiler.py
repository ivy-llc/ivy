from typing import Callable, Optional, List, Union, Iterable, Tuple
import json
import ivy


def trace_graph(
    *objs: Callable,
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    to: Optional[str] = None,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = True,
    backend_compile: bool = False,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
    mode: Optional[str] = None,
    graph_caching: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None
):
    """
    Takes `fn` and traces it into a more efficient composition of backend operations.

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
    backend_compile
        whether to apply the native compilers, i.e. tf.function, after ivy's tracing
    static_argnums
        for jax's jit compilation
    static_argnames
        for jax's jit compilation
    mode
        for torch's compilation
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
    >>> from ivy import trace_graph
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

    from ._compiler import compile as _trace_graph

    return _trace_graph(
        *objs,
        stateful=stateful,
        arg_stateful_idxs=arg_stateful_idxs,
        kwarg_stateful_idxs=kwarg_stateful_idxs,
        to=to,
        include_generators=include_generators,
        array_caching=array_caching,
        with_numpy=with_numpy,
        backend_compile=backend_compile,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        mode=mode,
        graph_caching=graph_caching,
        args=args,
        kwargs=kwargs,
    )


def transpile(
    *objs: Callable,
    source: Optional[str] = None,
    to: Optional[str] = None,
    with_numpy: bool = True,
    backend_compile: bool = False,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
    mode: Optional[str] = None,
    graph_caching: bool = False,
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    params_v=None,
    v=None
):
    """
    Transpiles Callable objects passed as arguments. If args and kwargs are specified,
    transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters
    ----------
    objs
        The native Callables to be transpiled
    source
        The framework that `obj` is from.
    to
        The target framework to transpile `obj` to.
    args
        If specified, arguments that will be used to transpile eagerly.
    kwargs
        If specified, keyword arguments that will be used to transpile eagerly.

    Returns
    -------
    Either a transpiled Graph or a non-initialized LazyGraph.
    """

    from ._compiler import transpile as _transpile

    return _transpile(
        *objs,
        source=source,
        to=to,
        with_numpy=with_numpy,
        backend_compile=backend_compile,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        mode=mode,
        graph_caching=graph_caching,
        stateful=stateful,
        arg_stateful_idxs=arg_stateful_idxs,
        kwarg_stateful_idxs=kwarg_stateful_idxs,
        args=args,
        kwargs=kwargs,
        params_v=params_v,
        v=v,
    )


def unify(
    *objs: Callable,
    source: Optional[str] = None,
    graph_caching: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    with_numpy: bool = True,
    **transpile_kwargs
):
    from ._compiler import unify as _unify

    return _unify(
        *objs,
        source=source,
        graph_caching=graph_caching,
        args=args,
        kwargs=kwargs,
        with_numpy=with_numpy,
        **transpile_kwargs,
    )


def native_compile(
    obj: Callable,
    backend_kwargs: Optional[dict] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
):
    raise NotImplementedError()

    return ivy.current_backend().native_compile(
        obj=obj, backend_kwargs=backend_kwargs, args=args, kwargs=kwargs
    )


def _apply_autotuner_config(
    obj: Callable,
    config: Union[dict, str],
    return_obj: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
):
    """
    Applies a given configuration file to a callable object and applies the
    corresponding transformations.

    Parameters
    ----------
    obj
        The Callable that will be transformed.
    config
        Configuration file generated by Ivy's autotuner. This can be either the direct output of the `autotune`
        function or a path to a json file.
    args
        The arguments to be used during tracing.
    kwargs
        The keyword arguments to be used during tracing.

    Returns
    -------
    A callable object with the transformations specified in the config file applied.
    """

    if not isinstance(config, dict):
        with open(config) as json_file:
            config = json.load(json_file)

    source_fw = config["source"]
    target_fw = config["framework"]
    device = config["device"]
    if target_fw == "torch":
        mode = config.get("mode")
        native_backend = config.get("native_backend")

    if hasattr(obj, "__call__"):
        obj = obj.__call__

    transpiled_obj = transpile(
        obj,
        source=source_fw,
        to=target_fw,
        mode=mode,
        args=args,
        kwargs=kwargs,
        backend_compile=False,
    )

    transpiled_obj.to_device(device)

    out_obj = ivy.with_backend(target_fw).native_compile(
        transpiled_obj, mode=mode, backend=native_backend
    )

    if return_obj:
        return_obj

    # TODO: else: serialize

    raise NotImplementedError(
        "Serialization of the compiled output is not yet implemented."
    )


def compile(
    obj: Callable,
    engine: Optional[str] = None,
    engine_kwargs: Optional[dict] = None,
    return_obj: bool = False,
    config=None,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
):
    if config:
        return _apply_autotuner_config(
            obj=obj, config=config, return_obj=return_obj, args=args, kwargs=kwargs
        )

    raise NotImplementedError()


def autotune(
    obj: Callable,
    cost_func: Optional[Callable] = None,
    source: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    search_config: Optional[dict] = None,
    save_fig: bool = False,
    crop_fig: str = "full",
    cache_results: bool = False,
):
    """
    .

    Parameters
    ----------
    obj
        The Callable that will be autotuned.
    source
        The framework that `obj` is from.
    args
        The arguments to be passed while calling `obj`.
    kwargs
        The keyword arguments to be passed while calling `obj`.
    save_fig
        If True, creates a scatter plot showing the benchmarked results
        for different frameworks.

    Returns
    -------
    The results containing different benchmarked metrics for different frameworks.
    """
    from ._compiler import autotune as _autotune

    return _autotune(
        obj,
        cost_func=cost_func,
        source=source,
        search_config=search_config,
        save_fig=save_fig,
        crop_fig=crop_fig,
        cache_results=cache_results,
        args=args,
        kwargs=kwargs,
    )


def compress():
    raise NotImplementedError()
