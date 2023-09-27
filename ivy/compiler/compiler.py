from typing import Callable, Optional, List, Union, Iterable, Tuple, Any


# TODO: create meaningful types for Graph and LazyGraph,
# will probably need a seperate file for that
class Graph:
    pass


class LazyGraph:
    pass


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
    kwargs: Optional[dict] = None,
) -> Union[Graph, LazyGraph]:
    """
    Take `fn` and traces it into a more efficient composition of backend operations.

    Parameters
    ----------
    objs
        callable(s) to trace and create a graph of
    stateful
        list of instances to be considered stateful during the graph compilation
    arg_stateful_idxs
        positional arguments to be considered stateful during the graph compilation
    kwarg_stateful_idxs
        keyword arguments to be considered stateful during the graph compilation
    include_generators
        include array creation/generation functions as part of the graph
    array_caching
        cache the constant arrays that appear as arguments to the functions in the graph
    backend_compile
        whether to apply the native compilers, i.e. tf.function, after ivy's compilation
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
    >>> from ivy import compile
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
    with_numpy: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    params_v=None,
    v=None, 
) -> Union[Graph, LazyGraph]:
    from ._compiler import transpile as _transpile

    """
    Transpile Callable objects passed as arguments. If args and kwargs are specified,
    transpilation is performed eagerly, otherwise, transpilation will happen lazily.

    Parameters
    ----------
    objs
        The native Callables to be transpiled
    source
        The framework that `obj` is from.
    to
        The target framework to transpile `obj` to.
    debug_mode
        Whether to transpile to ivy first, before the final compilation
        to the target framework. If the target is ivy, then this flag
        makes no difference.
    args
        If specified, arguments that will be used to transpile eagerly.
    kwargs
        If specified, keyword arguments that will be used to transpile eagerly.

    Returns
    -------
    Either a transpiled Graph or a non-initialized LazyGraph.
    """
    return _transpile(
        *objs,
        source=source,
        to=to,
        with_numpy=with_numpy,
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
    **transpile_kwargs,
) -> Callable:
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
