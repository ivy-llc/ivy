from typing import Callable, Optional, List, Union, Iterable, Tuple


# TODO: create meaningful types for Graph and LazyGraph,
# will probably need a seperate file for that
class Graph:
    pass


class LazyGraph:
    pass


def compile(
    *objs: Callable,
    stateful: Optional[List] = None,
    arg_stateful_idxs: Optional[List] = None,
    kwarg_stateful_idxs: Optional[List] = None,
    to: Optional[str] = None,
    include_generators: bool = True,
    array_caching: bool = True,
    with_numpy: bool = False,
    return_backend_compiled_fn: bool = False,
    static_argnums: Optional[Union[int, Iterable[int]]] = None,
    static_argnames: Optional[Union[str, Iterable[str]]] = None,
    # dynamic: bool = False, # for torch.jit.script compilation
    graph_caching: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
) -> Union[Graph, LazyGraph]:
    from ._compiler import compile as _compile
    """
    Take `fn` and compiles it into a more efficient composition of backend operations.

    Parameters
    ----------
    objs
        callable(s) to compile and create a graph of
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
    return_backend_compiled_fn
        whether to apply the native compilers, i.e. tf.function, after ivy's compilation
    static_argnums
        for jax's jit compilation
    static_argnames
        for jax's jit compilation
    graph_caching
        whether to cache the compiled graph
    args
        positional arguments for `obj`
    kwargs
        keyword arguments for `obj`

    Returns
    -------
    the compiled `Graph` object.

    Examples
    --------
    >>> import ivy, time
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
    >>> graph = ivy.compile(fn, args=(x,))
    Notice how the time taken to execute the compiled function is lower than
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
    return _compile(
        *objs,
        stateful=stateful,
        arg_stateful_idxs=arg_stateful_idxs,
        kwarg_stateful_idxs=kwarg_stateful_idxs,
        to=to,
        include_generators=include_generators,
        array_caching=array_caching,
        with_numpy=with_numpy,
        return_backend_compiled_fn=return_backend_compiled_fn,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        # dynamic: bool = False, # for torch.jit.script compilation
        graph_caching=graph_caching,
        args=args,
        kwargs=kwargs,
    )


def transpile(
    *objs: Callable,
    source: Optional[str] = None,
    to: Optional[str] = None,
    debug_mode: bool = False,
    with_numpy: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    params_v=None,
    v=None,  # Make this cleaner
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
        debug_mode=debug_mode,
        with_numpy=with_numpy,
        args=args,
        kwargs=kwargs,
        params_v=params_v,
        v=v,
    )


def unify(
    *objs: Callable,
    source: Optional[str] = None,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
    with_numpy: bool = False,
    **transpile_kwargs,
) -> Callable:
    """
    Compiles and unifies multiple callable objects into a single callable. This is
    primarily used to create a single function that encapsulates the behavior of
    multiple functions.

    Parameters
    ----------
    *objs : Callable
        The callable objects (e.g., functions) to unify.

    source : str, optional
        The source code string to be included in the unified function. This can be
        useful if the unified function needs to include some custom logic not
        encapsulated by the input callable objects.

    args : tuple, optional
        A tuple of arguments to be passed to the unified function.

    kwargs : dict, optional
        A dictionary of keyword arguments to be passed to the unified function.

    with_numpy : bool, optional
        Whether the numpy module is to be included in the unified function's namespace.
        By default, numpy is not included. This argument is useful when the input
        callable objects make use of numpy functions.

    **transpile_kwargs
        Arbitrary keyword arguments to be passed to the underlying transpiler.

    Returns
    -------
    Callable
        The unified callable object.
    """
    from ._compiler import unify as _unify

    return _unify(
        *objs,
        source=source,
        args=args,
        kwargs=kwargs,
        with_numpy=with_numpy,
        **transpile_kwargs,
    )
