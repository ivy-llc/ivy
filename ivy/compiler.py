# global
import ivy
import copy
import inspect

# local
from ivy.wrapper import _wrap_or_unwrap_methods, NON_WRAPPED_METHODS, NON_ARRAY_RET_METHODS


class Graph:

    # noinspection PyProtectedMember
    def __init__(self, fn, *args, **kwargs):

        # function being compiled into a graph
        self._fn = fn

        # positional args
        self._args = args
        self._arg_nest_idxs = ivy.nested_indices_where(args, lambda x: isinstance(x, ivy.Array))
        self._arg_param_ids = [id(x._data) for x in ivy.multi_index_nest(list(args), self._arg_nest_idxs)]

        # key-word args
        self._kwargs = kwargs
        self._kwarg_nest_idxs = ivy.nested_indices_where(kwargs, lambda x: isinstance(x, ivy.Array))
        self._kwarg_param_ids = [id(x._data) for x in ivy.multi_index_nest(kwargs, self._kwarg_nest_idxs)]

        # output idxs
        self._output_param_ids = list()

        # graph storage
        self._param_dict = dict()
        self._functions = list()

    # Compute output idxs #
    # --------------------#

    # noinspection PyProtectedMember
    def foward(self):
        ret = self._fn(*self._args, **self._kwargs)
        if not isinstance(ret, tuple):
            ret = (ret,)
        output_nest_idxs = ivy.nested_indices_where(ret, lambda x: ivy.is_array(x))
        self._output_param_ids = [id(x._data) for x in ivy.multi_index_nest(list(ret), output_nest_idxs)]

    # Setters #
    # --------#

    def set_param(self, idx, param):
        self._param_dict[idx] = param

    def add_fn(self, fn):
        self._functions.append(fn)

    # Function creation #
    # ------------------#

    def _call(self, *args, **kwargs):
        # ToDo: make this as efficient as possible; this is performed at runtime
        [self.set_param(pid, ivy.index_nest(args, idx))
         for pid, idx in zip(self._arg_param_ids, self._arg_nest_idxs)]
        [self.set_param(pid, ivy.index_nest(kwargs, idx))
         for pid, idx in zip(self._kwarg_param_ids, self._kwarg_nest_idxs)]
        for fn in self._functions:
            arg_vals = [self._param_dict[idx_] for idx_ in fn.arg_param_ids]
            kwarg_vals = [self._param_dict[idx_] for idx_ in fn.kwarg_param_ids]
            ret = fn(arg_vals, kwarg_vals)
            [self.set_param(pid, ivy.index_nest(ret, idx))
             for pid, idx in zip(fn.output_param_ids, fn.output_nest_idxs)]
        ret = [self._param_dict[pid] for pid in self._output_param_ids]
        if len(ret) == 1:
            return ret[0]
        return ret

    def to_function(self):
        return self._call

    # Clearing #
    # ---------#

    def clear(self):
        self._param_dict.clear()
        self._functions.clear()


graph = None


# Methods #

def _wrap_method_for_compiling(fn):

    if inspect.isclass(fn):
        return fn
    elif hasattr(fn, '__name__') and\
            (fn.__name__[0] == '_' or fn.__name__ in NON_WRAPPED_METHODS + NON_ARRAY_RET_METHODS) and\
            (not hasattr(fn, '__qualname__') or '__init__' in fn.__qualname__ or fn.__qualname__[0:8] != 'Array.__'):
        return fn
    elif hasattr(fn, 'wrapped_for_compiling') and fn.wrapped_for_compiling:
        return fn

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def _method_wrapped(*args, **kwargs):

        # convert to native if necessary
        args, kwargs = ivy.args_to_native(*args, **kwargs)
        args = list(args)

        # get array idxs for positional args
        arg_array_idxs = ivy.nested_indices_where(args, lambda x: ivy.is_array(x))
        args_cont = ivy.Container(args, types_to_iteratively_nest=(list, tuple))
        arg_idxs_cont = args_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        arg_param_ids = list(arg_idxs_cont.to_iterator_values())

        # get array idxs for key-word args
        kwarg_array_idxs = ivy.nested_indices_where(kwargs, lambda x: ivy.is_array(x))
        kwargs_cont = ivy.Container(kwargs, types_to_iteratively_nest=(list, tuple))
        kwarg_idxs_cont = kwargs_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        kwarg_param_ids = list(kwarg_idxs_cont.to_iterator_values())

        # initialize empty keys for these input parameters in the graph
        [graph.set_param(idx, None) for idx in arg_param_ids + kwarg_param_ids]

        # compute the return
        ret_raw = fn(*args, **kwargs)
        ret = ret_raw if isinstance(ret_raw, tuple) else (ret_raw,)

        # get array idxs for return
        ret_nest_idxs = ivy.nested_indices_where(ret, lambda x: ivy.is_array(x))
        ret_cont = ivy.Container(ret, types_to_iteratively_nest=(list, tuple))
        ret_idxs_cont = ret_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        ret_param_ids = list(ret_idxs_cont.to_iterator_values())

        # wrap the function
        def new_fn(arg_array_vals, kwarg_array_vals):
            # ToDo: make this as efficient as possible; this is performed at runtime
            ivy.set_nest_at_indices(args, arg_array_idxs, arg_array_vals)
            ivy.set_nest_at_indices(kwargs, kwarg_array_idxs, kwarg_array_vals)
            ret_ = fn(*args, **kwargs)
            if not isinstance(ret_, tuple):
                ret_ = (ret_,)
            return ret_

        # add function attributes which inform about the input idxs
        new_fn.arg_param_ids = arg_param_ids
        new_fn.kwarg_param_ids = kwarg_param_ids
        new_fn.output_param_ids = ret_param_ids
        new_fn.output_nest_idxs = ret_nest_idxs

        # initialize empty keys for these output parameters in the graph
        [graph.set_param(id(v), None)
         for _, v in ivy.Container(ret, types_to_iteratively_nest=(list, tuple)).to_iterator() if ivy.is_array(v)]

        # add this function to the graph
        graph.add_fn(new_fn)

        # return the function output, as ivy.Array instances
        return ivy.to_ivy(ret_raw)

    if hasattr(fn, '__name__'):
        _method_wrapped.__name__ = fn.__name__
    _method_wrapped.wrapped_for_compiling = True
    _method_wrapped.inner_fn = fn
    return _method_wrapped


def _unwrap_method_from_compiling(method_wrapped):
    if not hasattr(method_wrapped, 'wrapped_for_compiling') or not method_wrapped.wrapped_for_compiling:
        return method_wrapped
    return method_wrapped.inner_fn


def _wrap_methods_for_compiling():
    return _wrap_or_unwrap_methods(_wrap_method_for_compiling, classes_to_wrap=[ivy.Array])


def _unwrap_methods_from_compiling():
    return _wrap_or_unwrap_methods(_unwrap_method_from_compiling, classes_to_wrap=[ivy.Array])


def compile_ivy(fn, *args, **kwargs):
    args, kwargs = ivy.args_to_ivy(*args, **kwargs)
    global graph
    graph = Graph(fn, *args, **kwargs)
    _wrap_methods_for_compiling()
    # ToDo: work out why the command below assings to the graph param_dict, which cannot then be clearer by any means
    ivy.set_wrapped_mode()
    graph.foward()
    ivy.unset_wrapped_mode()
    _unwrap_methods_from_compiling()
    local_graph = copy.deepcopy(graph)
    fn = local_graph.to_function()
    graph.clear()
    return fn
