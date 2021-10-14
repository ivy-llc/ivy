# global
import copy

import ivy

# local
from ivy.wrapper import _wrap_or_unwrap_methods, NON_WRAPPED_METHODS, NON_ARRAY_RET_METHODS

param_dict = dict()

class Graph:

    def __init__(self, fn, *args, **kwargs):

        # function being compiled into a graph
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._arg_input_idxs =\
            ivy.Container(args, include_iters=True).map(lambda x, kc: id(x) if ivy.is_array(x) else None)
        self._kwarg_input_idxs = ivy.Container(kwargs, include_iters=True).map(
            lambda x, kc: id(x) if ivy.is_array(x) else None)
        self._output_idxs = ivy.Container()

        # graph storage
        self._param_dict = dict()
        self._fn_dict = dict()

    # Compute output idxs #
    # --------------------#

    def foward(self):
        ret_raw = self._fn(*self._args, **self._kwargs)
        ret = ret_raw if isinstance(ret_raw, tuple) else (ret_raw,)
        self._output_idxs = ivy.Container(ret, include_iters=True).map(lambda x, kc: id(x) if ivy.is_array(x) else None)

    # Setters #
    # --------#

    def add_param(self, idx, param):
        self._param_dict[idx] = param

    def add_fn(self, idx, fn):
        self._fn_dict[idx] = fn

    # Getters #
    # --------#

    def get_param(self, idx):
        param = self._param_dict[idx]
        if ivy.exists(param):
            return param
        fn = self._fn_dict[idx]
        input_idxs = fn.input_idxs
        inputs = {idx: self.get_param(idx) for idx in input_idxs}
        ret_dict = fn(inputs)
        for idx, param in ret_dict:
            self.add_param(idx, param)
        return ret_dict

    # Function creation #
    # ------------------#

    def _call(self, *args, **kwargs):
        # ToDo: set the input idxs
        # ToDo: query the output idxs
        return self._fn(*args, **kwargs)

    def to_function(self):
        return self._call

    # Clearing #
    # ---------#

    def clear(self):
        self._param_dict.clear()
        self._fn_dict.clear()


graph = None


# Methods #

def _wrap_method_for_compiling(fn):

    if hasattr(fn, '__name__') and\
            (fn.__name__[0] == '_' or fn.__name__ in NON_WRAPPED_METHODS + NON_ARRAY_RET_METHODS):
        return fn

    if hasattr(fn, 'wrapped_for_compiling') and fn.wrapped_for_compiling:
        return fn

    def _method_wrapped(*args, **kwargs):
        args_cont = ivy.Container(args, include_iters=True)
        arg_input_idxs = [id(v) for _, v in args_cont.to_iterator() if ivy.is_array(v)]
        kwargs_cont = ivy.Container(kwargs, include_iters=True)
        kwarg_input_idxs = [id(v) for _, v in kwargs_cont.to_iterator() if ivy.is_array(v)]
        input_idxs = arg_input_idxs + kwarg_input_idxs
        [graph.add_param(idx, None) for idx in input_idxs]
        ret_raw = fn(*args, **kwargs)
        # ToDo: implement this below
        new_fn = lambda *a, **kw: fn(*args, **kwargs)
        new_fn.input_idxs = input_idxs
        ret = ret_raw if isinstance(ret_raw, tuple) else (ret_raw,)
        [graph.add_param(id(v), None)
         for _, v in ivy.Container(ret, include_iters=True).to_iterator() if ivy.is_array(v)]
        return ret_raw

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
    return _wrap_or_unwrap_methods(_wrap_method_for_compiling)


def _unwrap_methods_from_compiling():
    return _wrap_or_unwrap_methods(_unwrap_method_from_compiling)


def compile_ivy(fn, *args, **kwargs):
    global graph
    graph = Graph(fn, *args, **kwargs)
    _wrap_methods_for_compiling()
    graph.foward()
    _unwrap_methods_from_compiling()
    local_graph = copy.deepcopy(graph)
    fn = local_graph.to_function()
    graph.clear()
    return fn
