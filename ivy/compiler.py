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
        self._args = args
        self._kwargs = kwargs
        self._arg_input_idxs =\
            ivy.Container(args, types_to_iteratively_nest=(list, tuple)).map(
                lambda x, kc: id(x._data) if isinstance(x, ivy.Array) else None)
        self._kwarg_input_idxs = ivy.Container(kwargs, types_to_iteratively_nest=(list, tuple)).map(
            lambda x, kc: id(x._data) if isinstance(x, ivy.Array) else None)
        self._output_idxs = ivy.Container()

        # graph storage
        self._param_dict = dict()
        self._fn_dict = dict()

    # Compute output idxs #
    # --------------------#

    # noinspection PyProtectedMember
    def foward(self):
        ret = self._fn(*self._args, **self._kwargs)
        if not isinstance(ret, tuple):
            ret = (ret,)
        self._output_idxs = ivy.Container(ret, types_to_iteratively_nest=(list, tuple)).map(
            lambda x, kc: id(x._data) if ivy.is_array(x) else None)

    # Setters #
    # --------#

    def set_param(self, idx, param):
        self._param_dict[idx] = param

    def set_fn(self, idx, fn):
        self._fn_dict[idx] = fn

    # Getters #
    # --------#

    def get_param(self, idx):
        # ToDo: make this more efficient, this is all called at runtime
        param = self._param_dict[idx]
        if ivy.exists(param):
            return param
        fn = self._fn_dict[idx]
        arg_param_cont = fn.arg_idxs_cont.map(lambda idx_, kc: self.get_param(idx_))
        kwarg_param_cont = fn.kwarg_idxs_cont.map(lambda idx_, kc: self.get_param(idx_))
        ret_cont = fn(arg_param_cont, kwarg_param_cont)
        ivy.Container.multi_map(lambda xs, kc: self.set_param(xs[0], xs[1]), [fn.ret_idxs_cont, ret_cont])
        return self._param_dict[idx]

    # Function creation #
    # ------------------#

    def _call(self, *args, **kwargs):
        # ToDo: make this more efficient, this is all called at runtime
        args_cont = ivy.Container(args, types_to_iteratively_nest=(list, tuple))
        ivy.Container.multi_map(lambda xs, kc: self.set_param(xs[0], xs[1]) if ivy.exists(xs[0]) else None,
                                [self._arg_input_idxs, args_cont])
        kwargs_cont = ivy.Container(kwargs, types_to_iteratively_nest=(list, tuple))
        ivy.Container.multi_map(lambda xs, kc: self.set_param(xs[0], xs[1]) if ivy.exists(xs[0]) else None,
                                [self._kwarg_input_idxs, kwargs_cont])
        output_cont = self._output_idxs.map(lambda x, kc: self.get_param(x) if ivy.exists(x) else None)
        ret = output_cont.to_raw()
        if len(ret) == 1:
            return ret[0]
        return ret

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

        # get array idxs for positional args
        args_cont = ivy.Container(args, types_to_iteratively_nest=(list, tuple))
        arg_idxs_cont = args_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        arg_input_idxs = [v for _, v in arg_idxs_cont.to_iterator()]

        # get array idxs for key-word args
        kwargs_cont = ivy.Container(kwargs, types_to_iteratively_nest=(list, tuple))
        kwarg_idxs_cont = kwargs_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        kwarg_input_idxs = [v for _, v in kwarg_idxs_cont.to_iterator()]

        # initialize empty keys for these input parameters in the graph
        [graph.set_param(idx, None) for idx in arg_input_idxs + kwarg_input_idxs]

        # compute the return
        ret_raw = fn(*args, **kwargs)
        ret = ret_raw if isinstance(ret_raw, tuple) else (ret_raw,)

        # get array idxs for return
        ret_cont = ivy.Container(ret, types_to_iteratively_nest=(list, tuple))
        ret_idxs_cont = ret_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()

        # wrap the function
        def new_fn(arg_params_cont, kwarg_params_cont):
            # ToDo: make this more efficient, this is all performed at runtime
            a_cont = args_cont.set_at_keys(arg_params_cont)
            kw_cont = kwargs_cont.set_at_keys(kwarg_params_cont)
            ret_ = fn(*a_cont.to_raw(), **kw_cont.to_raw())
            if not isinstance(ret_, tuple):
                ret_ = (ret_,)
            return ivy.Container(ret_, types_to_iteratively_nest=(list, tuple))

        # add function attributes which inform about the input idxs
        new_fn.args = args
        new_fn.kwargs = kwargs
        new_fn.ret = ret
        new_fn.arg_idxs_cont = arg_idxs_cont
        new_fn.kwarg_idxs_cont = kwarg_idxs_cont
        new_fn.ret_idxs_cont = ret_idxs_cont

        # initialize empty keys for these output parameters in the graph
        [graph.set_param(id(v), None)
         for _, v in ivy.Container(ret, types_to_iteratively_nest=(list, tuple)).to_iterator() if ivy.is_array(v)]

        # initialize empty keys for this function in the graph
        [graph.set_fn(id(v), new_fn)
         for _, v in ivy.Container(ret, types_to_iteratively_nest=(list, tuple)).to_iterator() if ivy.is_array(v)]

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
