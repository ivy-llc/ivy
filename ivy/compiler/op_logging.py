# global
import ivy
import time
import weakref
import inspect
import importlib

# local
from ivy.compiler import globals as glob
# noinspection PyProtectedMember
from ivy.compiler.helpers import _get_unique_id, _get_shape, _get_fn_signature, _clone_param, _delete_dependent_param,\
    _args_n_kwarg_reprs_from_keys_n_args_n_kwargs, _output_reprs_from_output
# noinspection PyProtectedMember
from ivy.wrapper import _wrap_or_unwrap_methods, NON_WRAPPED_METHODS, ARRAYLESS_RET_METHODS


def _wrap_method_for_op_logging(fn, graph, limit_attributes=True, stateful_classes=None):

    stateful_classes = tuple(ivy.default(stateful_classes, tuple()))

    if (inspect.isclass(fn) or (hasattr(fn, '__name__') and
                                ((fn.__name__[0] == '_' and fn.__name__ not in glob.ARRAY_BUILTINS) or
                                 fn.__name__ in NON_WRAPPED_METHODS + ARRAYLESS_RET_METHODS)) or
            (hasattr(fn, 'wrapped_for_compiling') and fn.wrapped_for_compiling)):
        return fn

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def _method_wrapped(*args, **kwargs):

        # if cloning a param currently, return directly via the original function
        if glob.wrapping_paused:
            return fn(*args, **kwargs)

        if glob.wrapped_stack:
            # return if the wrapping is already happening on a higher level, and it's not a built-in which legitimately
            # might need to be nested, unless it's a built-in recursion loop (ie for __getattribute__) in which case return
            if (glob.wrapped_stack[-1].__name__[0:2] != '__' or
                    (glob.wrapped_stack[-1].__name__ == fn.__name__ and args == args and kwargs == kwargs)):
                return fn(*args, **kwargs)

            # return if the current method is a (possibly reversed) built-in operator, and the last entry of the wrapped
            # stack is a version of that same operator
            elif fn.__name__.replace('r', '').replace('_', '') in\
                    glob.wrapped_stack[-1].__name__.replace('r', '').replace('_', ''):
                return fn(*args, **kwargs)

        # attributes to ignore
        if fn.__name__ in ['__getattr__', '__setattr__', '__getattribute__']:
            att_name = args[1]
            # return if the attribute being retrieved is another built-in method
            if att_name[0:2] == '__':
                return fn(*args, **kwargs)
            # if the attribute is not recognized as one which can form part of the graph, then return
            if limit_attributes and att_name not in glob.GRAPH_ATTRIBUTES[ivy.current_framework_str()]:
                return fn(*args, **kwargs)

        # otherwise, set wrapping as true
        glob.wrapped_stack.append(fn)

        # immutable tuple to mutable list
        args = list(ivy.nested_map(args, lambda a: a, to_mutable=True))
        kwargs = ivy.nested_map(kwargs, lambda v: v, to_mutable=True)

        # get array idxs for positional args
        # ToDo: work out why adding check_nests=True causes errors.
        #  This is needed in order to support stateful updates of ivy.Containers.
        # arg_tracked_idxs = ivy.nested_indices_where(
        #     args, lambda x: ivy.is_array(x) or isinstance(x, stateful_classes), check_nests=True)
        arg_tracked_idxs = ivy.nested_indices_where(
            args, lambda x_: ivy.is_array(x_) or isinstance(x_, stateful_classes))
        arg_vals = list(ivy.multi_index_nest(args, arg_tracked_idxs))
        arg_param_ids = [_get_unique_id(x) for x in arg_vals]
        for x in arg_vals:
            glob.raw_pids_to_weakrefs[id(x)] = weakref.ref(x)
        arg_param_types = [x.__class__ for x in arg_vals]
        arg_param_var_flags = [ivy.is_variable(x, exclusive=True) for x in arg_vals]
        arg_param_shapes = [_get_shape(x) for x in arg_vals]

        # get array idxs for key-word args
        # ToDo: work out why adding check_nests=True causes errors.
        #  This is needed in order to support stateful updates of ivy.Containers.
        # kwarg_tracked_idxs = ivy.nested_indices_where(
        #     kwargs, lambda x: ivy.is_array(x) or isinstance(x, stateful_classes), check_nests=True)
        kwarg_tracked_idxs = ivy.nested_indices_where(
            kwargs, lambda x_: ivy.is_array(x_) or isinstance(x_, stateful_classes))
        kwarg_vals = list(ivy.multi_index_nest(kwargs, kwarg_tracked_idxs))
        kwarg_param_ids = [_get_unique_id(x) for x in kwarg_vals]
        for x in kwarg_vals:
            glob.raw_pids_to_weakrefs[id(x)] = weakref.ref(x)
        kwarg_param_types = [x.__class__ for x in kwarg_vals]
        kwarg_param_var_flags = [ivy.is_variable(x, exclusive=True) for x in kwarg_vals]
        kwarg_param_shapes = [_get_shape(x) for x in kwarg_vals]

        # set the backend function
        backend_fn = fn

        # compute the return
        glob.wrapping_paused = True
        ret_raw = fn(*args, **kwargs)
        glob.wrapping_paused = False

        # provide return value for __setattr__
        if fn.__name__ == '__setattr__':
            ret_raw = args[0]

            # update the setattr method to return the object after attribute setting
            def backend_fn(__obj, __name, __value):
                setattr(__obj, __name, __value)
                return __obj

        # remove parameters from args and kwargs
        ivy.map_nest_at_indices(args, arg_tracked_idxs, lambda x_: _delete_dependent_param(x_, graph))
        ivy.map_nest_at_indices(kwargs, kwarg_tracked_idxs, lambda x_: _delete_dependent_param(x_, graph))

        # covert return to list
        ret_listified = False
        if isinstance(ret_raw, tuple):
            ret = list(ret_raw)
        else:
            ret = [ret_raw]
            ret_listified = True

        # get array idxs for return
        # ToDo: work out why adding check_nests=True causes errors.
        #  This is needed in order to support stateful updates of ivy.Containers.
        # output_tracked_idxs = ivy.nested_indices_where(
        #     ret, lambda x: ivy.is_array(x) or isinstance(x, stateful_classes), check_nests=True)
        output_tracked_idxs = ivy.nested_indices_where(
            ret, lambda x_: ivy.is_array(x_) or isinstance(x_, stateful_classes))
        output_vals = list(ivy.multi_index_nest(ret, output_tracked_idxs))
        output_param_ids = [_get_unique_id(x) for x in output_vals]
        output_param_types = [x.__class__ for x in output_vals]
        output_param_var_flags = [ivy.is_variable(x, exclusive=True) for x in output_vals]
        output_param_shapes = [_get_shape(x) for x in output_vals]

        # clone the param when getting an attribute, to preserve uniqueness in the graph
        if fn.__name__ in ['__getattr__', '__getattribute__']:
            # update the param_id for each param in the retreived attribute in the graph
            ivy.map_nest_at_indices(ret, output_tracked_idxs, lambda x: _clone_param(x, graph))

        # find all duplicate param ids from the input in the return
        duplicates = list()
        for i, ret_pid in enumerate(output_param_ids):
            if ret_pid in arg_param_ids + kwarg_param_ids:
                duplicates.append(i)

        # clone all repeated return parameters to give unique parameter ids in the graph
        duplicate_tracked_idxs = [output_tracked_idxs[i] for i in duplicates]
        ivy.map_nest_at_indices(ret, duplicate_tracked_idxs, lambda x: _clone_param(x, graph))

        # get return param ids after cloning
        output_vals = list(ivy.multi_index_nest(ret, output_tracked_idxs))
        output_param_ids = [_get_unique_id(x) for x in output_vals]
        for x in output_vals:
            glob.raw_pids_to_weakrefs[id(x)] = weakref.ref(x)

        # maybe add to set of dependent_pids
        if fn.__name__ in glob.GENERATOR_METHODS and graph.include_generators:
            [glob.dependent_pids.add(pid) for pid in output_param_ids]
        else:
            for pid in arg_param_ids + kwarg_param_ids:
                if pid in glob.dependent_pids:
                    [glob.dependent_pids.add(pid) for pid in output_param_ids]
                    break

        # wrap the function
        def new_fn(arg_array_vals, kwarg_array_vals):
            # ToDo: make this as efficient as possible; this is performed at runtime
            args_writeable = ivy.copy_nest(args)
            kwargs_writeable = ivy.copy_nest(kwargs)
            ivy.set_nest_at_indices(args_writeable, arg_tracked_idxs, arg_array_vals)
            ivy.set_nest_at_indices(kwargs_writeable, kwarg_tracked_idxs, kwarg_array_vals)
            return backend_fn(*args_writeable, **kwargs_writeable)

        # wrap the function with timing
        def new_fn_w_timing(arg_array_vals, kwarg_array_vals):
            start = time.perf_counter()
            args_writeable = ivy.copy_nest(args)
            kwargs_writeable = ivy.copy_nest(kwargs)
            graph.update_inference_times('2_0_arg_n_kwarg_copying', time.perf_counter() - start)
            start = time.perf_counter()
            ivy.set_nest_at_indices(args_writeable, arg_tracked_idxs, arg_array_vals)
            ivy.set_nest_at_indices(kwargs_writeable, kwarg_tracked_idxs, kwarg_array_vals)
            graph.update_inference_times('2_1_arg_n_kwarg_writing', time.perf_counter() - start)
            start = time.perf_counter()
            ret_ = backend_fn(*args_writeable, **kwargs_writeable)
            graph.update_inference_times('2_2_backend_fn', time.perf_counter() - start)
            return ret_

        # add function attributes which inform about the arguments and returns

        glob.wrapping_paused = True

        if glob.time_inference:
            new_fn = new_fn_w_timing

        new_fn.arg_reprs = str(args)
        new_fn.arg_tracked_idxs = arg_tracked_idxs
        new_fn.arg_param_ids = arg_param_ids
        new_fn.arg_param_types = arg_param_types
        new_fn.arg_param_var_flags = arg_param_var_flags
        new_fn.arg_param_shapes = arg_param_shapes

        new_fn.kwarg_reprs = str(kwargs)
        new_fn.kwarg_tracked_idxs = kwarg_tracked_idxs
        new_fn.kwarg_param_ids = kwarg_param_ids
        new_fn.kwarg_param_types = kwarg_param_types
        new_fn.kwarg_param_var_flags = kwarg_param_var_flags
        new_fn.kwarg_param_shapes = kwarg_param_shapes

        try:
            sig = inspect.signature(fn)
            sig_keys = list(sig.parameters.keys())
        except ValueError:
            sig_keys = list()
        new_fn.arg_n_kwarg_reprs = _args_n_kwarg_reprs_from_keys_n_args_n_kwargs(sig_keys, args, kwargs)

        new_fn.output_tracked_idxs = output_tracked_idxs
        new_fn.output_param_ids = output_param_ids
        new_fn.output_param_types = output_param_types
        new_fn.output_param_var_flags = output_param_var_flags
        new_fn.output_param_shapes = output_param_shapes

        new_fn.output_reprs = _output_reprs_from_output(ret)

        new_fn.timestamp = time.perf_counter()

        new_fn.signature = _get_fn_signature(backend_fn)
        new_fn.terminal = True
        new_fn.is_constant = len(arg_param_ids + kwarg_param_ids) == 0 and \
                             (not graph.include_generators or
                              fn.__name__ not in glob.GENERATOR_METHODS[ivy.current_framework_str()])

        glob.wrapping_paused = False

        fns_in = [graph._pid_to_functions_dict[pid]
                  for pid in arg_param_ids + kwarg_param_ids if pid in graph._pid_to_functions_dict]
        for fn_in in fns_in:
            fn_in.terminal = False
            if new_fn not in fn_in.fns_out:
                fn_in.fns_out.append(new_fn)

        new_fn.fns_in = fns_in
        new_fn.fns_out = list()

        new_fn.__repr__ = lambda: new_fn.__name__

        if hasattr(fn, '__name__'):
            new_fn.__name__ = fn.__name__

        # add to graph if compiling
        if glob.op_logging:

            # add this function to the graph for each output pid
            for pid in output_param_ids:
                if pid in graph._pid_to_functions_dict:
                    graph._register_output(ret)
                    glob.op_logging = False
                    _unwrap_methods_from_op_logging(list(graph._stateful_classes))
                    # noinspection PyBroadException
                    try:
                        graph.show(save_to_disk=True, output_connected_only=False)
                    except Exception:
                        pass
                    raise Exception(
                        '\n\ntried to add {} to graph._functions_dict, but function {} with the same output pid {} '
                        'already exists!'.format(
                            new_fn.__name__ + '(*{}, **{})'.format(new_fn.arg_reprs, new_fn.kwarg_reprs),
                            graph._pid_to_functions_dict[pid].__name__ + '(*{}, **{})'.format(
                                graph._pid_to_functions_dict[pid].arg_reprs,
                                graph._pid_to_functions_dict[pid].kwarg_reprs), pid))
                graph.add_fn_to_dict(pid, new_fn)

        # unset wrapping as true
        glob.wrapped_stack.pop(-1)

        # return the function output
        return ret[0] if ret_listified else tuple(ret)

    if hasattr(fn, '__name__'):
        _method_wrapped.__name__ = fn.__name__
    _method_wrapped.wrapped_for_compiling = True
    _method_wrapped.inner_fn = fn
    return _method_wrapped


def _unwrap_method_from_op_logging(method_wrapped):
    if not hasattr(method_wrapped, 'wrapped_for_compiling') or not method_wrapped.wrapped_for_compiling:
        return method_wrapped
    return method_wrapped.inner_fn


def _wrap_methods_for_op_logging(graph, stateful_classes=None):

    # wrap backend framework
    classes_to_wrap = [getattr(importlib.import_module(ctw[0]), ctw[1])
                       for ctw in glob.CLASSES_TO_WRAP[ivy.current_framework_str()]]
    _wrap_or_unwrap_methods(
        lambda fn: _wrap_method_for_op_logging(fn, graph), classes_to_wrap=classes_to_wrap, native=True)

    # wrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, '__setattr__') and (hasattr(cls, '__getattr__') or hasattr(cls, '__getattribute__'))
        cls.__setattr__ = _wrap_method_for_op_logging(
            cls.__setattr__, graph, limit_attributes=False, stateful_classes=stateful_classes)
        if hasattr(cls, '__getattr__'):
            cls.__getattr__ = _wrap_method_for_op_logging(
                cls.__getattr__, graph, limit_attributes=False, stateful_classes=stateful_classes)
        if hasattr(cls, '__getattribute__'):
            cls.__getattribute__ = _wrap_method_for_op_logging(
                cls.__getattribute__, graph, limit_attributes=False, stateful_classes=stateful_classes)


def _unwrap_methods_from_op_logging(stateful_classes=None):

    # unwrap backend framework
    classes_to_wrap = [getattr(importlib.import_module(ctw[0]), ctw[1])
                       for ctw in glob.CLASSES_TO_WRAP[ivy.current_framework_str()]] + stateful_classes
    _wrap_or_unwrap_methods(
        lambda fn: _unwrap_method_from_op_logging(fn), classes_to_wrap=classes_to_wrap, native=True)

    # unwrap stateful classes
    stateful_classes = ivy.default(stateful_classes, [])
    for cls in stateful_classes:
        assert hasattr(cls, '__setattr__') and (hasattr(cls, '__getattr__') or hasattr(cls, '__getattribute__'))
        cls.__setattr__ = _unwrap_method_from_op_logging(cls.__setattr__)
        if hasattr(cls, '__getattr__'):
            cls.__getattr__ = _unwrap_method_from_op_logging(cls.__getattr__)
        if hasattr(cls, '__getattribute__'):
            cls.__getattribute__ = _unwrap_method_from_op_logging(cls.__getattribute__)
