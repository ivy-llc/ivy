import ast
import inspect
from ivy_tests.test_ivy.helpers.function_testing import _is_frontend_array
from ivy_tests import config

# from ivy_tests.test_ivy.helpers.structs import FrontendMethodData
import sys
import jsonpickle
import importlib
from ivy_tests.test_ivy.helpers.testing_helpers import (
    _import_fn,
    _get_supported_devices_dtypes,
    _import_method,
    _get_method_supported_devices_dtypes,
)

# import paddle_bfloat


def _lstrip_lines(source: str) -> str:
    source = source.lstrip().split("\n")

    # If the first line is a decorator
    if source[0][0] == "@":
        # If the second line is a function definition
        if source[1].lstrip()[0:3] == "def":
            # Work out how many whitespace chars to remove
            num_chars_to_remove = source[1].find("d")

            # The first string needs no changes
            for i in range(1, len(source)):
                source[i] = source[i][num_chars_to_remove:]

    source = "\n".join(source)
    return source


def _get_functions_from_string(func_names, module):
    ret = set()
    # We only care about the functions in the ivy or the same module
    for func_name in func_names.keys():
        if hasattr(ivy, func_name) and callable(getattr(ivy, func_name, None)):
            ret.add(getattr(ivy, func_name))
        elif hasattr(module, func_name) and callable(getattr(ivy, func_name, None)):
            ret.add(getattr(module, func_name))
        elif callable(getattr(func_names[func_name], func_name, None)):
            ret.add(getattr(func_names[func_name], func_name))
    return ret


def _get_function_list(func):
    tree = ast.parse(_lstrip_lines(inspect.getsource(func)))
    names = {}
    # Extract all the call names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            nodef = node.func
            if isinstance(nodef, ast.Name):
                names[nodef.id] = getattr(
                    func,
                    "__self__",
                    getattr(
                        importlib.import_module(func.__module__),
                        func.__qualname__.split(".")[0],
                        None,
                    ),
                )
            elif isinstance(nodef, ast.Attribute):
                if (
                    hasattr(nodef, "value")
                    and hasattr(nodef.value, "id")
                    and nodef.value.id not in ["ivy", "self"]
                ):
                    continue
                names[nodef.attr] = getattr(
                    func,
                    "__self__",
                    getattr(
                        importlib.import_module(func.__module__),
                        func.__qualname__.split(".")[0],
                        None,
                    ),
                )

    return names


def get_ret_and_flattened_np_array(fn, *args, **kwargs):
    """
    Runs func with args and kwargs, and returns the result along with its flattened
    version.
    """
    ret = fn(*args, **kwargs)

    def map_fn(x):
        if _is_frontend_array(x):
            return x.ivy_array
        if isinstance(x, ivy.functional.frontends.numpy.ndarray):
            return x.ivy_array
        return x

    ret = ivy.nested_map(ret, map_fn, include_derived={tuple: True})
    return ret, flatten_and_to_np(ret=ret)


def gradient_unsupported_dtypes(*, fn):
    visited = set()
    to_visit = [fn]
    out, res = {}, {}
    while to_visit:
        fn = to_visit.pop()
        if fn in visited:
            continue
        visited.add(fn)
        unsupported_grads = (
            fn.unsupported_gradients if hasattr(fn, "unsupported_gradients") else {}
        )
        for k, v in unsupported_grads.items():
            if k not in out:
                out[k] = []
            out[k].extend(v)
        # skip if it's not a function
        if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
            continue
        fl = _get_function_list(fn)
        res = _get_functions_from_string(fl, __import__(fn.__module__))
        to_visit.extend(res)
    return out


def flatten_and_to_np(*, ret):
    # flatten the return
    ret_flat = flatten(ret=ret)
    return [ivy.to_numpy(x) for x in ret_flat]


def flatten(*, ret):
    """Returns a flattened numpy version of the arrays in ret."""
    if not isinstance(ret, tuple):
        ret = (ret,)
    ret_idxs = ivy.nested_argwhere(ret, ivy.is_ivy_array)
    # no ivy array in the returned values, which means it returned scalar
    if len(ret_idxs) == 0:
        ret_idxs = ivy.nested_argwhere(ret, ivy.isscalar)
        ret_flat = ivy.multi_index_nest(ret, ret_idxs)
        temp = []
        for x in ret_flat:
            temp.append(ivy.asarray(x, dtype=ivy.Dtype(str(numpy.asarray(x).dtype))))
        ret_flat = temp
    else:
        ret_flat = ivy.multi_index_nest(ret, ret_idxs)
    return ret_flat


def as_cont(*, x):
    """Returns x as an Ivy Container, containing x at all its leaves."""
    return ivy.Container({"a": x, "b": {"c": x, "d": x}})


def create_args_kwargs(
    *,
    args_np,
    arg_np_vals,
    args_idxs,
    kwargs_np,
    kwarg_np_vals,
    kwargs_idxs,
    input_dtypes,
    test_flags,
):
    """Creates arguments and keyword-arguments for the function to test.

    Parameters
    ----------
    args_np
        A dictionary of arguments in Numpy.
    kwargs_np
        A dictionary of keyword-arguments in Numpy.
    input_dtypes
        data-types of the input arguments and keyword-arguments.

    Returns
    -------
    Arguments, Keyword-arguments, number of arguments, and indexes on arguments and
    keyword-arguments.
    """

    def _apply_flags(args_to_iterate):
        ret = []
        for i, entry in enumerate(args_to_iterate):
            x = ivy.array(entry, dtype=input_dtypes[i])
            if test_flags.as_variable[i]:
                x = _variable(x)
            if test_flags.native_arrays[i]:
                x = ivy.to_native(x)
            if test_flags.container[i]:
                x = as_cont(x=x)
            ret.append(x)
        return ret

    # create args
    args = ivy.copy_nest(args_np, to_mutable=False)
    ivy.set_nest_at_indices(args, args_idxs, _apply_flags(arg_np_vals))

    # create kwargs
    kwargs = ivy.copy_nest(kwargs_np, to_mutable=False)
    ivy.set_nest_at_indices(kwargs, kwargs_idxs, _apply_flags(kwarg_np_vals))
    return args, kwargs, len(arg_np_vals), args_idxs, kwargs_idxs


def _get_fn_dtypes(framework, fn_tree, type, device=None, kind="valid"):
    if type == "1":
        callable_fn, fn_name, fn_mod = _import_fn(fn_tree)
        supported_device_dtypes = _get_supported_devices_dtypes(fn_name, fn_mod)
        return supported_device_dtypes[framework][device][kind]
    else:
        method_tree = fn_tree
        callable_method, method_name, _, class_name, method_mod = _import_method(
            method_tree
        )
        supported_device_dtypes = _get_method_supported_devices_dtypes(
            method_name, method_mod, class_name
        )
        return supported_device_dtypes[framework][device][kind]


def _get_type_dict(framework, fn_tree, type, device=None, kind="valid"):
    if kind == "valid":
        return framework.valid_dtypes
    elif kind == "numeric":
        return framework.valid_numeric_dtypes
    elif kind == "integer":
        return framework.valid_int_dtypes
    elif kind == "float":
        return framework.valid_float_dtypes
    elif kind == "unsigned":
        return framework.valid_int_dtypes
    elif kind == "signed_integer":
        return tuple(
            set(framework.valid_int_dtypes).difference(framework.valid_uint_dtypes)
        )
    elif kind == "complex":
        return framework.valid_complex_dtypes
    elif kind == "real_and_complex":
        return tuple(
            set(framework.valid_numeric_dtypes).union(framework.valid_complex_dtypes)
        )
    elif kind == "float_and_complex":
        return tuple(
            set(framework.valid_float_dtypes).union(framework.valid_complex_dtypes)
        )
    elif kind == "bool":
        return tuple(
            set(framework.valid_dtypes).difference(framework.valid_numeric_dtypes)
        )
    else:
        raise RuntimeError("{} is an unknown kind!".format(kind))


def dtype_handler(framework, type):
    global temp_store
    temp_store = []
    z = input()
    retrieval_fn = globals()[z]
    z = input()
    kind = z
    z = input()
    device = z
    z = input()
    fn_tree = z

    if retrieval_fn.__name__ == "_get_type_dict":
        framework = importlib.import_module("ivy.functional.backends." + framework)

    dtypes = retrieval_fn(framework, fn_tree, type, device, kind)

    dtypes = jsonpickle.dumps(dtypes)
    print(dtypes)


def make_json_pickable(s):
    s = s.replace("builtins.bfloat16", "ivy.bfloat16")
    # s = s.replace("jax._src.device_array.reconstruct_device_array", "jax.numpy.array")
    return s


temp_store = []  # for ins_gt


def check_unsupported_dtype(*, fn, input_dtypes, all_as_kwargs_np):
    """Checks whether a function does not support the input data types or the output
    data type.

    Parameters
    ----------
    fn
        The function to check.
    input_dtypes
        data-types of the input arguments and keyword-arguments.
    all_as_kwargs_np
        All arguments in Numpy Format, to check for the presence of dtype argument.

    Returns
    -------
    True if the function does not support the given input or output data types, False
    otherwise.
    """
    test_unsupported = False
    unsupported_dtypes_fn = ivy.function_unsupported_dtypes(fn)
    supported_dtypes_fn = ivy.function_supported_dtypes(fn)
    if unsupported_dtypes_fn:
        for d in input_dtypes:
            if d in unsupported_dtypes_fn:
                test_unsupported = True
                break
        if (
            "dtype" in all_as_kwargs_np
            and all_as_kwargs_np["dtype"] in unsupported_dtypes_fn
        ):
            test_unsupported = True
    if supported_dtypes_fn and not test_unsupported:
        for d in input_dtypes:
            if d not in supported_dtypes_fn:
                test_unsupported = True
                break
        if (
            "dtype" in all_as_kwargs_np
            and all_as_kwargs_np["dtype"] not in supported_dtypes_fn
        ):
            test_unsupported = True
    return test_unsupported


def gradient_test():
    def grad_fn(all_args):
        args, kwargs, i = all_args
        ret = (
            ivy.__dict__[fn](*args, **kwargs)
            if isinstance(fn, str)
            else fn[i](*args, **kwargs)
        )
        return ivy.nested_map(ret, ivy.mean, include_derived=True)

    args_np = jsonpickle.loads(make_json_pickable(input()))
    arg_np_vals = jsonpickle.loads(make_json_pickable(input()))
    args_idxs = jsonpickle.loads(make_json_pickable(input()))
    kwargs_np = jsonpickle.loads(make_json_pickable(input()))
    kwargs_idxs = jsonpickle.loads(make_json_pickable(input()))
    kwarg_np_vals = jsonpickle.loads(make_json_pickable(input()))
    input_dtypes = jsonpickle.loads(input())
    test_flags = jsonpickle.loads(make_json_pickable(input()))
    fn = jsonpickle.loads(make_json_pickable(input()))
    all_as_kwargs_np = jsonpickle.loads(make_json_pickable(input()))
    grad_fnn = jsonpickle.loads(make_json_pickable(input()))  # noqa: F841
    ret_grad_idxs = jsonpickle.loads(make_json_pickable(input()))
    global temp_store
    if not temp_store and isinstance(fn, list):
        raise Exception("trouble for ins_gt")
    elif isinstance(fn, list):
        fn[1] = temp_store[-1][0].__getattribute__(temp_store[-1][1])

    test_unsupported = check_unsupported_dtype(
        fn=ivy.__dict__[fn] if isinstance(fn, str) else fn[1],
        input_dtypes=input_dtypes,
        all_as_kwargs_np=all_as_kwargs_np,
    )
    if test_unsupported:
        print(jsonpickle.dumps(None))
        return
    args, kwargs, _, args_idxs, kwargs_idxs = create_args_kwargs(
        args_np=args_np,
        arg_np_vals=arg_np_vals,
        args_idxs=args_idxs,
        kwargs_np=kwargs_np,
        kwarg_np_vals=kwarg_np_vals,
        kwargs_idxs=kwargs_idxs,
        input_dtypes=input_dtypes,
        test_flags=test_flags,
    )
    _, grads_from_gt = ivy.execute_with_gradients(
        grad_fn,
        [args, kwargs, 1],
        xs_grad_idxs=xs_grad_idxs,  # noqa: F821
        ret_grad_idxs=ret_grad_idxs,
    )
    grads_np_from_gt_flat = flatten_and_to_np(ret=grads_from_gt)
    print(jsonpickle.dumps(grads_np_from_gt_flat))


def method_test():
    args_np_constructor = jsonpickle.loads(make_json_pickable(input()))
    con_arg_np_vals = jsonpickle.loads(make_json_pickable(input()))
    con_args_idxs = jsonpickle.loads(make_json_pickable(input()))
    kwargs_np_constructor = jsonpickle.loads(make_json_pickable(input()))
    con_kwarg_np_vals = jsonpickle.loads(make_json_pickable(input()))
    con_kwargs_idxs = jsonpickle.loads(make_json_pickable(input()))
    init_input_dtypes = jsonpickle.loads(make_json_pickable(input()))
    init_flags = jsonpickle.loads(make_json_pickable(input()))
    args_gt_constructor, kwargs_gt_constructor, _, _, _ = create_args_kwargs(
        args_np=args_np_constructor,
        arg_np_vals=con_arg_np_vals,
        args_idxs=con_args_idxs,
        kwargs_np=kwargs_np_constructor,
        kwargs_idxs=con_kwargs_idxs,
        kwarg_np_vals=con_kwarg_np_vals,
        input_dtypes=init_input_dtypes,
        test_flags=init_flags,
    )

    args_np = jsonpickle.loads(make_json_pickable(input()))
    arg_np_vals = jsonpickle.loads(make_json_pickable(input()))
    args_idxs = jsonpickle.loads(make_json_pickable(input()))
    kwargs_np = jsonpickle.loads(make_json_pickable(input()))
    kwargs_idxs = jsonpickle.loads(make_json_pickable(input()))
    kwarg_np_vals = jsonpickle.loads(make_json_pickable(input()))
    input_dtypes = jsonpickle.loads(make_json_pickable(input()))
    method_flags = jsonpickle.loads(make_json_pickable(input()))
    class_name = jsonpickle.loads(make_json_pickable(input()))
    method_name = jsonpickle.loads(make_json_pickable(input()))
    method_input_dtypes = jsonpickle.loads(make_json_pickable(input()))
    v_np = jsonpickle.loads(make_json_pickable(input()))
    args_gt_method, kwargs_gt_method, _, _, _ = create_args_kwargs(
        args_np=args_np,
        arg_np_vals=arg_np_vals,
        args_idxs=args_idxs,
        kwargs_np=kwargs_np,
        kwargs_idxs=kwargs_idxs,
        kwarg_np_vals=kwarg_np_vals,
        input_dtypes=input_dtypes,
        test_flags=method_flags,
    )
    ins_gt = ivy.__dict__[class_name](*args_gt_constructor, **kwargs_gt_constructor)
    temp_store.append([ins_gt, method_name])
    # ToDo : remove this when the handle_method can properly compute unsupported dtypes
    if any(
        dtype in ivy.function_unsupported_dtypes(ins_gt.__getattribute__(method_name))
        for dtype in method_input_dtypes
    ):
        return
    if isinstance(ins_gt, ivy.Module):
        v_gt = v_np.cont_map(
            lambda x, kc: ivy.asarray(x) if isinstance(x, numpy.ndarray) else x
        )
        kwargs_gt_method = dict(**kwargs_gt_method, v=v_gt)
    ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
        ins_gt.__getattribute__(method_name), *args_gt_method, **kwargs_gt_method
    )

    fw_list2 = gradient_unsupported_dtypes(fn=ins_gt.__getattribute__(method_name))
    for k, v in fw_list2.items():
        if k not in fw_list:
            fw_list[k] = []
        fw_list[k].extend(v)

    print(jsonpickle.dumps([ret_np_from_gt_flat, fw_list2]))


if __name__ == "__main__":
    from ivy.functional.ivy.gradients import _variable

    arg_lis = sys.argv
    fw_lis = []
    for i in arg_lis[1:]:
        if i.split("/")[0] == "jax":
            fw_lis.append(i.split("/")[0] + "/" + i.split("/")[1])
            fw_lis.append(i.split("/")[2] + "/" + i.split("/")[3])
        else:
            fw_lis.append(i)
    config.allow_global_framework_imports(fw=fw_lis)
    j = 1
    import ivy

    ivy.set_backend(arg_lis[2].split("/")[0])
    import numpy

    while j:
        try:
            z = input()
            if z == "1" or z == "1a":
                dtype_handler(arg_lis[2].split("/")[0], z)
                continue
            if z == "2":
                gradient_test()
                continue
            if z == "3":
                method_test()
                continue

            args_np = jsonpickle.loads(make_json_pickable(z))
            arg_np_vals = jsonpickle.loads(make_json_pickable(input()))
            args_idxs = jsonpickle.loads(make_json_pickable(input()))
            kwargs_np = jsonpickle.loads(make_json_pickable(input()))
            kwargs_idxs = jsonpickle.loads(make_json_pickable(input()))
            kwarg_np_vals = jsonpickle.loads(make_json_pickable(input()))
            input_dtypes = jsonpickle.loads(make_json_pickable(input()))
            test_flags = jsonpickle.loads(make_json_pickable(input()))
            fn_name = jsonpickle.loads(make_json_pickable(input()))
            with_out = test_flags.with_out
            args, kwargs, _, _, _ = create_args_kwargs(
                args_np=args_np,
                arg_np_vals=arg_np_vals,
                args_idxs=args_idxs,
                kwargs_np=kwargs_np,
                kwargs_idxs=kwargs_idxs,
                kwarg_np_vals=kwarg_np_vals,
                input_dtypes=input_dtypes,
                test_flags=test_flags,
            )
            ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
                ivy.__dict__[fn_name], *args, **kwargs
            )
            if with_out:
                test_ret_from_gt = (
                    ret_from_gt[getattr(ivy.__dict__[fn_name], "out_index")]
                    if hasattr(ivy.__dict__[fn_name], "out_index")
                    else ret_from_gt
                )
                out_from_gt = ivy.nested_map(
                    test_ret_from_gt,
                    ivy.zeros_like,
                    to_mutable=True,
                    include_derived=True,
                )
                ret_from_gt, ret_np_from_gt_flat = get_ret_and_flattened_np_array(
                    ivy.__dict__[fn_name], *args, **kwargs, out=out_from_gt
                )
            fw_list = gradient_unsupported_dtypes(fn=ivy.__dict__[fn_name])

            ground_output = jsonpickle.dumps(
                [ivy.to_numpy(ret_from_gt), ret_np_from_gt_flat, fw_list]
            )
            print(ground_output)
        except EOFError:
            continue
        except Exception as e:
            raise e
