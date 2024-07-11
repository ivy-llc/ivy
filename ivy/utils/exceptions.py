import warnings
import ivy
import functools
from typing import Callable
import traceback as tb
import inspect
import os
import ast
import builtins

# Helpers #
# ------- #


def _remove_so_log(trace):
    old_stack_trace = tb.extract_tb(trace)
    old_frames = inspect.getinnerframes(trace)

    transpile_frame = None
    module_frame = None
    module_st = None
    traced_lineno = None

    new_stack_trace = []
    track = False

    for idx, st in enumerate(old_stack_trace):
        if ".pyx" in repr(st):
            continue
        if "<string>" in repr(st):
            if "compiled_fn" in repr(st) and module_frame:
                track = True
                traced_lineno = st.lineno

        if "<module>" in repr(st):
            module_frame = old_frames[idx]
            module_st = st
        elif (
            transpile_frame is None
            and os.path.join("ivy", "compiler") in st.filename
            and st.name in ["compile", "transpile"]
        ):
            transpile_frame = old_frames[idx]
        elif track:
            ret_st = _align_source(
                st, transpile_frame, module_frame, module_st, traced_lineno
            )
            if ret_st:
                [new_stack_trace.append(r) for r in ret_st]

        if track:
            track = False
        else:
            new_stack_trace.append(st)

    return new_stack_trace


def _align_source(st, transpile_frame, module_frame, module_st, traced_lineno):
    from ivy.compiler.utils.VVX import trace_obj
    from ivy.compiler.utils.IIV import Graph

    curr_obj = [None, None, "", ""]
    if transpile_frame:
        t_v = inspect.getargvalues(transpile_frame.frame)
        obj = t_v.locals[t_v.varargs][0]

        traced_data = trace_obj(obj, t_v.locals["args"], t_v.locals["kwargs"], {})
        curr_obj[0] = traced_data[1]
        curr_obj[1] = traced_data[2]
        curr_obj[2] = traced_data[3]

    if module_frame:
        t_v = inspect.getargvalues(module_frame.frame)
        for k, v in t_v.locals.items():
            if k in module_st.line and isinstance(v, Graph):
                traced_data = trace_obj(t_v.locals[v.__name__], (), {}, {})
                curr_obj[0] = traced_data[1]
                curr_obj[1] = traced_data[2]
                curr_obj[2] = v.__name__

                if traced_lineno:
                    line = v._Graph__fn_str.split("\n")[traced_lineno - 1]
                    line = line.split("=")[1].strip()
                    line = line.split("(")[0].strip()
                    target_name = line.split(".")[-1].strip()
                    curr_obj[3] = line
                    area = traced_lineno / len(v._Graph__fn_str.strip().split("\n"))

                    curr_obj = _get_traces(curr_obj, area, t_v.locals, target_name)

    if curr_obj[0] is None:
        return None
    if not isinstance(curr_obj[0], list):
        curr_obj = [curr_obj]
    return curr_obj


def _get_traces(curr_obj, area, local_dict, target_name):
    from ivy.compiler.utils.VVX import trace_obj, get_source_code, CallVistior

    traces_list = []
    func = local_dict[curr_obj[2]]
    func_module = inspect.getmodule(func)
    rooted_source = get_source_code(func).strip()

    try:
        module_ast = ast.parse(rooted_source)
        visitor = CallVistior(func_module)
        visitor.visit(module_ast)
    except SyntaxError:
        pass

    non_lib_objs_name_list = [f.__name__ for f in visitor.non_lib_objs]
    rooted_src_list = rooted_source.split("\n")
    max_idx = round(len(rooted_src_list) * area) - 1

    for i in range(max_idx, 0, -1):
        if target_name in rooted_src_list[i]:
            curr_obj[3] = rooted_src_list[i]
            curr_obj[1] += i
            break
        elif builtins.any(
            [name in rooted_src_list[i] for name in non_lib_objs_name_list]
        ):
            found = False
            for name in non_lib_objs_name_list:
                if name in rooted_src_list[i]:
                    traced_data = trace_obj(local_dict[name], (), {}, {})
                    ret_obj = [traced_data[1], traced_data[2], name, curr_obj[3]]
                    ret_obj = _get_traces(ret_obj, 1, local_dict, target_name)
                    if ret_obj:
                        traces_list += ret_obj
                        found = True
                        break
            if found:
                curr_obj[3] = rooted_src_list[i]
                curr_obj[1] += i
                break
    return [curr_obj] + traces_list


def _check_if_path_found(path, full_path):
    """Check if the path is found in the full path.

    Parameters
    ----------
    path
        the path to check
    full_path
        the full path to check

    Returns
    -------
    ret
        True if the path is found, False otherwise
    """
    return path in full_path


def _configure_stack_trace(traceback):
    """Configure the stack trace to be displayed in the console.

    Parameters
    ----------
    traceback
        the traceback object
    """
    tb = traceback
    trace_mode = ivy.exception_trace_mode
    show_wrappers = ivy.show_func_wrapper_trace_mode

    ivy_path = os.path.join("ivy", "functional", "ivy")
    frontend_path = os.path.join("ivy", "functional", "frontends")
    wrapper_path = os.path.join("ivy", "func_wrapper.py")

    while tb.tb_next:
        frame = tb.tb_next.tb_frame
        file_path = frame.f_code.co_filename
        if trace_mode == "ivy":
            if _check_if_path_found(ivy_path, file_path):
                tb = tb.tb_next
            else:
                tb.tb_next = tb.tb_next.tb_next
        elif trace_mode == "frontend":
            if _check_if_path_found(frontend_path, file_path) or _check_if_path_found(
                ivy_path, file_path
            ):
                tb = tb.tb_next
            else:
                tb.tb_next = tb.tb_next.tb_next
        elif not show_wrappers:
            if _check_if_path_found(wrapper_path, file_path):
                tb.tb_next = tb.tb_next.tb_next
            else:
                tb = tb.tb_next
        else:
            tb = tb.tb_next


def _add_native_error(default):
    """Append the native error to the message if it exists.

    Parameters
    ----------
    default
        list containing all the messages

    Returns
    -------
    ret
        list containing all the messages, with the native error appended if it exists
    """
    trace_mode = ivy.exception_trace_mode
    if isinstance(default[-1], Exception):
        if isinstance(default[-1], IvyException):
            if default[-1].native_error is not None:
                # native error was passed in the message
                native_error = default[-1].native_error
            else:
                # a string was passed in the message
                # hence the last element is an IvyException
                default[-1] = str(default[-1])
                return default
        else:
            # exception was raised by the backend natively
            native_error = default[-1]
        if trace_mode == "full":
            default[-1] = native_error.__class__.__name__
            default.append(str(native_error))
        else:
            default[-1] = str(native_error)
    return default


def _combine_messages(*messages, include_backend=True):
    if not include_backend:
        return " ".join(messages)
    default = [
        "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
    ]
    delimiter = ": "
    for message in messages:
        default.append(message)

    # adding the native error as well if it exists and the trace mode is set to "full"
    default = _add_native_error(default)
    return delimiter.join(default)


class IvyException(Exception):
    def __init__(self, *messages, include_backend=False):
        self.native_error = (
            messages[0]
            if len(messages) == 1
            and isinstance(messages[0], Exception)
            and not include_backend
            else None
        )
        if self.native_error is None:
            super().__init__(
                _combine_messages(*messages, include_backend=include_backend)
            )
        else:
            super().__init__(str(messages[0]))


class IvyBackendException(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyInvalidBackendException(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyNotImplementedException(IvyException, NotImplementedError):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyError(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyIndexError(IvyException, IndexError):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyAttributeError(IvyException, AttributeError):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyValueError(IvyException, ValueError):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyBroadcastShapeError(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyDtypePromotionError(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class IvyDeviceError(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


class InplaceUpdateException(IvyException):
    def __init__(self, *messages, include_backend=False):
        super().__init__(*messages, include_backend=include_backend)


_non_ivy_exceptions_mapping = {
    IndexError: IvyIndexError,
    AttributeError: IvyAttributeError,
    ValueError: IvyValueError,
    Exception: IvyBackendException,
    NotImplementedError: IvyNotImplementedException,
}


def handle_exceptions(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def _handle_exceptions(*args, **kwargs):
        """Catch all exceptions and raise them in IvyException.

        Parameters
        ----------
        args
            The arguments to be passed to the function.

        kwargs
            The keyword arguments to be passed to the function.

        Returns
        -------
            The return of the function, or raise IvyException if error is thrown.
        """
        try:
            return fn(*args, **kwargs)
        except IvyException as e:
            _handle_exceptions_helper(e, type(e))
        except Exception as e:
            ivy_exception = _non_ivy_exceptions_mapping.get(
                type(e), IvyBackendException
            )
            _handle_exceptions_helper(e, ivy_exception)

    def _handle_exceptions_helper(e, cls):
        _configure_stack_trace(e.__traceback__)
        raise cls(fn.__name__, str(e), include_backend=True)

    _handle_exceptions.handle_exceptions = True
    return _handle_exceptions


# Inplace Update

# to avoid raising warnings on setting the same backend multiple times
_inplace_warning_cache = {}


def _handle_inplace_mode(ivy_pack=None):
    if not ivy_pack:
        ivy_pack = ivy
    current_backend = ivy_pack.current_backend_str()
    if (
        current_backend != ""
        and not _inplace_warning_cache.get(current_backend)
        and not ivy_pack.native_inplace_support
        and ivy_pack.inplace_mode == "lenient"
    ):
        warnings.warn(
            f"The current backend: '{current_backend}' does not support "
            "inplace updates natively. Ivy would quietly create new arrays when "
            "using inplace updates with this backend, leading to memory overhead "
            "(same applies for views). If you want to control your memory "
            "management, consider doing ivy.set_inplace_mode('strict') which "
            "should raise an error whenever an inplace update is attempted "
            "with this backend."
        )
        _inplace_warning_cache[current_backend] = True


def _check_inplace_update_support(x, ensure_in_backend):
    current_backend = ivy.current_backend_str()
    is_tf_variable = current_backend == "tensorflow" and not ivy.is_ivy_array(
        x, exclusive=True
    )
    if (
        ensure_in_backend
        or ivy.is_native_array(x)
        or (ivy.inplace_mode == "strict" and not is_tf_variable)
    ):
        raise ivy.utils.exceptions.InplaceUpdateException(
            f"{current_backend} does not support inplace updates "
            "and ivy cannot support the operation in 'strict' mode\n"
            "To enable inplace update, use ivy.set_inplace_mode('lenient')\n"
        )
