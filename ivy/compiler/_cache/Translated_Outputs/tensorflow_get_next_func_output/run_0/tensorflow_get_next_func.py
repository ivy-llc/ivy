import inspect
import ast

from .tensorflow__helpers import tensorflow_get_item
from .tensorflow__helpers import tensorflow_retrieve_object


def tensorflow_get_next_func(obj):
    from .tensorflow_CallVisitor import tensorflow_CallVisitor

    stack = inspect.stack()
    for frame_info in stack:
        if frame_info == obj._previous_frame_info:
            calling_frame = frame_info.frame
            break
    else:
        return None
    if "Sequential" in frame_info.filename:
        try:
            self_seq = calling_frame.f_locals["self"]
            idx = calling_frame.f_locals["i"]
            next_func = tensorflow_get_item(self_seq, idx + 1)
            return next_func
        except IndexError:
            for frame_info in tensorflow_get_item(
                stack, slice(stack.index(frame_info) + 1, None, None)
            ):
                if frame_info == self_seq._previous_frame_info:
                    calling_frame = frame_info.frame
                    break
            else:
                return None
    lines, start_line_no = inspect.getsourcelines(calling_frame)
    current_line_no = calling_frame.f_lineno
    relative_line_no = current_line_no - start_line_no
    try:
        next_line = tensorflow_get_item(lines, relative_line_no + 1).strip()
        tree = ast.parse(next_line)
        visitor = tensorflow_CallVisitor()
        visitor.visit(tree)
        next_call_str = visitor.func_name
    except Exception:
        next_call_str = ""
    next_func = tensorflow_retrieve_object(calling_frame, next_call_str)
    return next_func
