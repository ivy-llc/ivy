# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Operators corresponding to Python builtin functions.

List of built-in functions: https://docs.python.org/3/library/functions.html
"""

import inspect
from typing import Any
import traceback

from .... import globals as glob


def _find_originating_frame(caller_fn_scope, innermost=True):
    """Locates the frame in which `caller_fn_scope` was defined."""
    ctx_frame = inspect.currentframe()
    result = None
    while ctx_frame is not None:
        # Note it should not be normally possible to get false positives this way
        # because the function scope object is not accessible to user code (barring
        # call stack introspection).
        if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
            result = ctx_frame
            if innermost:
                break
        ctx_frame = ctx_frame.f_back

    assert result is not None, (
        "the conversion process should ensure the caller_fn_scope is always"
        " found somewhere on the call stack"
    )

    return result


def locals_in_original_context(caller_fn_scope):
    """Executes the locals function in the context of a specified function."""
    return _find_originating_frame(caller_fn_scope, innermost=True).f_locals


def globals_in_original_context(caller_fn_scope):
    """Executes the locals function in the context of a specified function."""
    return _find_originating_frame(caller_fn_scope, innermost=True).f_globals


def eval_in_original_context(f, args, caller_fn_scope):
    """Executes the eval function in the context of a specified function."""
    # When control flow is rewritten using functions, eval should use the
    # variables found in the same block where it was called. That is equivalent
    # to the innermost function call.
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)

    args = (
        args[0],
        ctx_frame.f_globals if len(args) < 2 else args[1],
        ctx_frame.f_locals if len(args) < 3 else args[2],
    )
    return f(*args)


def super_in_original_context(f, args, caller_fn_scope):
    """Executes the super function in the context of a specified function.

    See https://docs.python.org/3/library/functions.html#super for the exact
    details

    Args:
      f: Callable, typically the super builtin
      args: List[Any], the original call arguments
      caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
        scope of the converted function in which this call was originally made

    Returns:
      The result of calling `f` as if it was called in the frame indicated by
        `caller_fn_scope`.
    """

    # Only the no-arg call is desugared.
    if args:
        return f(*args)

    # Inner functions seem to include their closure in f_locals, so we need
    # to find the outermost frame.
    ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)

    # When super(..) is called without arguments, it looks for __class__ cell
    # variable and the first argument passed in the enclosing function according
    # to the spec https://www.python.org/dev/peps/pep-3135/ .
    #
    # We couldn't verify if `inspect.currentframe().f_code.co_varnames[0]` is
    # guaranteed to be the first argument from an official doc or PEP, however,
    # it's fairly stable and well established:
    # - An unofficial community doc mentions it.
    #   https://python-reference.readthedocs.io/en/latest/docs/code/varnames.html
    # - CPython has tests checking that order, which was merged in 2008, and
    #   unchanged since then.
    #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py2_test_grammar.py#L157
    #   https://github.com/python/cpython/blame/2f224a077a83ac9de8a12bb7dcc516642b8176d8/Lib/lib2to3/tests/data/py3_test_grammar.py#L192
    #
    # Note: the name can be more reliably obtained by inspecting the calling
    # function's argspec.
    #
    # Even though methods can be declared using *args (def method(*args)),
    # that pattern is disallowed by super() -- it raises super() no arguments.
    # Method definitions using **kwargs are not allowed at all.
    # In other words, we can always assume that self is on the first positional
    # argument (for correct code).
    #
    # TODO(mdan): Consider additional checks in case the input code is incorrect.
    # For example, the error might be cryptic compared to what super() regularly
    # raises.

    type_arg = ctx_frame.f_locals["__class__"]
    self_arg_name = ctx_frame.f_code.co_varnames[0]
    self_arg = ctx_frame.f_locals[self_arg_name]
    ret = f(type_arg, self_arg)

    # make the object class returned by super() patchable.
    if len(type_arg.__mro__) == 2 and type_arg.__mro__[1] == object:
        try:
            ret = WrappableSuper(type_arg, self_arg)
        except Exception as e:
            pass
        pass

    return ret


class WrappableSuper(super):
    def __init__(self, type_arg, self_arg):
        super().__init__(type_arg, self_arg)
        self.instance = self_arg

    def __getattribute__(self, __name: str) -> Any:
        try:
            ret = super().__getattribute__(__name)
        except Exception as e:
            pass

        if __name == "__setattr__":
            try:

                def method_wrapper(str, val):
                    try:
                        glob.wrapped_fns[id(object.__setattr__)][1](
                            self.instance, str, val
                        )
                    except Exception as e:
                        traceback.print_exception()
                    return self.instance

                ret = method_wrapper
            except Exception as e:
                traceback.print_exception()

        return ret

    def __getattr__(self, __name: str) -> Any:
        return self.__getattribute__(__name)
