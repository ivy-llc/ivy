# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import codecs
import inspect
import platform
import sys
import typing
from functools import partial
from typing import Any, ForwardRef, Tuple

try:
    from typing import get_args as get_args
except ImportError:
    # remove at Python 3.7 end-of-life
    from collections.abc import Callable as _Callable

    def get_args(
        tp: Any,
    ) -> Tuple[Any, ...]:  # pragma: no cover
        """
        Examples
        --------
        >>> assert get_args(int) == ()
        >>> assert get_args(Dict[str, int]) == (str, int)
        >>> assert get_args(Union[int, Union[T, int], str][int]) == (int, str)
        >>> assert get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
        >>> assert get_args(Callable[[], T][int]) == ([], int)
        """
        if hasattr(tp, "__origin__") and hasattr(tp, "__args__"):
            args = tp.__args__
            if (
                getattr(tp, "__origin__", None) is _Callable
                and args
                and args[0] is not Ellipsis
            ):
                args = (list(args[:-1]), args[-1])
            return args
        return ()


try:
    from typing import get_origin as get_origin
except ImportError:
    # remove at Python 3.7 end-of-life
    from collections.abc import Callable as _Callable

    def get_origin(tp: Any) -> typing.Optional[Any]:  # pragma: no cover
        """Get the unsubscripted version of a type.
        This supports generic types, Callable, Tuple, Union, Literal, Final and ClassVar.
        Return None for unsupported types. Examples::
            get_origin(Literal[42]) is Literal
            get_origin(int) is None
            get_origin(ClassVar[int]) is ClassVar
            get_origin(Generic) is Generic
            get_origin(Generic[T]) is Generic
            get_origin(Union[T, int]) is Union
            get_origin(List[Tuple[T, T]][int]) == list
        """
        if hasattr(tp, "__origin__"):
            return tp.__origin__
        if tp is typing.Generic:
            return typing.Generic
        return None


try:
    BaseExceptionGroup = BaseExceptionGroup
    ExceptionGroup = ExceptionGroup  # pragma: no cover
except NameError:
    from exceptiongroup import (
        BaseExceptionGroup as BaseExceptionGroup,
        ExceptionGroup as ExceptionGroup,
    )
if typing.TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import Concatenate as Concatenate, ParamSpec as ParamSpec
else:
    try:
        from typing import Concatenate as Concatenate, ParamSpec as ParamSpec
    except ImportError:
        try:
            from typing_extensions import (
                Concatenate as Concatenate,
                ParamSpec as ParamSpec,
            )
        except ImportError:
            Concatenate, ParamSpec = None, None

PYPY = platform.python_implementation() == "PyPy"
GRAALPY = platform.python_implementation() == "GraalVM"
WINDOWS = platform.system() == "Windows"


def escape_unicode_characters(s: str) -> str:
    return codecs.encode(s, "unicode_escape").decode("ascii")


def int_from_bytes(data: typing.Union[bytes, bytearray]) -> int:
    return int.from_bytes(data, "big")


def int_to_bytes(i: int, size: int) -> bytes:
    return i.to_bytes(size, "big")


def int_to_byte(i: int) -> bytes:
    return bytes([i])


def is_typed_named_tuple(cls):
    """Return True if cls is probably a subtype of `typing.NamedTuple`.

    Unfortunately types created with `class T(NamedTuple):` actually
    subclass `tuple` directly rather than NamedTuple.  This is annoying,
    and means we just have to hope that nobody defines a different tuple
    subclass with similar attributes.
    """
    return (
        issubclass(cls, tuple)
        and hasattr(cls, "_fields")
        and (hasattr(cls, "_field_types") or hasattr(cls, "__annotations__"))
    )


def _hint_and_args(x):
    return (x,) + get_args(x)


def get_type_hints(thing):
    """Like the typing version, but tries harder and never errors.

    Tries harder: if the thing to inspect is a class but typing.get_type_hints
    raises an error or returns no hints, then this function will try calling it
    on the __init__ method. This second step often helps with user-defined
    classes on older versions of Python. The third step we take is trying
    to fetch types from the __signature__ property.
    They override any other ones we found earlier.

    Never errors: instead of raising TypeError for uninspectable objects, or
    NameError for unresolvable forward references, just return an empty dict.
    """
    if isinstance(thing, partial):
        from hypothesis.internal.reflection import get_signature

        bound = set(get_signature(thing.func).parameters).difference(
            get_signature(thing).parameters
        )
        return {k: v for k, v in get_type_hints(thing.func).items() if k not in bound}

    kwargs = {} if sys.version_info[:2] < (3, 9) else {"include_extras": True}

    try:
        hints = typing.get_type_hints(thing, **kwargs)
    except (AttributeError, TypeError, NameError):  # pragma: no cover
        hints = {}

    if inspect.isclass(thing):
        try:
            hints.update(typing.get_type_hints(thing.__init__, **kwargs))
        except (TypeError, NameError, AttributeError):
            pass

    try:
        if hasattr(thing, "__signature__"):
            # It is possible for the signature and annotations attributes to
            # differ on an object due to renamed arguments.
            from hypothesis.internal.reflection import get_signature
            from hypothesis.strategies._internal.types import is_a_type

            vkinds = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in get_signature(thing).parameters.values():
                if (
                    p.kind not in vkinds
                    and is_a_type(p.annotation)
                    and p.annotation is not p.empty
                ):
                    p_hint = p.annotation

                    # Defer to `get_type_hints` if signature annotation is, or
                    # contains, a forward reference that is otherwise resolved.
                    if any(
                        isinstance(sig_hint, ForwardRef)
                        and not isinstance(hint, ForwardRef)
                        for sig_hint, hint in zip(
                            _hint_and_args(p.annotation),
                            _hint_and_args(hints.get(p.name, Any)),
                        )
                    ):
                        p_hint = hints[p.name]
                    if p.default is None:
                        hints[p.name] = typing.Optional[p_hint]
                    else:
                        hints[p.name] = p_hint
    except (AttributeError, TypeError, NameError):  # pragma: no cover
        pass

    return hints


def update_code_location(code, newfile, newlineno):
    """Take a code object and lie shamelessly about where it comes from.

    Why do we want to do this? It's for really shallow reasons involving
    hiding the hypothesis_temporary_module code from test runners like
    pytest's verbose mode. This is a vastly disproportionate terrible
    hack that I've done purely for vanity, and if you're reading this
    code you're probably here because it's broken something and now
    you're angry at me. Sorry.
    """
    if hasattr(code, "replace"):
        # Python 3.8 added positional-only params (PEP 570), and thus changed
        # the layout of code objects.  In beta1, the `.replace()` method was
        # added to facilitate future-proof code.  See BPO-37032 for details.
        return code.replace(co_filename=newfile, co_firstlineno=newlineno)

    else:  # pragma: no cover
        # This field order is accurate for 3.5 - 3.7, but not 3.8 when a new field
        # was added for positional-only arguments.  However it also added a .replace()
        # method that we use instead of field indices, so they're fine as-is.
        CODE_FIELD_ORDER = [
            "co_argcount",
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_filename",
            "co_name",
            "co_firstlineno",
            "co_lnotab",
            "co_freevars",
            "co_cellvars",
        ]
        unpacked = [getattr(code, name) for name in CODE_FIELD_ORDER]
        unpacked[CODE_FIELD_ORDER.index("co_filename")] = newfile
        unpacked[CODE_FIELD_ORDER.index("co_firstlineno")] = newlineno
        return type(code)(*unpacked)


# Under Python 2, math.floor and math.ceil returned floats, which cannot
# represent large integers - eg `float(2**53) == float(2**53 + 1)`.
# We therefore implement them entirely in (long) integer operations.
# We still use the same trick on Python 3, because Numpy values and other
# custom __floor__ or __ceil__ methods may convert via floats.
# See issue #1667, Numpy issue 9068.
def floor(x):
    y = int(x)
    if y != x and x < 0:
        return y - 1
    return y


def ceil(x):
    y = int(x)
    if y != x and x > 0:
        return y + 1
    return y


def bad_django_TestCase(runner):
    if runner is None or "django.test" not in sys.modules:
        return False
    else:  # pragma: no cover
        if not isinstance(runner, sys.modules["django.test"].TransactionTestCase):
            return False

        from hypothesis.extra.django._impl import HypothesisTestCase

        return not isinstance(runner, HypothesisTestCase)
