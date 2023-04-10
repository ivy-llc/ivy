# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import math
from collections import defaultdict
from typing import NoReturn, Union

from hypothesis import Verbosity, settings
from hypothesis.errors import InvalidArgument, UnsatisfiedAssumption
from hypothesis.internal.compat import BaseExceptionGroup
from hypothesis.internal.conjecture.data import ConjectureData
from hypothesis.internal.reflection import get_pretty_function_description
from hypothesis.internal.validation import check_type
from hypothesis.reporting import report, verbose_report
from hypothesis.utils.dynamicvariables import DynamicVariable
from hypothesis.vendor.pretty import IDKey


def reject() -> NoReturn:
    raise UnsatisfiedAssumption()


def assume(condition: object) -> bool:
    """Calling ``assume`` is like an :ref:`assert <python:assert>` that marks
    the example as bad, rather than failing the test.

    This allows you to specify properties that you *assume* will be
    true, and let Hypothesis try to avoid similar examples in future.
    """
    if not condition:
        raise UnsatisfiedAssumption()
    return True


_current_build_context = DynamicVariable(None)


def currently_in_test_context() -> bool:
    """Return ``True`` if the calling code is currently running inside an
    :func:`@given <hypothesis.given>` or :doc:`stateful <stateful>` test,
    ``False`` otherwise.

    This is useful for third-party integrations and assertion helpers which
    may be called from traditional or property-based tests, but can only use
    :func:`~hypothesis.assume` or :func:`~hypothesis.target` in the latter case.
    """
    return _current_build_context.value is not None


def current_build_context() -> "BuildContext":
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("No build context registered")
    return context


class BuildContext:
    def __init__(self, data, is_final=False, close_on_capture=True):
        assert isinstance(data, ConjectureData)
        self.data = data
        self.tasks = []
        self.is_final = is_final
        self.close_on_capture = close_on_capture
        self.close_on_del = False
        # Use defaultdict(list) here to handle the possibility of having multiple
        # functions registered for the same object (due to caching, small ints, etc).
        # The printer will discard duplicates which return different representations.
        self.known_object_printers = defaultdict(list)

    def record_call(self, obj, func, a, kw):
        name = get_pretty_function_description(func)
        self.known_object_printers[IDKey(obj)].append(
            lambda obj, p, cycle: p.text("<...>") if cycle else p.repr_call(name, a, kw)
        )

    def __enter__(self):
        self.assign_variable = _current_build_context.with_value(self)
        self.assign_variable.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.assign_variable.__exit__(exc_type, exc_value, tb)
        errors = []
        for task in self.tasks:
            try:
                task()
            except BaseException as err:
                errors.append(err)
        if errors:
            if len(errors) == 1:
                raise errors[0] from exc_value
            raise BaseExceptionGroup("Cleanup failed", errors) from exc_value


def cleanup(teardown):
    """Register a function to be called when the current test has finished
    executing. Any exceptions thrown in teardown will be printed but not
    rethrown.

    Inside a test this isn't very interesting, because you can just use
    a finally block, but note that you can use this inside map, flatmap,
    etc. in order to e.g. insist that a value is closed at the end.
    """
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("Cannot register cleanup outside of build context")
    context.tasks.append(teardown)


def should_note():
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("Cannot make notes outside of a test")
    return context.is_final or settings.default.verbosity >= Verbosity.verbose


def note(value: str) -> None:
    """Report this value for the minimal failing example."""
    if should_note():
        report(value)


def event(value: str) -> None:
    """Record an event that occurred this test. Statistics on number of test
    runs with each event will be reported at the end if you run Hypothesis in
    statistics reporting mode.

    Events should be strings or convertible to them.
    """
    context = _current_build_context.value
    if context is None:
        raise InvalidArgument("Cannot make record events outside of a test")

    context.data.note_event(value)


def target(observation: Union[int, float], *, label: str = "") -> Union[int, float]:
    """Calling this function with an ``int`` or ``float`` observation gives it feedback
    with which to guide our search for inputs that will cause an error, in
    addition to all the usual heuristics.  Observations must always be finite.

    Hypothesis will try to maximize the observed value over several examples;
    almost any metric will work so long as it makes sense to increase it.
    For example, ``-abs(error)`` is a metric that increases as ``error``
    approaches zero.

    Example metrics:

    - Number of elements in a collection, or tasks in a queue
    - Mean or maximum runtime of a task (or both, if you use ``label``)
    - Compression ratio for data (perhaps per-algorithm or per-level)
    - Number of steps taken by a state machine

    The optional ``label`` argument can be used to distinguish between
    and therefore separately optimise distinct observations, such as the
    mean and standard deviation of a dataset.  It is an error to call
    ``target()`` with any label more than once per test case.

    .. note::
        **The more examples you run, the better this technique works.**

        As a rule of thumb, the targeting effect is noticeable above
        :obj:`max_examples=1000 <hypothesis.settings.max_examples>`,
        and immediately obvious by around ten thousand examples
        *per label* used by your test.

    :ref:`statistics` include the best score seen for each label,
    which can help avoid `the threshold problem
    <https://hypothesis.works/articles/threshold-problem/>`__ when the minimal
    example shrinks right down to the threshold of failure (:issue:`2180`).
    """
    check_type((int, float), observation, "observation")
    if not math.isfinite(observation):
        raise InvalidArgument(f"observation={observation!r} must be a finite float.")
    check_type(str, label, "label")

    context = _current_build_context.value
    if context is None:
        raise InvalidArgument(
            "Calling target() outside of a test is invalid.  "
            "Consider guarding this call with `if currently_in_test_context(): ...`"
        )
    verbose_report(f"Saw target(observation={observation!r}, label={label!r})")

    if label in context.data.target_observations:
        raise InvalidArgument(
            f"Calling target({observation!r}, label={label!r}) would overwrite "
            f"target({context.data.target_observations[label]!r}, label={label!r})"
        )
    else:
        context.data.target_observations[label] = observation

    return observation
