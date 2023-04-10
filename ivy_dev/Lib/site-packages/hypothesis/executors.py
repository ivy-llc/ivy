# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.


def default_executor(function):  # pragma: nocover
    raise NotImplementedError()  # We don't actually use this any more


def setup_teardown_executor(setup, teardown):
    setup = setup or (lambda: None)
    teardown = teardown or (lambda ex: None)

    def execute(function):
        token = None
        try:
            token = setup()
            return function()
        finally:
            teardown(token)

    return execute


def executor(runner):
    try:
        return runner.execute_example
    except AttributeError:
        pass

    if hasattr(runner, "setup_example") or hasattr(runner, "teardown_example"):
        return setup_teardown_executor(
            getattr(runner, "setup_example", None),
            getattr(runner, "teardown_example", None),
        )

    return default_executor


def default_new_style_executor(data, function):
    return function(data)


class ConjectureRunner:
    def hypothesis_execute_example_with_data(self, data, function):
        return function(data)


def new_style_executor(runner):
    if runner is None:
        return default_new_style_executor
    if isinstance(runner, ConjectureRunner):
        return runner.hypothesis_execute_example_with_data

    old_school = executor(runner)
    if old_school is default_executor:
        return default_new_style_executor
    else:
        return lambda data, function: old_school(lambda: function(data))
