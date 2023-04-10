# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run all functions registered for the "hypothesis" entry point.

This can be used with `st.register_type_strategy` to register strategies for your
custom types, running the relevant code when *hypothesis* is imported instead of
your package.
"""

try:
    # We prefer to use importlib.metadata, or the backport on Python <= 3.7,
    # because it's much faster than pkg_resources (200ms import time speedup).
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:
        import importlib_metadata  # type: ignore  # mypy thinks this is a redefinition

    def get_entry_points():
        try:
            eps = importlib_metadata.entry_points(group="hypothesis")
        except TypeError:  # pragma: no cover
            # Load-time selection requires Python >= 3.10 or importlib_metadata >= 3.6,
            # so we'll retain this fallback logic for some time to come.  See also
            # https://importlib-metadata.readthedocs.io/en/latest/using.html
            eps = importlib_metadata.entry_points().get("hypothesis", [])
        yield from eps

except ImportError:
    # But if we're not on Python >= 3.8 and the importlib_metadata backport
    # is not installed, we fall back to pkg_resources anyway.
    try:
        import pkg_resources
    except ImportError:
        import warnings

        from hypothesis.errors import HypothesisWarning

        warnings.warn(
            "Under Python <= 3.7, Hypothesis requires either the importlib_metadata "
            "or setuptools package in order to load plugins via entrypoints.",
            HypothesisWarning,
        )

        def get_entry_points():
            yield from ()

    else:

        def get_entry_points():
            yield from pkg_resources.iter_entry_points("hypothesis")


def run():
    for entry in get_entry_points():  # pragma: no cover
        hook = entry.load()
        if callable(hook):
            hook()
