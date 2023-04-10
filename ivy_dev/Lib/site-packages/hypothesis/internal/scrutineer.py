# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import sys
from collections import defaultdict
from functools import lru_cache, reduce
from os import sep
from pathlib import Path

from hypothesis._settings import Phase, Verbosity
from hypothesis.internal.escalation import is_hypothesis_file


@lru_cache(maxsize=None)
def should_trace_file(fname):
    # fname.startswith("<") indicates runtime code-generation via compile,
    # e.g. compile("def ...", "<string>", "exec") in e.g. attrs methods.
    return not (is_hypothesis_file(fname) or fname.startswith("<"))


class Tracer:
    """A super-simple branch coverage tracer."""

    __slots__ = ("branches", "_previous_location")

    def __init__(self):
        self.branches = set()
        self._previous_location = None

    def trace(self, frame, event, arg):
        if event == "call":
            return self.trace
        elif event == "line":
            fname = frame.f_code.co_filename
            if should_trace_file(fname):
                current_location = (fname, frame.f_lineno)
                self.branches.add((self._previous_location, current_location))
                self._previous_location = current_location


UNHELPFUL_LOCATIONS = (
    # There's a branch which is only taken when an exception is active while exiting
    # a contextmanager; this is probably after the fault has been triggered.
    # Similar reasoning applies to a few other standard-library modules: even
    # if the fault was later, these still aren't useful locations to report!
    f"{sep}contextlib.py",
    f"{sep}inspect.py",
    f"{sep}re.py",
    f"{sep}re{sep}__init__.py",  # refactored in Python 3.11
    f"{sep}warnings.py",
    # Quite rarely, the first AFNP line is in Pytest's internals.
    f"{sep}_pytest{sep}assertion{sep}rewrite.py",
    f"{sep}_pytest{sep}_io{sep}saferepr.py",
)


def get_explaining_locations(traces):
    # Traces is a dict[interesting_origin | None, set[frozenset[tuple[str, int]]]]
    # Each trace in the set might later become a Counter instead of frozenset.
    if not traces:
        return {}

    unions = {origin: set().union(*values) for origin, values in traces.items()}
    seen_passing = {None}.union(*unions.pop(None, set()))

    always_failing_never_passing = {
        origin: reduce(set.intersection, [set().union(*v) for v in values])
        - seen_passing
        for origin, values in traces.items()
        if origin is not None
    }

    # Build the observed parts of the control-flow graph for each origin
    cf_graphs = {origin: defaultdict(set) for origin in unions}
    for origin, seen_arcs in unions.items():
        for src, dst in seen_arcs:
            cf_graphs[origin][src].add(dst)
        assert cf_graphs[origin][None], "Expected start node with >=1 successor"

    # For each origin, our explanation is the always_failing_never_passing lines
    # which are reachable from the start node (None) without passing through another
    # AFNP line.  So here's a whatever-first search with early stopping:
    explanations = defaultdict(set)
    for origin in unions:
        queue = {None}
        seen = set()
        while queue:
            assert queue.isdisjoint(seen), f"Intersection: {queue & seen}"
            src = queue.pop()
            seen.add(src)
            if src in always_failing_never_passing[origin]:
                explanations[origin].add(src)
            else:
                queue.update(cf_graphs[origin][src] - seen)

    # The last step is to filter out explanations that we know would be uninformative.
    # When this is the first AFNP location, we conclude that Scrutineer missed the
    # real divergence (earlier in the trace) and drop that unhelpful explanation.
    return {
        origin: {loc for loc in afnp_locs if not loc[0].endswith(UNHELPFUL_LOCATIONS)}
        for origin, afnp_locs in explanations.items()
    }


LIB_DIR = str(Path(sys.executable).parent / "lib")
EXPLANATION_STUB = (
    "Explanation:",
    "    These lines were always and only run by failing examples:",
)


def make_report(explanations, cap_lines_at=5):
    report = defaultdict(list)
    for origin, locations in explanations.items():
        report_lines = [f"        {fname}:{lineno}" for fname, lineno in locations]
        report_lines.sort(key=lambda line: (line.startswith(LIB_DIR), line))
        if len(report_lines) > cap_lines_at + 1:
            msg = "        (and {} more with settings.verbosity >= verbose)"
            report_lines[cap_lines_at:] = [msg.format(len(report_lines[cap_lines_at:]))]
        if report_lines:  # We might have filtered out every location as uninformative.
            report[origin] = list(EXPLANATION_STUB) + report_lines
    return report


def explanatory_lines(traces, settings):
    if Phase.explain in settings.phases and sys.gettrace() and not traces:
        return defaultdict(list)
    # Return human-readable report lines summarising the traces
    explanations = get_explaining_locations(traces)
    max_lines = 5 if settings.verbosity <= Verbosity.normal else float("inf")
    return make_report(explanations, cap_lines_at=max_lines)
