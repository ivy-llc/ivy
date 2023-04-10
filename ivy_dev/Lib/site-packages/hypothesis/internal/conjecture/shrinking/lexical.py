# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from hypothesis.internal.compat import int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.shrinking.common import Shrinker
from hypothesis.internal.conjecture.shrinking.integer import Integer
from hypothesis.internal.conjecture.shrinking.ordering import Ordering

"""
This module implements a lexicographic minimizer for blocks of bytes.
"""


class Lexical(Shrinker):
    def make_immutable(self, value):
        return bytes(value)

    @property
    def size(self):
        return len(self.current)

    def check_invariants(self, value):
        assert len(value) == self.size

    def left_is_better(self, left, right):
        return left < right

    def incorporate_int(self, i):
        return self.incorporate(int_to_bytes(i, self.size))

    @property
    def current_int(self):
        return int_from_bytes(self.current)

    def minimize_as_integer(self, full=False):
        Integer.shrink(
            self.current_int,
            lambda c: c == self.current_int or self.incorporate_int(c),
            random=self.random,
            full=full,
        )

    def partial_sort(self):
        Ordering.shrink(self.current, self.consider, random=self.random)

    def short_circuit(self):
        """This is just an assemblage of other shrinkers, so we rely on their
        short circuiting."""
        return False

    def run_step(self):
        self.minimize_as_integer()
        self.partial_sort()
