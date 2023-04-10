# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.


class IntervalSet:
    def __init__(self, intervals):
        self.intervals = tuple(intervals)
        self.offsets = [0]
        for u, v in self.intervals:
            self.offsets.append(self.offsets[-1] + v - u + 1)
        self.size = self.offsets.pop()

    def __len__(self):
        return self.size

    def __iter__(self):
        for u, v in self.intervals:
            yield from range(u, v + 1)

    def __getitem__(self, i):
        if i < 0:
            i = self.size + i
        if i < 0 or i >= self.size:
            raise IndexError(f"Invalid index {i} for [0, {self.size})")
        # Want j = maximal such that offsets[j] <= i

        j = len(self.intervals) - 1
        if self.offsets[j] > i:
            hi = j
            lo = 0
            # Invariant: offsets[lo] <= i < offsets[hi]
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if self.offsets[mid] <= i:
                    lo = mid
                else:
                    hi = mid
            j = lo
        t = i - self.offsets[j]
        u, v = self.intervals[j]
        r = u + t
        assert r <= v
        return r

    def __repr__(self):
        return f"IntervalSet({self.intervals!r})"

    def index(self, value):
        for offset, (u, v) in zip(self.offsets, self.intervals):
            if u == value:
                return offset
            elif u > value:
                raise ValueError(f"{value} is not in list")
            if value <= v:
                return offset + (value - u)
        raise ValueError(f"{value} is not in list")

    def index_above(self, value):
        for offset, (u, v) in zip(self.offsets, self.intervals):
            if u >= value:
                return offset
            if value <= v:
                return offset + (value - u)
        return self.size
