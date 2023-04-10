# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import gzip
import json
import os
import sys
import tempfile
import unicodedata
from typing import Dict, Tuple

from hypothesis.configuration import mkdir_p, storage_directory
from hypothesis.errors import InvalidArgument

intervals = Tuple[Tuple[int, int], ...]
cache_type = Dict[Tuple[Tuple[str, ...], int, int, intervals], intervals]


def charmap_file():
    return storage_directory(
        "unicode_data", unicodedata.unidata_version, "charmap.json.gz"
    )


_charmap = None


def charmap():
    """Return a dict that maps a Unicode category, to a tuple of 2-tuples
    covering the codepoint intervals for characters in that category.

    >>> charmap()['Co']
    ((57344, 63743), (983040, 1048573), (1048576, 1114109))
    """
    global _charmap
    # Best-effort caching in the face of missing files and/or unwritable
    # filesystems is fairly simple: check if loaded, else try loading,
    # else calculate and try writing the cache.
    if _charmap is None:
        f = charmap_file()
        try:
            with gzip.GzipFile(f, "rb") as i:
                tmp_charmap = dict(json.load(i))

        except Exception:
            # This loop is reduced to using only local variables for performance;
            # indexing and updating containers is a ~3x slowdown.  This doesn't fix
            # https://github.com/HypothesisWorks/hypothesis/issues/2108 but it helps.
            category = unicodedata.category  # Local variable -> ~20% speedup!
            tmp_charmap = {}
            last_cat = category(chr(0))
            last_start = 0
            for i in range(1, sys.maxunicode + 1):
                cat = category(chr(i))
                if cat != last_cat:
                    tmp_charmap.setdefault(last_cat, []).append([last_start, i - 1])
                    last_cat, last_start = cat, i
            tmp_charmap.setdefault(last_cat, []).append([last_start, sys.maxunicode])

            try:
                # Write the Unicode table atomically
                tmpdir = storage_directory("tmp")
                mkdir_p(tmpdir)
                fd, tmpfile = tempfile.mkstemp(dir=tmpdir)
                os.close(fd)
                # Explicitly set the mtime to get reproducible output
                with gzip.GzipFile(tmpfile, "wb", mtime=1) as o:
                    result = json.dumps(sorted(tmp_charmap.items()))
                    o.write(result.encode())

                os.renames(tmpfile, f)
            except Exception:
                pass

        # convert between lists and tuples
        _charmap = {
            k: tuple(tuple(pair) for pair in pairs) for k, pairs in tmp_charmap.items()
        }
        # each value is a tuple of 2-tuples (that is, tuples of length 2)
        # and that both elements of that tuple are integers.
        for vs in _charmap.values():
            ints = list(sum(vs, ()))
            assert all(isinstance(x, int) for x in ints)
            assert ints == sorted(ints)
            assert all(len(tup) == 2 for tup in vs)

    assert _charmap is not None
    return _charmap


_categories = None


def categories():
    """Return a tuple of Unicode categories in a normalised order.

    >>> categories() # doctest: +ELLIPSIS
    ('Zl', 'Zp', 'Co', 'Me', 'Pc', ..., 'Cc', 'Cs')
    """
    global _categories
    if _categories is None:
        cm = charmap()
        _categories = sorted(cm.keys(), key=lambda c: len(cm[c]))
        _categories.remove("Cc")  # Other, Control
        _categories.remove("Cs")  # Other, Surrogate
        _categories.append("Cc")
        _categories.append("Cs")
    return tuple(_categories)


def as_general_categories(cats, name="cats"):
    """Return a tuple of Unicode categories in a normalised order.

    This function expands one-letter designations of a major class to include
    all subclasses:

    >>> as_general_categories(['N'])
    ('Nd', 'Nl', 'No')

    See section 4.5 of the Unicode standard for more on classes:
    https://www.unicode.org/versions/Unicode10.0.0/ch04.pdf

    If the collection ``cats`` includes any elements that do not represent a
    major class or a class with subclass, a deprecation warning is raised.
    """
    if cats is None:
        return None
    major_classes = ("L", "M", "N", "P", "S", "Z", "C")
    cs = categories()
    out = set(cats)
    for c in cats:
        if c in major_classes:
            out.discard(c)
            out.update(x for x in cs if x.startswith(c))
        elif c not in cs:
            raise InvalidArgument(
                f"In {name}={cats!r}, {c!r} is not a valid Unicode category."
            )
    return tuple(c for c in cs if c in out)


def _union_intervals(x, y):
    """Merge two sequences of intervals into a single tuple of intervals.

    Any integer bounded by `x` or `y` is also bounded by the result.

    >>> _union_intervals([(3, 10)], [(1, 2), (5, 17)])
    ((1, 17),)
    """
    if not x:
        return tuple((u, v) for u, v in y)
    if not y:
        return tuple((u, v) for u, v in x)
    intervals = sorted(x + y, reverse=True)
    result = [intervals.pop()]
    while intervals:
        # 1. intervals is in descending order
        # 2. pop() takes from the RHS.
        # 3. (a, b) was popped 1st, then (u, v) was popped 2nd
        # 4. Therefore: a <= u
        # 5. We assume that u <= v and a <= b
        # 6. So we need to handle 2 cases of overlap, and one disjoint case
        #    |   u--v     |   u----v   |       u--v  |
        #    |   a----b   |   a--b     |  a--b       |
        u, v = intervals.pop()
        a, b = result[-1]
        if u <= b + 1:
            # Overlap cases
            result[-1] = (a, max(v, b))
        else:
            # Disjoint case
            result.append((u, v))
    return tuple(result)


def _subtract_intervals(x, y):
    """Set difference for lists of intervals. That is, returns a list of
    intervals that bounds all values bounded by x that are not also bounded by
    y. x and y are expected to be in sorted order.

    For example _subtract_intervals([(1, 10)], [(2, 3), (9, 15)]) would
    return [(1, 1), (4, 8)], removing the values 2, 3, 9 and 10 from the
    interval.
    """
    if not y:
        return tuple(x)
    x = list(map(list, x))
    i = 0
    j = 0
    result = []
    while i < len(x) and j < len(y):
        # Iterate in parallel over x and y. j stays pointing at the smallest
        # interval in the left hand side that could still overlap with some
        # element of x at index >= i.
        # Similarly, i is not incremented until we know that it does not
        # overlap with any element of y at index >= j.

        xl, xr = x[i]
        assert xl <= xr
        yl, yr = y[j]
        assert yl <= yr

        if yr < xl:
            # The interval at y[j] is strictly to the left of the interval at
            # x[i], so will not overlap with it or any later interval of x.
            j += 1
        elif yl > xr:
            # The interval at y[j] is strictly to the right of the interval at
            # x[i], so all of x[i] goes into the result as no further intervals
            # in y will intersect it.
            result.append(x[i])
            i += 1
        elif yl <= xl:
            if yr >= xr:
                # x[i] is contained entirely in y[j], so we just skip over it
                # without adding it to the result.
                i += 1
            else:
                # The beginning of x[i] is contained in y[j], so we update the
                # left endpoint of x[i] to remove this, and increment j as we
                # now have moved past it. Note that this is not added to the
                # result as is, as more intervals from y may intersect it so it
                # may need updating further.
                x[i][0] = yr + 1
                j += 1
        else:
            # yl > xl, so the left hand part of x[i] is not contained in y[j],
            # so there are some values we should add to the result.
            result.append((xl, yl - 1))

            if yr + 1 <= xr:
                # If y[j] finishes before x[i] does, there may be some values
                # in x[i] left that should go in the result (or they may be
                # removed by a later interval in y), so we update x[i] to
                # reflect that and increment j because it no longer overlaps
                # with any remaining element of x.
                x[i][0] = yr + 1
                j += 1
            else:
                # Every element of x[i] other than the initial part we have
                # already added is contained in y[j], so we move to the next
                # interval.
                i += 1
    # Any remaining intervals in x do not overlap with any of y, as if they did
    # we would not have incremented j to the end, so can be added to the result
    # as they are.
    result.extend(x[i:])
    return tuple(map(tuple, result))


def _intervals(s):
    """Return a tuple of intervals, covering the codepoints of characters in
    `s`.

    >>> _intervals('abcdef0123456789')
    ((48, 57), (97, 102))
    """
    intervals = tuple((ord(c), ord(c)) for c in sorted(s))
    return _union_intervals(intervals, intervals)


category_index_cache = {(): ()}


def _category_key(exclude, include):
    """Return a normalised tuple of all Unicode categories that are in
    `include`, but not in `exclude`.

    If include is None then default to including all categories.
    Any item in include that is not a unicode character will be excluded.

    >>> _category_key(exclude=['So'], include=['Lu', 'Me', 'Cs', 'So'])
    ('Me', 'Lu', 'Cs')
    """
    cs = categories()
    if include is None:
        include = set(cs)
    else:
        include = set(include)
    exclude = set(exclude or ())
    assert include.issubset(cs)
    assert exclude.issubset(cs)
    include -= exclude
    return tuple(c for c in cs if c in include)


def _query_for_key(key):
    """Return a tuple of codepoint intervals covering characters that match one
    or more categories in the tuple of categories `key`.

    >>> _query_for_key(categories())
    ((0, 1114111),)
    >>> _query_for_key(('Zl', 'Zp', 'Co'))
    ((8232, 8233), (57344, 63743), (983040, 1048573), (1048576, 1114109))
    """
    try:
        return category_index_cache[key]
    except KeyError:
        pass
    assert key
    if set(key) == set(categories()):
        result = ((0, sys.maxunicode),)
    else:
        result = _union_intervals(_query_for_key(key[:-1]), charmap()[key[-1]])
    category_index_cache[key] = result
    return result


limited_category_index_cache: cache_type = {}


def query(
    exclude_categories=(),
    include_categories=None,
    min_codepoint=None,
    max_codepoint=None,
    include_characters="",
    exclude_characters="",
):
    """Return a tuple of intervals covering the codepoints for all characters
    that meet the criteria (min_codepoint <= codepoint(c) <= max_codepoint and
    any(cat in include_categories for cat in categories(c)) and all(cat not in
    exclude_categories for cat in categories(c)) or (c in include_characters)

    >>> query()
    ((0, 1114111),)
    >>> query(min_codepoint=0, max_codepoint=128)
    ((0, 128),)
    >>> query(min_codepoint=0, max_codepoint=128, include_categories=['Lu'])
    ((65, 90),)
    >>> query(min_codepoint=0, max_codepoint=128, include_categories=['Lu'],
    ...       include_characters=u'☃')
    ((65, 90), (9731, 9731))
    """
    if min_codepoint is None:
        min_codepoint = 0
    if max_codepoint is None:
        max_codepoint = sys.maxunicode
    catkey = _category_key(exclude_categories, include_categories)
    character_intervals = _intervals(include_characters or "")
    exclude_intervals = _intervals(exclude_characters or "")
    qkey = (
        catkey,
        min_codepoint,
        max_codepoint,
        character_intervals,
        exclude_intervals,
    )
    try:
        return limited_category_index_cache[qkey]
    except KeyError:
        pass
    base = _query_for_key(catkey)
    result = []
    for u, v in base:
        if v >= min_codepoint and u <= max_codepoint:
            result.append((max(u, min_codepoint), min(v, max_codepoint)))
    result = tuple(result)
    result = _union_intervals(result, character_intervals)
    result = _subtract_intervals(result, exclude_intervals)
    limited_category_index_cache[qkey] = result
    return result
