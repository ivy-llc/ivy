# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import enum
import hashlib
import heapq
import math
import sys
from collections import OrderedDict, abc
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from hypothesis.errors import InvalidArgument
from hypothesis.internal.compat import floor, int_from_bytes
from hypothesis.internal.floats import int_to_float, next_up

if TYPE_CHECKING:
    from hypothesis.internal.conjecture.data import ConjectureData


LABEL_MASK = 2**64 - 1


def calc_label_from_name(name: str) -> int:
    hashed = hashlib.sha384(name.encode()).digest()
    return int_from_bytes(hashed[:8])


def calc_label_from_cls(cls: type) -> int:
    return calc_label_from_name(cls.__qualname__)


def combine_labels(*labels: int) -> int:
    label = 0
    for l in labels:
        label = (label << 1) & LABEL_MASK
        label ^= l
    return label


INTEGER_RANGE_DRAW_LABEL = calc_label_from_name("another draw in integer_range()")
BIASED_COIN_LABEL = calc_label_from_name("biased_coin()")
BIASED_COIN_INNER_LABEL = calc_label_from_name("inside biased_coin()")
SAMPLE_IN_SAMPLER_LABEL = calc_label_from_name("a sample() in Sampler")
ONE_FROM_MANY_LABEL = calc_label_from_name("one more from many()")


def unbounded_integers(data: "ConjectureData") -> int:
    size = INT_SIZES[INT_SIZES_SAMPLER.sample(data)]
    r = data.draw_bits(size)
    sign = r & 1
    r >>= 1
    if sign:
        r = -r
    return int(r)


def integer_range(
    data: "ConjectureData",
    lower: int,
    upper: int,
    center: Optional[int] = None,
    forced: Optional[int] = None,
) -> int:
    assert lower <= upper
    assert forced is None or lower <= forced <= upper
    if lower == upper:
        # Write a value even when this is trivial so that when a bound depends
        # on other values we don't suddenly disappear when the gap shrinks to
        # zero - if that happens then often the data stream becomes misaligned
        # and we fail to shrink in cases where we really should be able to.
        data.draw_bits(1, forced=0)
        return int(lower)

    if center is None:
        center = lower
    center = min(max(center, lower), upper)

    if center == upper:
        above = False
    elif center == lower:
        above = True
    else:
        force_above = None if forced is None else forced < center
        above = not data.draw_bits(1, forced=force_above)

    if above:
        gap = upper - center
    else:
        gap = center - lower

    assert gap > 0

    bits = gap.bit_length()
    probe = gap + 1

    if bits > 24 and data.draw_bits(3, forced=None if forced is None else 0):
        # For large ranges, we combine the uniform random distribution from draw_bits
        # with a weighting scheme with moderate chance.  Cutoff at 2 ** 24 so that our
        # choice of unicode characters is uniform but the 32bit distribution is not.
        idx = INT_SIZES_SAMPLER.sample(data)
        bits = min(bits, INT_SIZES[idx])

    while probe > gap:
        data.start_example(INTEGER_RANGE_DRAW_LABEL)
        probe = data.draw_bits(
            bits, forced=None if forced is None else abs(forced - center)
        )
        data.stop_example(discard=probe > gap)

    if above:
        result = center + probe
    else:
        result = center - probe

    assert lower <= result <= upper
    assert forced is None or result == forced, (result, forced, center, above)
    return result


T = TypeVar("T")


def check_sample(
    values: Union[Type[enum.Enum], Sequence[T]], strategy_name: str
) -> Sequence[T]:
    if "numpy" in sys.modules and isinstance(values, sys.modules["numpy"].ndarray):
        if values.ndim != 1:
            raise InvalidArgument(
                "Only one-dimensional arrays are supported for sampling, "
                f"and the given value has {values.ndim} dimensions (shape "
                f"{values.shape}).  This array would give samples of array slices "
                "instead of elements!  Use np.ravel(values) to convert "
                "to a one-dimensional array, or tuple(values) if you "
                "want to sample slices."
            )
    elif not isinstance(values, (OrderedDict, abc.Sequence, enum.EnumMeta)):
        raise InvalidArgument(
            f"Cannot sample from {values!r}, not an ordered collection. "
            f"Hypothesis goes to some length to ensure that the {strategy_name} "
            "strategy has stable results between runs. To replay a saved "
            "example, the sampled values must have the same iteration order "
            "on every run - ruling out sets, dicts, etc due to hash "
            "randomization. Most cases can simply use `sorted(values)`, but "
            "mixed types or special values such as math.nan require careful "
            "handling - and note that when simplifying an example, "
            "Hypothesis treats earlier values as simpler."
        )
    if isinstance(values, range):
        return values
    return tuple(values)


def choice(data: "ConjectureData", values: Sequence[T]) -> T:
    return values[integer_range(data, 0, len(values) - 1)]


FLOAT_PREFIX = 0b1111111111 << 52
FULL_FLOAT = int_to_float(FLOAT_PREFIX | ((2 << 53) - 1)) - 1


def fractional_float(data: "ConjectureData") -> float:
    return (int_to_float(FLOAT_PREFIX | data.draw_bits(52)) - 1) / FULL_FLOAT


def biased_coin(
    data: "ConjectureData", p: float, *, forced: Optional[bool] = None
) -> bool:
    """Return True with probability p (assuming a uniform generator),
    shrinking towards False. If ``forced`` is set to a non-None value, this
    will always return that value but will write choices appropriate to having
    drawn that value randomly."""

    # NB this function is vastly more complicated than it may seem reasonable
    # for it to be. This is because it is used in a lot of places and it's
    # important for it to shrink well, so it's worth the engineering effort.

    if p <= 0 or p >= 1:
        bits = 1
    else:
        # When there is a meaningful draw, in order to shrink well we will
        # set things up so that 0 and 1 always correspond to False and True
        # respectively. This means we want enough bits available that in a
        # draw we will always have at least one truthy value and one falsey
        # value.
        bits = math.ceil(-math.log(min(p, 1 - p), 2))
    # In order to avoid stupidly large draws where the probability is
    # effectively zero or one, we treat probabilities of under 2^-64 to be
    # effectively zero.
    if bits > 64:
        # There isn't enough precision near one for this to occur for values
        # far from 0.
        p = 0.0
        bits = 1

    size = 2**bits

    data.start_example(BIASED_COIN_LABEL)
    while True:
        # The logic here is a bit complicated and special cased to make it
        # play better with the shrinker.

        # We imagine partitioning the real interval [0, 1] into 2**n equal parts
        # and looking at each part and whether its interior is wholly <= p
        # or wholly >= p. At most one part can be neither.

        # We then pick a random part. If it's wholly on one side or the other
        # of p then we use that as the answer. If p is contained in the
        # interval then we start again with a new probability that is given
        # by the fraction of that interval that was <= our previous p.

        # We then take advantage of the fact that we have control of the
        # labelling to make this shrink better, using the following tricks:

        # If p is <= 0 or >= 1 the result of this coin is certain. We make sure
        # to write a byte to the data stream anyway so that these don't cause
        # difficulties when shrinking.
        if p <= 0:
            data.draw_bits(1, forced=0)
            result = False
        elif p >= 1:
            data.draw_bits(1, forced=1)
            result = True
        else:
            falsey = floor(size * (1 - p))
            truthy = floor(size * p)
            remainder = size * p - truthy

            if falsey + truthy == size:
                partial = False
            else:
                partial = True

            if forced is None:
                # We want to get to the point where True is represented by
                # 1 and False is represented by 0 as quickly as possible, so
                # we use the remove_discarded machinery in the shrinker to
                # achieve that by discarding any draws that are > 1 and writing
                # a suitable draw into the choice sequence at the end of the
                # loop.
                data.start_example(BIASED_COIN_INNER_LABEL)
                i = data.draw_bits(bits)
                data.stop_example(discard=i > 1)
            else:
                i = data.draw_bits(bits, forced=int(forced))

            # We always choose the region that causes us to repeat the loop as
            # the maximum value, so that shrinking the drawn bits never causes
            # us to need to draw more data.
            if partial and i == size - 1:
                p = remainder
                continue
            if falsey == 0:
                # Every other partition is truthy, so the result is true
                result = True
            elif truthy == 0:
                # Every other partition is falsey, so the result is false
                result = False
            elif i <= 1:
                # We special case so that zero is always false and 1 is always
                # true which makes shrinking easier because we can always
                # replace a truthy block with 1. This has the slightly weird
                # property that shrinking from 2 to 1 can cause the result to
                # grow, but the shrinker always tries 0 and 1 first anyway, so
                # this will usually be fine.
                result = bool(i)
            else:
                # Originally everything in the region 0 <= i < falsey was false
                # and everything above was true. We swapped one truthy element
                # into this region, so the region becomes 0 <= i <= falsey
                # except for i = 1. We know i > 1 here, so the test for truth
                # becomes i > falsey.
                result = i > falsey

            if i > 1:  # pragma: no branch
                # Thanks to bytecode optimisations on CPython >= 3.7 and PyPy
                # (see https://bugs.python.org/issue2506), coverage incorrectly
                # thinks that this condition is always true.  You can trivially
                # check by adding `else: assert False` and running the tests.
                data.draw_bits(bits, forced=int(result))
        break
    data.stop_example()
    return result


class Sampler:
    """Sampler based on Vose's algorithm for the alias method. See
    http://www.keithschwarz.com/darts-dice-coins/ for a good explanation.

    The general idea is that we store a table of triples (base, alternate, p).
    base. We then pick a triple uniformly at random, and choose its alternate
    value with probability p and else choose its base value. The triples are
    chosen so that the resulting mixture has the right distribution.

    We maintain the following invariants to try to produce good shrinks:

    1. The table is in lexicographic (base, alternate) order, so that choosing
       an earlier value in the list always lowers (or at least leaves
       unchanged) the value.
    2. base[i] < alternate[i], so that shrinking the draw always results in
       shrinking the chosen element.
    """

    table: List[Tuple[int, int, float]]  # (base_idx, alt_idx, alt_chance)

    def __init__(self, weights: Sequence[float]):
        n = len(weights)

        table: "list[list[int | float | None]]" = [[i, None, None] for i in range(n)]

        total = sum(weights)

        num_type = type(total)

        zero = num_type(0)  # type: ignore
        one = num_type(1)  # type: ignore

        small: "List[int]" = []
        large: "List[int]" = []

        probabilities = [w / total for w in weights]
        scaled_probabilities: "List[float]" = []

        for i, alternate_chance in enumerate(probabilities):
            scaled = alternate_chance * n
            scaled_probabilities.append(scaled)
            if scaled == 1:
                table[i][2] = zero
            elif scaled < 1:
                small.append(i)
            else:
                large.append(i)
        heapq.heapify(small)
        heapq.heapify(large)

        while small and large:
            lo = heapq.heappop(small)
            hi = heapq.heappop(large)

            assert lo != hi
            assert scaled_probabilities[hi] > one
            assert table[lo][1] is None
            table[lo][1] = hi
            table[lo][2] = one - scaled_probabilities[lo]
            scaled_probabilities[hi] = (
                scaled_probabilities[hi] + scaled_probabilities[lo]
            ) - one

            if scaled_probabilities[hi] < 1:
                heapq.heappush(small, hi)
            elif scaled_probabilities[hi] == 1:
                table[hi][2] = zero
            else:
                heapq.heappush(large, hi)
        while large:
            table[large.pop()][2] = zero
        while small:
            table[small.pop()][2] = zero

        self.table: "List[Tuple[int, int, float]]" = []
        for base, alternate, alternate_chance in table:  # type: ignore
            assert isinstance(base, int)
            assert isinstance(alternate, int) or alternate is None
            if alternate is None:
                self.table.append((base, base, alternate_chance))
            elif alternate < base:
                self.table.append((alternate, base, one - alternate_chance))
            else:
                self.table.append((base, alternate, alternate_chance))
        self.table.sort()

    def sample(self, data: "ConjectureData") -> int:
        data.start_example(SAMPLE_IN_SAMPLER_LABEL)
        base, alternate, alternate_chance = choice(data, self.table)
        use_alternate = biased_coin(data, alternate_chance)
        data.stop_example()
        if use_alternate:
            return alternate
        else:
            return base


INT_SIZES = (8, 16, 32, 64, 128)
INT_SIZES_SAMPLER = Sampler((4.0, 8.0, 1.0, 1.0, 0.5))


class many:
    """Utility class for collections. Bundles up the logic we use for "should I
    keep drawing more values?" and handles starting and stopping examples in
    the right place.

    Intended usage is something like:

    elements = many(data, ...)
    while elements.more():
        add_stuff_to_result()
    """

    def __init__(
        self,
        data: "ConjectureData",
        min_size: int,
        max_size: Union[int, float],
        average_size: Union[int, float],
    ) -> None:
        assert 0 <= min_size <= average_size <= max_size
        self.min_size = min_size
        self.max_size = max_size
        self.data = data
        self.p_continue = _calc_p_continue(average_size - min_size, max_size - min_size)
        self.count = 0
        self.rejections = 0
        self.drawn = False
        self.force_stop = False
        self.rejected = False

    def more(self) -> bool:
        """Should I draw another element to add to the collection?"""
        if self.drawn:
            self.data.stop_example(discard=self.rejected)

        self.drawn = True
        self.rejected = False

        self.data.start_example(ONE_FROM_MANY_LABEL)

        if self.min_size == self.max_size:
            should_continue = self.count < self.min_size
        else:
            forced_result = None
            if self.force_stop:
                forced_result = False
            elif self.count < self.min_size:
                forced_result = True
            elif self.count >= self.max_size:
                forced_result = False
            should_continue = biased_coin(
                self.data, self.p_continue, forced=forced_result
            )

        if should_continue:
            self.count += 1
            return True
        else:
            self.data.stop_example()
            return False

    def reject(self):
        """Reject the last example (i.e. don't count it towards our budget of
        elements because it's not going to go in the final collection)."""
        assert self.count > 0
        self.count -= 1
        self.rejections += 1
        self.rejected = True
        # We set a minimum number of rejections before we give up to avoid
        # failing too fast when we reject the first draw.
        if self.rejections > max(3, 2 * self.count):
            if self.count < self.min_size:
                self.data.mark_invalid()
            else:
                self.force_stop = True


SMALLEST_POSITIVE_FLOAT: float = next_up(0.0) or sys.float_info.min


@lru_cache()
def _calc_p_continue(desired_avg: float, max_size: int) -> float:
    """Return the p_continue which will generate the desired average size."""
    assert desired_avg <= max_size, (desired_avg, max_size)
    if desired_avg == max_size:
        return 1.0
    p_continue = 1 - 1.0 / (1 + desired_avg)
    if p_continue == 0 or max_size == float("inf"):
        assert 0 <= p_continue < 1, p_continue
        return p_continue
    assert 0 < p_continue < 1, p_continue
    # For small max_size, the infinite-series p_continue is a poor approximation,
    # and while we can't solve the polynomial a few rounds of iteration quickly
    # gets us a good approximate solution in almost all cases (sometimes exact!).
    while _p_continue_to_avg(p_continue, max_size) > desired_avg:
        # This is impossible over the reals, but *can* happen with floats.
        p_continue -= 0.0001
        # If we've reached zero or gone negative, we want to break out of this loop,
        # and do so even if we're on a system with the unsafe denormals-are-zero flag.
        # We make that an explicit error in st.floats(), but here we'd prefer to
        # just get somewhat worse precision on collection lengths.
        if p_continue < SMALLEST_POSITIVE_FLOAT:
            p_continue = SMALLEST_POSITIVE_FLOAT
            break
    # Let's binary-search our way to a better estimate!  We tried fancier options
    # like gradient descent, but this is numerically stable and works better.
    hi = 1.0
    while desired_avg - _p_continue_to_avg(p_continue, max_size) > 0.01:
        assert 0 < p_continue < hi, (p_continue, hi)
        mid = (p_continue + hi) / 2
        if _p_continue_to_avg(mid, max_size) <= desired_avg:
            p_continue = mid
        else:
            hi = mid
    assert 0 < p_continue < 1, p_continue
    assert _p_continue_to_avg(p_continue, max_size) <= desired_avg
    return p_continue


def _p_continue_to_avg(p_continue: float, max_size: int) -> float:
    """Return the average_size generated by this p_continue and max_size."""
    if p_continue >= 1:
        return max_size
    return (1.0 / (1 - p_continue) - 1) * (1 - p_continue**max_size)
