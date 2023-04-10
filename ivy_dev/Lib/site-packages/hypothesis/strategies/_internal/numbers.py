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
from decimal import Decimal
from fractions import Fraction
from sys import float_info
from typing import Optional, Union

from hypothesis.control import reject
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import floats as flt, utils as d
from hypothesis.internal.conjecture.utils import calc_label_from_name
from hypothesis.internal.filtering import (
    get_float_predicate_bounds,
    get_integer_predicate_bounds,
)
from hypothesis.internal.floats import (
    float_of,
    float_to_int,
    int_to_float,
    is_negative,
    make_float_clamper,
    next_down,
    next_down_normal,
    next_up,
    next_up_normal,
    width_smallest_normals,
)
from hypothesis.internal.validation import (
    check_type,
    check_valid_bound,
    check_valid_interval,
)
from hypothesis.strategies._internal.misc import nothing
from hypothesis.strategies._internal.strategies import (
    SampledFromStrategy,
    SearchStrategy,
)
from hypothesis.strategies._internal.utils import cacheable, defines_strategy

# See https://github.com/python/mypy/issues/3186 - numbers.Real is wrong!
Real = Union[int, float, Fraction, Decimal]
ONE_BOUND_INTEGERS_LABEL = d.calc_label_from_name("trying a one-bound int allowing 0")


class IntegersStrategy(SearchStrategy):
    def __init__(self, start, end):
        assert isinstance(start, int) or start is None
        assert isinstance(end, int) or end is None
        assert start is None or end is None or start <= end
        self.start = start
        self.end = end

    def __repr__(self):
        if self.start is None and self.end is None:
            return "integers()"
        if self.end is None:
            return f"integers(min_value={self.start})"
        if self.start is None:
            return f"integers(max_value={self.end})"
        return f"integers({self.start}, {self.end})"

    def do_draw(self, data):
        if self.start is None and self.end is None:
            return d.unbounded_integers(data)

        if self.start is None:
            if self.end <= 0:
                return self.end - abs(d.unbounded_integers(data))
            else:
                probe = self.end + 1
                while self.end < probe:
                    data.start_example(ONE_BOUND_INTEGERS_LABEL)
                    probe = d.unbounded_integers(data)
                    data.stop_example(discard=self.end < probe)
                return probe

        if self.end is None:
            if self.start >= 0:
                return self.start + abs(d.unbounded_integers(data))
            else:
                probe = self.start - 1
                while probe < self.start:
                    data.start_example(ONE_BOUND_INTEGERS_LABEL)
                    probe = d.unbounded_integers(data)
                    data.stop_example(discard=probe < self.start)
                return probe

        # For bounded integers, make the bounds and near-bounds more likely.
        forced = None
        if self.end - self.start > 127:
            forced = {
                122: self.start,
                123: self.start,
                124: self.end,
                125: self.end,
                126: self.start + 1,
                127: self.end - 1,
            }.get(data.draw_bits(7))

        return d.integer_range(data, self.start, self.end, center=0, forced=forced)

    def filter(self, condition):
        if condition is math.isfinite:
            return self
        if condition in [math.isinf, math.isnan]:
            return nothing()
        kwargs, pred = get_integer_predicate_bounds(condition)

        start, end = self.start, self.end
        if "min_value" in kwargs:
            start = max(kwargs["min_value"], -math.inf if start is None else start)
        if "max_value" in kwargs:
            end = min(kwargs["max_value"], math.inf if end is None else end)

        if start != self.start or end != self.end:
            if start is not None and end is not None and start > end:
                return nothing()
            self = type(self)(start, end)
        if pred is None:
            return self
        return super().filter(pred)


@cacheable
@defines_strategy(force_reusable_values=True)
def integers(
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> SearchStrategy[int]:
    """Returns a strategy which generates integers.

    If min_value is not None then all values will be >= min_value. If
    max_value is not None then all values will be <= max_value

    Examples from this strategy will shrink towards zero, and negative values
    will also shrink towards positive (i.e. -n may be replaced by +n).
    """
    check_valid_bound(min_value, "min_value")
    check_valid_bound(max_value, "max_value")
    check_valid_interval(min_value, max_value, "min_value", "max_value")

    if min_value is not None:
        if min_value != int(min_value):
            raise InvalidArgument(
                "min_value=%r of type %r cannot be exactly represented as an integer."
                % (min_value, type(min_value))
            )
        min_value = int(min_value)
    if max_value is not None:
        if max_value != int(max_value):
            raise InvalidArgument(
                "max_value=%r of type %r cannot be exactly represented as an integer."
                % (max_value, type(max_value))
            )
        max_value = int(max_value)

    return IntegersStrategy(min_value, max_value)


SMALLEST_SUBNORMAL = next_up(0.0)
SIGNALING_NAN = int_to_float(0x7FF8_0000_0000_0001)  # nonzero mantissa
assert math.isnan(SIGNALING_NAN) and math.copysign(1, SIGNALING_NAN) == 1

NASTY_FLOATS = sorted(
    [
        0.0,
        0.5,
        1.1,
        1.5,
        1.9,
        1.0 / 3,
        10e6,
        10e-6,
        1.175494351e-38,
        next_up(0.0),
        float_info.min,
        float_info.max,
        3.402823466e38,
        9007199254740992,
        1 - 10e-6,
        2 + 10e-6,
        1.192092896e-07,
        2.2204460492503131e-016,
    ]
    + [2.0**-n for n in (24, 14, 149, 126)]  # minimum (sub)normals for float16,32
    + [float_info.min / n for n in (2, 10, 1000, 100_000)]  # subnormal in float64
    + [math.inf, math.nan] * 5
    + [SIGNALING_NAN],
    key=flt.float_to_lex,
)
NASTY_FLOATS = list(map(float, NASTY_FLOATS))
NASTY_FLOATS.extend([-x for x in NASTY_FLOATS])

FLOAT_STRATEGY_DO_DRAW_LABEL = calc_label_from_name(
    "getting another float in FloatStrategy"
)


def _sign_aware_lte(x: float, y: float) -> bool:
    """Less-than-or-equals, but strictly orders -0.0 and 0.0"""
    if x == 0.0 == y:
        return math.copysign(1.0, x) <= math.copysign(1.0, y)
    else:
        return x <= y


class FloatStrategy(SearchStrategy):
    """A strategy for floating point numbers."""

    def __init__(
        self,
        *,
        min_value: float,
        max_value: float,
        allow_nan: bool,
        # The smallest nonzero number we can represent is usually a subnormal, but may
        # be the smallest normal if we're running in unsafe denormals-are-zero mode.
        # While that's usually an explicit error, we do need to handle the case where
        # the user passes allow_subnormal=False.
        smallest_nonzero_magnitude: float = SMALLEST_SUBNORMAL,
    ):
        super().__init__()
        assert isinstance(allow_nan, bool)
        assert smallest_nonzero_magnitude >= 0.0, "programmer error if this is negative"
        if smallest_nonzero_magnitude == 0.0:  # pragma: no cover
            raise FloatingPointError(
                "Got allow_subnormal=True, but we can't represent subnormal floats "
                "right now, in violation of the IEEE-754 floating-point "
                "specification.  This is usually because something was compiled with "
                "-ffast-math or a similar option, which sets global processor state.  "
                "See https://simonbyrne.github.io/notes/fastmath/ for a more detailed "
                "writeup - and good luck!"
            )
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.smallest_nonzero_magnitude = smallest_nonzero_magnitude

        boundary_values = [
            min_value,
            next_up(min_value),
            min_value + 1,
            max_value - 1,
            next_down(max_value),
            max_value,
        ]
        self.nasty_floats = [
            f for f in NASTY_FLOATS + boundary_values if self.permitted(f)
        ]
        weights = [0.2 * len(self.nasty_floats)] + [0.8] * len(self.nasty_floats)
        self.sampler = d.Sampler(weights) if self.nasty_floats else None

        self.pos_clamper = self.neg_clamper = None
        if _sign_aware_lte(0.0, max_value):
            pos_min = max(min_value, smallest_nonzero_magnitude)
            allow_zero = _sign_aware_lte(min_value, 0.0)
            self.pos_clamper = make_float_clamper(pos_min, max_value, allow_zero)
        if _sign_aware_lte(min_value, -0.0):
            neg_max = min(max_value, -smallest_nonzero_magnitude)
            allow_zero = _sign_aware_lte(-0.0, max_value)
            self.neg_clamper = make_float_clamper(-neg_max, -min_value, allow_zero)

        self.forced_sign_bit: Optional[int] = None
        if (self.pos_clamper is None) != (self.neg_clamper is None):
            self.forced_sign_bit = 1 if self.neg_clamper else 0

    def __repr__(self):
        return "{}(min_value={}, max_value={}, allow_nan={}, smallest_nonzero_magnitude={})".format(
            self.__class__.__name__,
            self.min_value,
            self.max_value,
            self.allow_nan,
            self.smallest_nonzero_magnitude,
        )

    def permitted(self, f):
        assert isinstance(f, float)
        if math.isnan(f):
            return self.allow_nan
        if 0 < abs(f) < self.smallest_nonzero_magnitude:
            return False
        return _sign_aware_lte(self.min_value, f) and _sign_aware_lte(f, self.max_value)

    def do_draw(self, data):
        while True:
            data.start_example(FLOAT_STRATEGY_DO_DRAW_LABEL)
            i = self.sampler.sample(data) if self.sampler else 0
            data.start_example(flt.DRAW_FLOAT_LABEL)
            if i == 0:
                result = flt.draw_float(data, forced_sign_bit=self.forced_sign_bit)
                is_negative = flt.float_to_int(result) >> 63
                if is_negative:
                    clamped = -self.neg_clamper(-result)
                else:
                    clamped = self.pos_clamper(result)
                if clamped != result:
                    data.stop_example(discard=True)
                    data.start_example(flt.DRAW_FLOAT_LABEL)
                    flt.write_float(data, clamped)
                    result = clamped
            else:
                result = self.nasty_floats[i - 1]

                flt.write_float(data, result)

            data.stop_example()  # (DRAW_FLOAT_LABEL)
            data.stop_example()  # (FLOAT_STRATEGY_DO_DRAW_LABEL)
            return result

    def filter(self, condition):
        # Handle a few specific weird cases.
        if condition is math.isfinite:
            return FloatStrategy(
                min_value=max(self.min_value, next_up(float("-inf"))),
                max_value=min(self.max_value, next_down(float("inf"))),
                allow_nan=False,
                smallest_nonzero_magnitude=self.smallest_nonzero_magnitude,
            )
        if condition is math.isinf:
            permitted_infs = [x for x in (-math.inf, math.inf) if self.permitted(x)]
            if not permitted_infs:
                return nothing()
            return SampledFromStrategy(permitted_infs)
        if condition is math.isnan:
            if not self.allow_nan:
                return nothing()
            return NanStrategy()

        kwargs, pred = get_float_predicate_bounds(condition)
        if not kwargs:
            return super().filter(pred)
        min_bound = max(kwargs.get("min_value", -math.inf), self.min_value)
        max_bound = min(kwargs.get("max_value", math.inf), self.max_value)

        # Adjustments for allow_subnormal=False, if any need to be made
        if -self.smallest_nonzero_magnitude < min_bound < 0:
            min_bound = -0.0
        elif 0 < min_bound < self.smallest_nonzero_magnitude:
            min_bound = self.smallest_nonzero_magnitude
        if -self.smallest_nonzero_magnitude < max_bound < 0:
            max_bound = -self.smallest_nonzero_magnitude
        elif 0 < max_bound < self.smallest_nonzero_magnitude:
            max_bound = 0.0

        if min_bound > max_bound:
            return nothing()
        if (
            min_bound > self.min_value
            or self.max_value > max_bound
            or (self.allow_nan and (-math.inf < min_bound or max_bound < math.inf))
        ):
            self = type(self)(
                min_value=min_bound,
                max_value=max_bound,
                allow_nan=False,
                smallest_nonzero_magnitude=self.smallest_nonzero_magnitude,
            )
        if pred is None:
            return self
        return super().filter(pred)


@cacheable
@defines_strategy(force_reusable_values=True)
def floats(
    min_value: Optional[Real] = None,
    max_value: Optional[Real] = None,
    *,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    allow_subnormal: Optional[bool] = None,
    width: int = 64,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> SearchStrategy[float]:
    """Returns a strategy which generates floats.

    - If min_value is not None, all values will be ``>= min_value``
      (or ``> min_value`` if ``exclude_min``).
    - If max_value is not None, all values will be ``<= max_value``
      (or ``< max_value`` if ``exclude_max``).
    - If min_value or max_value is not None, it is an error to enable
      allow_nan.
    - If both min_value and max_value are not None, it is an error to enable
      allow_infinity.
    - If inferred values range does not include subnormal values, it is an error
      to enable allow_subnormal.

    Where not explicitly ruled out by the bounds,
    :wikipedia:`subnormals <Subnormal_number>`, infinities, and NaNs are possible
    values generated by this strategy.

    The width argument specifies the maximum number of bits of precision
    required to represent the generated float. Valid values are 16, 32, or 64.
    Passing ``width=32`` will still use the builtin 64-bit :class:`~python:float` class,
    but always for values which can be exactly represented as a 32-bit float.

    The exclude_min and exclude_max argument can be used to generate numbers
    from open or half-open intervals, by excluding the respective endpoints.
    Excluding either signed zero will also exclude the other.
    Attempting to exclude an endpoint which is None will raise an error;
    use ``allow_infinity=False`` to generate finite floats.  You can however
    use e.g. ``min_value=-math.inf, exclude_min=True`` to exclude only
    one infinite endpoint.

    Examples from this strategy have a complicated and hard to explain
    shrinking behaviour, but it tries to improve "human readability". Finite
    numbers will be preferred to infinity and infinity will be preferred to
    NaN.
    """
    check_type(bool, exclude_min, "exclude_min")
    check_type(bool, exclude_max, "exclude_max")

    if allow_nan is None:
        allow_nan = bool(min_value is None and max_value is None)
    elif allow_nan and (min_value is not None or max_value is not None):
        raise InvalidArgument(
            f"Cannot have allow_nan={allow_nan!r}, with min_value or max_value"
        )

    if width not in (16, 32, 64):
        raise InvalidArgument(
            f"Got width={width!r}, but the only valid values "
            "are the integers 16, 32, and 64."
        )

    check_valid_bound(min_value, "min_value")
    check_valid_bound(max_value, "max_value")

    if math.copysign(1.0, -0.0) == 1.0:  # pragma: no cover
        raise FloatingPointError(
            "You Python install can't represent -0.0, which is required by the "
            "IEEE-754 floating-point specification.  This is probably because it was "
            "compiled with an unsafe option like -ffast-math; for a more detailed "
            "explanation see https://simonbyrne.github.io/notes/fastmath/"
        )
    if allow_subnormal and next_up(0.0, width=width) == 0:  # pragma: no cover
        # Not worth having separate CI envs and dependencies just to cover this branch;
        # discussion in https://github.com/HypothesisWorks/hypothesis/issues/3092
        #
        # Erroring out here ensures that the database contents are interpreted
        # consistently - which matters for such a foundational strategy, even if it's
        # not always true for all user-composed strategies further up the stack.
        from _hypothesis_ftz_detector import identify_ftz_culprits

        try:
            ftz_pkg = identify_ftz_culprits()
        except Exception:
            ftz_pkg = None
        if ftz_pkg:
            ftz_msg = (
                f"This seems to be because the `{ftz_pkg}` package was compiled with "
                f"-ffast-math or a similar option, which sets global processor state "
                f"- see https://simonbyrne.github.io/notes/fastmath/ for details.  "
                f"If you don't know why {ftz_pkg} is installed, `pipdeptree -rp "
                f"{ftz_pkg}` will show which packages depend on it."
            )
        else:
            ftz_msg = (
                "This is usually because something was compiled with -ffast-math "
                "or a similar option, which sets global processor state.  See "
                "https://simonbyrne.github.io/notes/fastmath/ for a more detailed "
                "writeup - and good luck!"
            )
        raise FloatingPointError(
            f"Got allow_subnormal={allow_subnormal!r}, but we can't represent "
            f"subnormal floats right now, in violation of the IEEE-754 floating-point "
            f"specification.  {ftz_msg}"
        )

    min_arg, max_arg = min_value, max_value
    if min_value is not None:
        min_value = float_of(min_value, width)
        assert isinstance(min_value, float)
    if max_value is not None:
        max_value = float_of(max_value, width)
        assert isinstance(max_value, float)

    if min_value != min_arg:
        raise InvalidArgument(
            f"min_value={min_arg!r} cannot be exactly represented as a float "
            f"of width {width} - use min_value={min_value!r} instead."
        )
    if max_value != max_arg:
        raise InvalidArgument(
            f"max_value={max_arg!r} cannot be exactly represented as a float "
            f"of width {width} - use max_value={max_value!r} instead."
        )

    if exclude_min and (min_value is None or min_value == math.inf):
        raise InvalidArgument(f"Cannot exclude min_value={min_value!r}")
    if exclude_max and (max_value is None or max_value == -math.inf):
        raise InvalidArgument(f"Cannot exclude max_value={max_value!r}")

    assumed_allow_subnormal = allow_subnormal is None or allow_subnormal
    if min_value is not None and (
        exclude_min or (min_arg is not None and min_value < min_arg)
    ):
        min_value = next_up_normal(min_value, width, assumed_allow_subnormal)
        if min_value == min_arg:
            assert min_value == min_arg == 0
            assert is_negative(min_arg) and not is_negative(min_value)
            min_value = next_up_normal(min_value, width, assumed_allow_subnormal)
        assert min_value > min_arg  # type: ignore
    if max_value is not None and (
        exclude_max or (max_arg is not None and max_value > max_arg)
    ):
        max_value = next_down_normal(max_value, width, assumed_allow_subnormal)
        if max_value == max_arg:
            assert max_value == max_arg == 0
            assert is_negative(max_value) and not is_negative(max_arg)
            max_value = next_down_normal(max_value, width, assumed_allow_subnormal)
        assert max_value < max_arg  # type: ignore

    if min_value == -math.inf:
        min_value = None
    if max_value == math.inf:
        max_value = None

    bad_zero_bounds = (
        min_value == max_value == 0
        and is_negative(max_value)
        and not is_negative(min_value)
    )
    if (
        min_value is not None
        and max_value is not None
        and (min_value > max_value or bad_zero_bounds)
    ):
        # This is a custom alternative to check_valid_interval, because we want
        # to include the bit-width and exclusion information in the message.
        msg = (
            "There are no %s-bit floating-point values between min_value=%r "
            "and max_value=%r" % (width, min_arg, max_arg)
        )
        if exclude_min or exclude_max:
            msg += f", exclude_min={exclude_min!r} and exclude_max={exclude_max!r}"
        raise InvalidArgument(msg)

    if allow_infinity is None:
        allow_infinity = bool(min_value is None or max_value is None)
    elif allow_infinity:
        if min_value is not None and max_value is not None:
            raise InvalidArgument(
                f"Cannot have allow_infinity={allow_infinity!r}, "
                "with both min_value and max_value"
            )
    elif min_value == math.inf:
        if min_arg == math.inf:
            raise InvalidArgument("allow_infinity=False excludes min_value=inf")
        raise InvalidArgument(
            f"exclude_min=True turns min_value={min_arg!r} into inf, "
            "but allow_infinity=False"
        )
    elif max_value == -math.inf:
        if max_arg == -math.inf:
            raise InvalidArgument("allow_infinity=False excludes max_value=-inf")
        raise InvalidArgument(
            f"exclude_max=True turns max_value={max_arg!r} into -inf, "
            "but allow_infinity=False"
        )

    smallest_normal = width_smallest_normals[width]
    if allow_subnormal is None:
        if min_value is not None and max_value is not None:
            if min_value == max_value:
                allow_subnormal = -smallest_normal < min_value < smallest_normal
            else:
                allow_subnormal = (
                    min_value < smallest_normal and max_value > -smallest_normal
                )
        elif min_value is not None:
            allow_subnormal = min_value < smallest_normal
        elif max_value is not None:
            allow_subnormal = max_value > -smallest_normal
        else:
            allow_subnormal = True
    if allow_subnormal:
        if min_value is not None and min_value >= smallest_normal:
            raise InvalidArgument(
                f"allow_subnormal=True, but minimum value {min_value} "
                f"excludes values below float{width}'s "
                f"smallest positive normal {smallest_normal}"
            )
        if max_value is not None and max_value <= -smallest_normal:
            raise InvalidArgument(
                f"allow_subnormal=True, but maximum value {max_value} "
                f"excludes values above float{width}'s "
                f"smallest negative normal {-smallest_normal}"
            )

    if min_value is None:
        min_value = float("-inf")
    if max_value is None:
        max_value = float("inf")
    if not allow_infinity:
        min_value = max(min_value, next_up(float("-inf")))
        max_value = min(max_value, next_down(float("inf")))
    assert isinstance(min_value, float)
    assert isinstance(max_value, float)
    smallest_nonzero_magnitude = (
        SMALLEST_SUBNORMAL if allow_subnormal else smallest_normal
    )
    result: SearchStrategy = FloatStrategy(
        min_value=min_value,
        max_value=max_value,
        allow_nan=allow_nan,
        smallest_nonzero_magnitude=smallest_nonzero_magnitude,
    )

    if width < 64:

        def downcast(x):
            try:
                return float_of(x, width)
            except OverflowError:  # pragma: no cover
                reject()

        result = result.map(downcast)
    return result


class NanStrategy(SearchStrategy):
    """Strategy for sampling the space of nan float values."""

    def do_draw(self, data):
        # Nans must have all exponent bits and the first mantissa bit set, so
        # we generate by taking 64 random bits and setting the required ones.
        sign_bit = data.draw_bits(1) << 63
        nan_bits = float_to_int(math.nan)
        mantissa_bits = data.draw_bits(52)
        return int_to_float(sign_bit | nan_bits | mantissa_bits)
