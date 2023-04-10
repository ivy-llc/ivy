# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

from collections import defaultdict
from typing import Dict

import attr

from hypothesis.internal.compat import int_from_bytes, int_to_bytes
from hypothesis.internal.conjecture.choicetree import (
    ChoiceTree,
    prefix_selection_order,
    random_selection_order,
)
from hypothesis.internal.conjecture.data import ConjectureResult, Status
from hypothesis.internal.conjecture.floats import (
    DRAW_FLOAT_LABEL,
    float_to_lex,
    lex_to_float,
)
from hypothesis.internal.conjecture.junkdrawer import (
    binary_search,
    find_integer,
    replace_all,
)
from hypothesis.internal.conjecture.shrinking import Float, Integer, Lexical, Ordering
from hypothesis.internal.conjecture.shrinking.learned_dfas import SHRINKING_DFAS


def sort_key(buffer):
    """Returns a sort key such that "simpler" buffers are smaller than
    "more complicated" ones.

    We define sort_key so that x is simpler than y if x is shorter than y or if
    they have the same length and x < y lexicographically. This is called the
    shortlex order.

    The reason for using the shortlex order is:

    1. If x is shorter than y then that means we had to make fewer decisions
       in constructing the test case when we ran x than we did when we ran y.
    2. If x is the same length as y then replacing a byte with a lower byte
       corresponds to reducing the value of an integer we drew with draw_bits
       towards zero.
    3. We want a total order, and given (2) the natural choices for things of
       the same size are either the lexicographic or colexicographic orders
       (the latter being the lexicographic order of the reverse of the string).
       Because values drawn early in generation potentially get used in more
       places they potentially have a more significant impact on the final
       result, so it makes sense to prioritise reducing earlier values over
       later ones. This makes the lexicographic order the more natural choice.
    """
    return (len(buffer), buffer)


SHRINK_PASS_DEFINITIONS: Dict[str, "ShrinkPassDefinition"] = {}


@attr.s()
class ShrinkPassDefinition:
    """A shrink pass bundles together a large number of local changes to
    the current shrink target.

    Each shrink pass is defined by some function and some arguments to that
    function. The ``generate_arguments`` function returns all arguments that
    might be useful to run on the current shrink target.

    The guarantee made by methods defined this way is that after they are
    called then *either* the shrink target has changed *or* each of
    ``fn(*args)`` has been called for every ``args`` in ``generate_arguments(self)``.
    No guarantee is made that all of these will be called if the shrink target
    changes.
    """

    run_with_chooser = attr.ib()

    @property
    def name(self):
        return self.run_with_chooser.__name__

    def __attrs_post_init__(self):
        assert self.name not in SHRINK_PASS_DEFINITIONS, self.name
        SHRINK_PASS_DEFINITIONS[self.name] = self


def defines_shrink_pass():
    """A convenient decorator for defining shrink passes."""

    def accept(run_step):
        ShrinkPassDefinition(run_with_chooser=run_step)

        def run(self):
            raise NotImplementedError("Shrink passes should not be run directly")

        run.__name__ = run_step.__name__
        run.is_shrink_pass = True
        return run

    return accept


class Shrinker:
    """A shrinker is a child object of a ConjectureRunner which is designed to
    manage the associated state of a particular shrink problem. That is, we
    have some initial ConjectureData object and some property of interest
    that it satisfies, and we want to find a ConjectureData object with a
    shortlex (see sort_key above) smaller buffer that exhibits the same
    property.

    Currently the only property of interest we use is that the status is
    INTERESTING and the interesting_origin takes on some fixed value, but we
    may potentially be interested in other use cases later.
    However we assume that data with a status < VALID never satisfies the predicate.

    The shrinker keeps track of a value shrink_target which represents the
    current best known ConjectureData object satisfying the predicate.
    It refines this value by repeatedly running *shrink passes*, which are
    methods that perform a series of transformations to the current shrink_target
    and evaluate the underlying test function to find new ConjectureData
    objects. If any of these satisfy the predicate, the shrink_target
    is updated automatically. Shrinking runs until no shrink pass can
    improve the shrink_target, at which point it stops. It may also be
    terminated if the underlying engine throws RunIsComplete, but that
    is handled by the calling code rather than the Shrinker.

    =======================
    Designing Shrink Passes
    =======================

    Generally a shrink pass is just any function that calls
    cached_test_function and/or incorporate_new_buffer a number of times,
    but there are a couple of useful things to bear in mind.

    A shrink pass *makes progress* if running it changes self.shrink_target
    (i.e. it tries a shortlex smaller ConjectureData object satisfying
    the predicate). The desired end state of shrinking is to find a
    value such that no shrink pass can make progress, i.e. that we
    are at a local minimum for each shrink pass.

    In aid of this goal, the main invariant that a shrink pass much
    satisfy is that whether it makes progress must be deterministic.
    It is fine (encouraged even) for the specific progress it makes
    to be non-deterministic, but if you run a shrink pass, it makes
    no progress, and then you immediately run it again, it should
    never succeed on the second time. This allows us to stop as soon
    as we have run each shrink pass and seen no progress on any of
    them.

    This means that e.g. it's fine to try each of N deletions
    or replacements in a random order, but it's not OK to try N random
    deletions (unless you have already shrunk at least once, though we
    don't currently take advantage of this loophole).

    Shrink passes need to be written so as to be robust against
    change in the underlying shrink target. It is generally safe
    to assume that the shrink target does not change prior to the
    point of first modification - e.g. if you change no bytes at
    index ``i``, all examples whose start is ``<= i`` still exist,
    as do all blocks, and the data object is still of length
    ``>= i + 1``. This can only be violated by bad user code which
    relies on an external source of non-determinism.

    When the underlying shrink_target changes, shrink
    passes should not run substantially more test_function calls
    on success than they do on failure. Say, no more than a constant
    factor more. In particular shrink passes should not iterate to a
    fixed point.

    This means that shrink passes are often written with loops that
    are carefully designed to do the right thing in the case that no
    shrinks occurred and try to adapt to any changes to do a reasonable
    job. e.g. say we wanted to write a shrink pass that tried deleting
    each individual byte (this isn't an especially good choice,
    but it leads to a simple illustrative example), we might do it
    by iterating over the buffer like so:

    .. code-block:: python

        i = 0
        while i < len(self.shrink_target.buffer):
            if not self.incorporate_new_buffer(
                self.shrink_target.buffer[:i] + self.shrink_target.buffer[i + 1 :]
            ):
                i += 1

    The reason for writing the loop this way is that i is always a
    valid index into the current buffer, even if the current buffer
    changes as a result of our actions. When the buffer changes,
    we leave the index where it is rather than restarting from the
    beginning, and carry on. This means that the number of steps we
    run in this case is always bounded above by the number of steps
    we would run if nothing works.

    Another thing to bear in mind about shrink pass design is that
    they should prioritise *progress*. If you have N operations that
    you need to run, you should try to order them in such a way as
    to avoid stalling, where you have long periods of test function
    invocations where no shrinks happen. This is bad because whenever
    we shrink we reduce the amount of work the shrinker has to do
    in future, and often speed up the test function, so we ideally
    wanted those shrinks to happen much earlier in the process.

    Sometimes stalls are inevitable of course - e.g. if the pass
    makes no progress, then the entire thing is just one long stall,
    but it's helpful to design it so that stalls are less likely
    in typical behaviour.

    The two easiest ways to do this are:

    * Just run the N steps in random order. As long as a
      reasonably large proportion of the operations succeed, this
      guarantees the expected stall length is quite short. The
      book keeping for making sure this does the right thing when
      it succeeds can be quite annoying.
    * When you have any sort of nested loop, loop in such a way
      that both loop variables change each time. This prevents
      stalls which occur when one particular value for the outer
      loop is impossible to make progress on, rendering the entire
      inner loop into a stall.

    However, although progress is good, too much progress can be
    a bad sign! If you're *only* seeing successful reductions,
    that's probably a sign that you are making changes that are
    too timid. Two useful things to offset this:

    * It's worth writing shrink passes which are *adaptive*, in
      the sense that when operations seem to be working really
      well we try to bundle multiple of them together. This can
      often be used to turn what would be O(m) successful calls
      into O(log(m)).
    * It's often worth trying one or two special minimal values
      before trying anything more fine grained (e.g. replacing
      the whole thing with zero).

    """

    def derived_value(fn):  # noqa: B902
        """It's useful during shrinking to have access to derived values of
        the current shrink target.

        This decorator allows you to define these as cached properties. They
        are calculated once, then cached until the shrink target changes, then
        recalculated the next time they are used."""

        def accept(self):
            try:
                return self.__derived_values[fn.__name__]
            except KeyError:
                return self.__derived_values.setdefault(fn.__name__, fn(self))

        accept.__name__ = fn.__name__
        return property(accept)

    def __init__(self, engine, initial, predicate, allow_transition):
        """Create a shrinker for a particular engine, with a given starting
        point and predicate. When shrink() is called it will attempt to find an
        example for which predicate is True and which is strictly smaller than
        initial.

        Note that initial is a ConjectureData object, and predicate
        takes ConjectureData objects.
        """
        assert predicate is not None or allow_transition is not None
        self.engine = engine
        self.__predicate = predicate or (lambda data: True)
        self.__allow_transition = allow_transition or (lambda source, destination: True)
        self.__derived_values = {}
        self.__pending_shrink_explanation = None

        self.initial_size = len(initial.buffer)

        # We keep track of the current best example on the shrink_target
        # attribute.
        self.shrink_target = initial
        self.clear_change_tracking()
        self.shrinks = 0

        # We terminate shrinks that seem to have reached their logical
        # conclusion: If we've called the underlying test function at
        # least self.max_stall times since the last time we shrunk,
        # it's time to stop shrinking.
        self.max_stall = 200
        self.initial_calls = self.engine.call_count
        self.calls_at_last_shrink = self.initial_calls

        self.passes_by_name = {}
        self.passes = []

        # Extra DFAs that may be installed. This is used solely for
        # testing and learning purposes.
        self.extra_dfas = {}

    @derived_value  # type: ignore
    def cached_calculations(self):
        return {}

    def cached(self, *keys):
        def accept(f):
            cache_key = (f.__name__,) + keys
            try:
                return self.cached_calculations[cache_key]
            except KeyError:
                return self.cached_calculations.setdefault(cache_key, f())

        return accept

    def add_new_pass(self, run):
        """Creates a shrink pass corresponding to calling ``run(self)``"""

        definition = SHRINK_PASS_DEFINITIONS[run]

        p = ShrinkPass(
            run_with_chooser=definition.run_with_chooser,
            shrinker=self,
            index=len(self.passes),
        )
        self.passes.append(p)
        self.passes_by_name[p.name] = p
        return p

    def shrink_pass(self, name):
        """Return the ShrinkPass object for the pass with the given name."""
        if isinstance(name, ShrinkPass):
            return name
        if name not in self.passes_by_name:
            self.add_new_pass(name)
        return self.passes_by_name[name]

    @derived_value  # type: ignore
    def match_cache(self):
        return {}

    def matching_regions(self, dfa):
        """Returns all pairs (u, v) such that self.buffer[u:v] is accepted
        by this DFA."""

        try:
            return self.match_cache[dfa]
        except KeyError:
            pass
        results = dfa.all_matching_regions(self.buffer)
        results.sort(key=lambda t: (t[1] - t[0], t[1]))
        assert all(dfa.matches(self.buffer[u:v]) for u, v in results)
        self.match_cache[dfa] = results
        return results

    @property
    def calls(self):
        """Return the number of calls that have been made to the underlying
        test function."""
        return self.engine.call_count

    def consider_new_buffer(self, buffer):
        """Returns True if after running this buffer the result would be
        the current shrink_target."""
        buffer = bytes(buffer)
        return buffer.startswith(self.buffer) or self.incorporate_new_buffer(buffer)

    def incorporate_new_buffer(self, buffer):
        """Either runs the test function on this buffer and returns True if
        that changed the shrink_target, or determines that doing so would
        be useless and returns False without running it."""

        buffer = bytes(buffer[: self.shrink_target.index])
        # Sometimes an attempt at lexicographic minimization will do the wrong
        # thing because the buffer has changed under it (e.g. something has
        # turned into a write, the bit size has changed). The result would be
        # an invalid string, but it's better for us to just ignore it here as
        # it turns out to involve quite a lot of tricky book-keeping to get
        # this right and it's better to just handle it in one place.
        if sort_key(buffer) >= sort_key(self.shrink_target.buffer):
            return False

        if self.shrink_target.buffer.startswith(buffer):
            return False

        previous = self.shrink_target
        self.cached_test_function(buffer)
        return previous is not self.shrink_target

    def incorporate_test_data(self, data):
        """Takes a ConjectureData or Overrun object updates the current
        shrink_target if this data represents an improvement over it."""
        if data.status < Status.VALID or data is self.shrink_target:
            return
        if (
            self.__predicate(data)
            and sort_key(data.buffer) < sort_key(self.shrink_target.buffer)
            and self.__allow_transition(self.shrink_target, data)
        ):
            self.update_shrink_target(data)

    def cached_test_function(self, buffer):
        """Returns a cached version of the underlying test function, so
        that the result is either an Overrun object (if the buffer is
        too short to be a valid test case) or a ConjectureData object
        with status >= INVALID that would result from running this buffer."""

        buffer = bytes(buffer)
        result = self.engine.cached_test_function(buffer)
        self.incorporate_test_data(result)
        if self.calls - self.calls_at_last_shrink >= self.max_stall:
            raise StopShrinking()
        return result

    def debug(self, msg):
        self.engine.debug(msg)

    @property
    def random(self):
        return self.engine.random

    def shrink(self):
        """Run the full set of shrinks and update shrink_target.

        This method is "mostly idempotent" - calling it twice is unlikely to
        have any effect, though it has a non-zero probability of doing so.
        """
        # We assume that if an all-zero block of bytes is an interesting
        # example then we're not going to do better than that.
        # This might not technically be true: e.g. for integers() | booleans()
        # the simplest example is actually [1, 0]. Missing this case is fairly
        # harmless and this allows us to make various simplifying assumptions
        # about the structure of the data (principally that we're never
        # operating on a block of all zero bytes so can use non-zeroness as a
        # signpost of complexity).
        if not any(self.shrink_target.buffer) or self.incorporate_new_buffer(
            bytes(len(self.shrink_target.buffer))
        ):
            return

        try:
            self.greedy_shrink()
        except StopShrinking:
            pass
        finally:
            if self.engine.report_debug_info:

                def s(n):
                    return "s" if n != 1 else ""

                total_deleted = self.initial_size - len(self.shrink_target.buffer)
                calls = self.engine.call_count - self.initial_calls

                self.debug(
                    "---------------------\n"
                    "Shrink pass profiling\n"
                    "---------------------\n\n"
                    f"Shrinking made a total of {calls} call{s(calls)} of which "
                    f"{self.shrinks} shrank. This deleted {total_deleted} bytes out "
                    f"of {self.initial_size}."
                )
                for useful in [True, False]:
                    self.debug("")
                    if useful:
                        self.debug("Useful passes:")
                    else:
                        self.debug("Useless passes:")
                    self.debug("")
                    for p in sorted(
                        self.passes, key=lambda t: (-t.calls, t.deletions, t.shrinks)
                    ):
                        if p.calls == 0:
                            continue
                        if (p.shrinks != 0) != useful:
                            continue

                        self.debug(
                            "  * %s made %d call%s of which "
                            "%d shrank, deleting %d byte%s."
                            % (
                                p.name,
                                p.calls,
                                s(p.calls),
                                p.shrinks,
                                p.deletions,
                                s(p.deletions),
                            )
                        )
                self.debug("")

    def greedy_shrink(self):
        """Run a full set of greedy shrinks (that is, ones that will only ever
        move to a better target) and update shrink_target appropriately.

        This method iterates to a fixed point and so is idempontent - calling
        it twice will have exactly the same effect as calling it once.
        """
        self.fixate_shrink_passes(
            [
                block_program("X" * 5),
                block_program("X" * 4),
                block_program("X" * 3),
                block_program("X" * 2),
                block_program("X" * 1),
                "pass_to_descendant",
                "reorder_examples",
                "minimize_floats",
                "minimize_duplicated_blocks",
                block_program("-XX"),
                "minimize_individual_blocks",
                block_program("--X"),
                "redistribute_block_pairs",
                "lower_blocks_together",
            ]
            + [dfa_replacement(n) for n in SHRINKING_DFAS]
        )

    @derived_value  # type: ignore
    def shrink_pass_choice_trees(self):
        return defaultdict(ChoiceTree)

    def fixate_shrink_passes(self, passes):
        """Run steps from each pass in ``passes`` until the current shrink target
        is a fixed point of all of them."""
        passes = list(map(self.shrink_pass, passes))

        any_ran = True
        while any_ran:
            any_ran = False

            reordering = {}

            # We run remove_discarded after every pass to do cleanup
            # keeping track of whether that actually works. Either there is
            # no discarded data and it is basically free, or it reliably works
            # and deletes data, or it doesn't work. In that latter case we turn
            # it off for the rest of this loop through the passes, but will
            # try again once all of the passes have been run.
            can_discard = self.remove_discarded()

            calls_at_loop_start = self.calls

            # We keep track of how many calls can be made by a single step
            # without making progress and use this to test how much to pad
            # out self.max_stall by as we go along.
            max_calls_per_failing_step = 1

            for sp in passes:
                if can_discard:
                    can_discard = self.remove_discarded()

                before_sp = self.shrink_target

                # Run the shrink pass until it fails to make any progress
                # max_failures times in a row. This implicitly boosts shrink
                # passes that are more likely to work.
                failures = 0
                max_failures = 20
                while failures < max_failures:
                    # We don't allow more than max_stall consecutive failures
                    # to shrink, but this means that if we're unlucky and the
                    # shrink passes are in a bad order where only the ones at
                    # the end are useful, if we're not careful this heuristic
                    # might stop us before we've tried everything. In order to
                    # avoid that happening, we make sure that there's always
                    # plenty of breathing room to make it through a single
                    # iteration of the fixate_shrink_passes loop.
                    self.max_stall = max(
                        self.max_stall,
                        2 * max_calls_per_failing_step
                        + (self.calls - calls_at_loop_start),
                    )

                    prev = self.shrink_target
                    initial_calls = self.calls
                    # It's better for us to run shrink passes in a deterministic
                    # order, to avoid repeat work, but this can cause us to create
                    # long stalls when there are a lot of steps which fail to do
                    # anything useful. In order to avoid this, once we've noticed
                    # we're in a stall (i.e. half of max_failures calls have failed
                    # to do anything) we switch to randomly jumping around. If we
                    # find a success then we'll resume deterministic order from
                    # there which, with any luck, is in a new good region.
                    if not sp.step(random_order=failures >= max_failures // 2):
                        # step returns False when there is nothing to do because
                        # the entire choice tree is exhausted. If this happens
                        # we break because we literally can't run this pass any
                        # more than we already have until something else makes
                        # progress.
                        break
                    any_ran = True

                    # Don't count steps that didn't actually try to do
                    # anything as failures. Otherwise, this call is a failure
                    # if it failed to make any changes to the shrink target.
                    if initial_calls != self.calls:
                        if prev is not self.shrink_target:
                            failures = 0
                        else:
                            max_calls_per_failing_step = max(
                                max_calls_per_failing_step, self.calls - initial_calls
                            )
                            failures += 1

                # We reorder the shrink passes so that on our next run through
                # we try good ones first. The rule is that shrink passes that
                # did nothing useful are the worst, shrink passes that reduced
                # the length are the best.
                if self.shrink_target is before_sp:
                    reordering[sp] = 1
                elif len(self.buffer) < len(before_sp.buffer):
                    reordering[sp] = -1
                else:
                    reordering[sp] = 0

            passes.sort(key=reordering.__getitem__)

    @property
    def buffer(self):
        return self.shrink_target.buffer

    @property
    def blocks(self):
        return self.shrink_target.blocks

    @property
    def examples(self):
        return self.shrink_target.examples

    def all_block_bounds(self):
        return self.shrink_target.blocks.all_bounds()

    @derived_value  # type: ignore
    def examples_by_label(self):
        """An index of all examples grouped by their label, with
        the examples stored in their normal index order."""

        examples_by_label = defaultdict(list)
        for ex in self.examples:
            examples_by_label[ex.label].append(ex)
        return dict(examples_by_label)

    @derived_value  # type: ignore
    def distinct_labels(self):
        return sorted(self.examples_by_label, key=str)

    @defines_shrink_pass()
    def pass_to_descendant(self, chooser):
        """Attempt to replace each example with a descendant example.

        This is designed to deal with strategies that call themselves
        recursively. For example, suppose we had:

        binary_tree = st.deferred(
            lambda: st.one_of(
                st.integers(), st.tuples(binary_tree, binary_tree)))

        This pass guarantees that we can replace any binary tree with one of
        its subtrees - each of those will create an interval that the parent
        could validly be replaced with, and this pass will try doing that.

        This is pretty expensive - it takes O(len(intervals)^2) - so we run it
        late in the process when we've got the number of intervals as far down
        as possible.
        """

        label = chooser.choose(
            self.distinct_labels, lambda l: len(self.examples_by_label[l]) >= 2
        )

        ls = self.examples_by_label[label]

        i = chooser.choose(range(len(ls) - 1))

        ancestor = ls[i]

        if i + 1 == len(ls) or ls[i + 1].start >= ancestor.end:
            return

        @self.cached(label, i)
        def descendants():
            lo = i + 1
            hi = len(ls)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if ls[mid].start >= ancestor.end:
                    hi = mid
                else:
                    lo = mid
            return [t for t in ls[i + 1 : hi] if t.length < ancestor.length]

        descendant = chooser.choose(descendants, lambda ex: ex.length > 0)

        assert ancestor.start <= descendant.start
        assert ancestor.end >= descendant.end
        assert descendant.length < ancestor.length

        self.incorporate_new_buffer(
            self.buffer[: ancestor.start]
            + self.buffer[descendant.start : descendant.end]
            + self.buffer[ancestor.end :]
        )

    def lower_common_block_offset(self):
        """Sometimes we find ourselves in a situation where changes to one part
        of the byte stream unlock changes to other parts. Sometimes this is
        good, but sometimes this can cause us to exhibit exponential slow
        downs!

        e.g. suppose we had the following:

        m = draw(integers(min_value=0))
        n = draw(integers(min_value=0))
        assert abs(m - n) > 1

        If this fails then we'll end up with a loop where on each iteration we
        reduce each of m and n by 2 - m can't go lower because of n, then n
        can't go lower because of m.

        This will take us O(m) iterations to complete, which is exponential in
        the data size, as we gradually zig zag our way towards zero.

        This can only happen if we're failing to reduce the size of the byte
        stream: The number of iterations that reduce the length of the byte
        stream is bounded by that length.

        So what we do is this: We keep track of which blocks are changing, and
        then if there's some non-zero common offset to them we try and minimize
        them all at once by lowering that offset.

        This may not work, and it definitely won't get us out of all possible
        exponential slow downs (an example of where it doesn't is where the
        shape of the blocks changes as a result of this bouncing behaviour),
        but it fails fast when it doesn't work and gets us out of a really
        nastily slow case when it does.
        """
        if len(self.__changed_blocks) <= 1:
            return

        current = self.shrink_target

        blocked = [current.buffer[u:v] for u, v in self.all_block_bounds()]

        changed = [
            i
            for i in sorted(self.__changed_blocks)
            if not self.shrink_target.blocks[i].trivial
        ]

        if not changed:
            return

        ints = [int_from_bytes(blocked[i]) for i in changed]
        offset = min(ints)
        assert offset > 0

        for i in range(len(ints)):
            ints[i] -= offset

        def reoffset(o):
            new_blocks = list(blocked)
            for i, v in zip(changed, ints):
                new_blocks[i] = int_to_bytes(v + o, len(blocked[i]))
            return self.incorporate_new_buffer(b"".join(new_blocks))

        Integer.shrink(offset, reoffset, random=self.random)
        self.clear_change_tracking()

    def clear_change_tracking(self):
        self.__last_checked_changed_at = self.shrink_target
        self.__all_changed_blocks = set()

    def mark_changed(self, i):
        self.__changed_blocks.add(i)

    @property
    def __changed_blocks(self):
        if self.__last_checked_changed_at is not self.shrink_target:
            prev_target = self.__last_checked_changed_at
            new_target = self.shrink_target
            assert prev_target is not new_target
            prev = prev_target.buffer
            new = new_target.buffer
            assert sort_key(new) < sort_key(prev)

            if (
                len(new_target.blocks) != len(prev_target.blocks)
                or new_target.blocks.endpoints != prev_target.blocks.endpoints
            ):
                self.__all_changed_blocks = set()
            else:
                blocks = new_target.blocks

                # Index of last block whose contents have been modified, found
                # by checking if the tail past this point has been modified.
                last_changed = binary_search(
                    0,
                    len(blocks),
                    lambda i: prev[blocks.start(i) :] != new[blocks.start(i) :],
                )

                # Index of the first block whose contents have been changed,
                # because we know that this predicate is true for zero (because
                # the prefix from the start is empty), so the result must be True
                # for the bytes from the start of this block and False for the
                # bytes from the end, hence the change is in this block.
                first_changed = binary_search(
                    0,
                    len(blocks),
                    lambda i: prev[: blocks.start(i)] == new[: blocks.start(i)],
                )

                # Between these two changed regions we now do a linear scan to
                # check if any specific block values have changed.
                for i in range(first_changed, last_changed + 1):
                    u, v = blocks.bounds(i)
                    if i not in self.__all_changed_blocks and prev[u:v] != new[u:v]:
                        self.__all_changed_blocks.add(i)
            self.__last_checked_changed_at = new_target
        assert self.__last_checked_changed_at is self.shrink_target
        return self.__all_changed_blocks

    def update_shrink_target(self, new_target):
        assert isinstance(new_target, ConjectureResult)
        self.shrinks += 1
        # If we are just taking a long time to shrink we don't want to
        # trigger this heuristic, so whenever we shrink successfully
        # we give ourselves a bit of breathing room to make sure we
        # would find a shrink that took that long to find the next time.
        # The case where we're taking a long time but making steady
        # progress is handled by `finish_shrinking_deadline` in engine.py
        self.max_stall = max(
            self.max_stall, (self.calls - self.calls_at_last_shrink) * 2
        )
        self.calls_at_last_shrink = self.calls
        self.shrink_target = new_target
        self.__derived_values = {}

    def try_shrinking_blocks(self, blocks, b):
        """Attempts to replace each block in the blocks list with b. Returns
        True if it succeeded (which may include some additional modifications
        to shrink_target).

        In current usage it is expected that each of the blocks currently have
        the same value, although this is not essential. Note that b must be
        < the block at min(blocks) or this is not a valid shrink.

        This method will attempt to do some small amount of work to delete data
        that occurs after the end of the blocks. This is useful for cases where
        there is some size dependency on the value of a block.
        """
        initial_attempt = bytearray(self.shrink_target.buffer)
        for i, block in enumerate(blocks):
            if block >= len(self.blocks):
                blocks = blocks[:i]
                break
            u, v = self.blocks[block].bounds
            n = min(self.blocks[block].length, len(b))
            initial_attempt[v - n : v] = b[-n:]

        if not blocks:
            return False

        start = self.shrink_target.blocks[blocks[0]].start
        end = self.shrink_target.blocks[blocks[-1]].end

        initial_data = self.cached_test_function(initial_attempt)

        if initial_data is self.shrink_target:
            self.lower_common_block_offset()
            return True

        # If this produced something completely invalid we ditch it
        # here rather than trying to persevere.
        if initial_data.status < Status.VALID:
            return False

        # We've shrunk inside our group of blocks, so we have no way to
        # continue. (This only happens when shrinking more than one block at
        # a time).
        if len(initial_data.buffer) < v:
            return False

        lost_data = len(self.shrink_target.buffer) - len(initial_data.buffer)

        # If this did not in fact cause the data size to shrink we
        # bail here because it's not worth trying to delete stuff from
        # the remainder.
        if lost_data <= 0:
            return False

        # We now look for contiguous regions to delete that might help fix up
        # this failed shrink. We only look for contiguous regions of the right
        # lengths because doing anything more than that starts to get very
        # expensive. See minimize_individual_blocks for where we
        # try to be more aggressive.
        regions_to_delete = {(end, end + lost_data)}

        for j in (blocks[-1] + 1, blocks[-1] + 2):
            if j >= min(len(initial_data.blocks), len(self.blocks)):
                continue
            # We look for a block very shortly after the last one that has
            # lost some of its size, and try to delete from the beginning so
            # that it retains the same integer value. This is a bit of a hyper
            # specific trick designed to make our integers() strategy shrink
            # well.
            r1, s1 = self.shrink_target.blocks[j].bounds
            r2, s2 = initial_data.blocks[j].bounds
            lost = (s1 - r1) - (s2 - r2)
            # Apparently a coverage bug? An assert False in the body of this
            # will reliably fail, but it shows up as uncovered.
            if lost <= 0 or r1 != r2:  # pragma: no cover
                continue
            regions_to_delete.add((r1, r1 + lost))

        for ex in self.shrink_target.examples:
            if ex.start > start:
                continue
            if ex.end <= end:
                continue

            replacement = initial_data.examples[ex.index]

            in_original = [c for c in ex.children if c.start >= end]

            in_replaced = [c for c in replacement.children if c.start >= end]

            if len(in_replaced) >= len(in_original) or not in_replaced:
                continue

            # We've found an example where some of the children went missing
            # as a result of this change, and just replacing it with the data
            # it would have had and removing the spillover didn't work. This
            # means that some of its children towards the right must be
            # important, so we try to arrange it so that it retains its
            # rightmost children instead of its leftmost.
            regions_to_delete.add(
                (in_original[0].start, in_original[-len(in_replaced)].start)
            )

        for u, v in sorted(regions_to_delete, key=lambda x: x[1] - x[0], reverse=True):
            try_with_deleted = bytearray(initial_attempt)
            del try_with_deleted[u:v]
            if self.incorporate_new_buffer(try_with_deleted):
                return True
        return False

    def remove_discarded(self):
        """Try removing all bytes marked as discarded.

        This is primarily to deal with data that has been ignored while
        doing rejection sampling - e.g. as a result of an integer range, or a
        filtered strategy.

        Such data will also be handled by the adaptive_example_deletion pass,
        but that pass is necessarily more conservative and will try deleting
        each interval individually. The common case is that all data drawn and
        rejected can just be thrown away immediately in one block, so this pass
        will be much faster than trying each one individually when it works.

        returns False if there is discarded data and removing it does not work,
        otherwise returns True.
        """
        while self.shrink_target.has_discards:
            discarded = []

            for ex in self.shrink_target.examples:
                if (
                    ex.length > 0
                    and ex.discarded
                    and (not discarded or ex.start >= discarded[-1][-1])
                ):
                    discarded.append((ex.start, ex.end))

            # This can happen if we have discards but they are all of
            # zero length. This shouldn't happen very often so it's
            # faster to check for it here than at the point of example
            # generation.
            if not discarded:
                break

            attempt = bytearray(self.shrink_target.buffer)
            for u, v in reversed(discarded):
                del attempt[u:v]

            if not self.incorporate_new_buffer(attempt):
                return False
        return True

    @derived_value  # type: ignore
    def blocks_by_non_zero_suffix(self):
        """Returns a list of blocks grouped by their non-zero suffix,
        as a list of (suffix, indices) pairs, skipping all groupings
        where there is only one index.

        This is only used for the arguments of minimize_duplicated_blocks.
        """
        duplicates = defaultdict(list)
        for block in self.blocks:
            duplicates[non_zero_suffix(self.buffer[block.start : block.end])].append(
                block.index
            )
        return duplicates

    @derived_value  # type: ignore
    def duplicated_block_suffixes(self):
        return sorted(self.blocks_by_non_zero_suffix)

    @defines_shrink_pass()
    def minimize_duplicated_blocks(self, chooser):
        """Find blocks that have been duplicated in multiple places and attempt
        to minimize all of the duplicates simultaneously.

        This lets us handle cases where two values can't be shrunk
        independently of each other but can easily be shrunk together.
        For example if we had something like:

        ls = data.draw(lists(integers()))
        y = data.draw(integers())
        assert y not in ls

        Suppose we drew y = 3 and after shrinking we have ls = [3]. If we were
        to replace both 3s with 0, this would be a valid shrink, but if we were
        to replace either 3 with 0 on its own the test would start passing.

        It is also useful for when that duplication is accidental and the value
        of the blocks doesn't matter very much because it allows us to replace
        more values at once.
        """
        block = chooser.choose(self.duplicated_block_suffixes)
        targets = self.blocks_by_non_zero_suffix[block]
        if len(targets) <= 1:
            return
        Lexical.shrink(
            block,
            lambda b: self.try_shrinking_blocks(targets, b),
            random=self.random,
            full=False,
        )

    @defines_shrink_pass()
    def minimize_floats(self, chooser):
        """Some shrinks that we employ that only really make sense for our
        specific floating point encoding that are hard to discover from any
        sort of reasonable general principle. This allows us to make
        transformations like replacing a NaN with an Infinity or replacing
        a float with its nearest integers that we would otherwise not be
        able to due to them requiring very specific transformations of
        the bit sequence.

        We only apply these transformations to blocks that "look like" our
        standard float encodings because they are only really meaningful
        there. The logic for detecting this is reasonably precise, but
        it doesn't matter if it's wrong. These are always valid
        transformations to make, they just don't necessarily correspond to
        anything particularly meaningful for non-float values.
        """

        ex = chooser.choose(
            self.examples,
            lambda ex: (
                ex.label == DRAW_FLOAT_LABEL
                and len(ex.children) == 2
                and ex.children[1].length == 8
            ),
        )

        u = ex.children[1].start
        v = ex.children[1].end
        buf = self.shrink_target.buffer
        b = buf[u:v]
        f = lex_to_float(int_from_bytes(b))
        b2 = int_to_bytes(float_to_lex(f), 8)
        if b == b2 or self.consider_new_buffer(buf[:u] + b2 + buf[v:]):
            Float.shrink(
                f,
                lambda x: self.consider_new_buffer(
                    self.shrink_target.buffer[:u]
                    + int_to_bytes(float_to_lex(x), 8)
                    + self.shrink_target.buffer[v:]
                ),
                random=self.random,
            )

    @defines_shrink_pass()
    def redistribute_block_pairs(self, chooser):
        """If there is a sum of generated integers that we need their sum
        to exceed some bound, lowering one of them requires raising the
        other. This pass enables that."""

        block = chooser.choose(self.blocks, lambda b: not b.all_zero)

        for j in range(block.index + 1, len(self.blocks)):
            next_block = self.blocks[j]
            if next_block.length == block.length:
                break
        else:
            return

        buffer = self.buffer

        m = int_from_bytes(buffer[block.start : block.end])
        n = int_from_bytes(buffer[next_block.start : next_block.end])

        def boost(k):
            if k > m:
                return False
            attempt = bytearray(buffer)
            attempt[block.start : block.end] = int_to_bytes(m - k, block.length)
            try:
                attempt[next_block.start : next_block.end] = int_to_bytes(
                    n + k, next_block.length
                )
            except OverflowError:
                return False
            return self.consider_new_buffer(attempt)

        find_integer(boost)

    @defines_shrink_pass()
    def lower_blocks_together(self, chooser):
        block = chooser.choose(self.blocks, lambda b: not b.all_zero)

        # Choose the next block to be up to eight blocks onwards. We don't
        # want to go too far (to avoid quadratic time) but it's worth a
        # reasonable amount of lookahead, especially as we expect most
        # blocks are zero by this point anyway.
        next_block = self.blocks[
            chooser.choose(
                range(block.index + 1, min(len(self.blocks), block.index + 9)),
                lambda j: not self.blocks[j].all_zero,
            )
        ]

        buffer = self.buffer

        m = int_from_bytes(buffer[block.start : block.end])
        n = int_from_bytes(buffer[next_block.start : next_block.end])

        def lower(k):
            if k > min(m, n):
                return False
            attempt = bytearray(buffer)
            attempt[block.start : block.end] = int_to_bytes(m - k, block.length)
            attempt[next_block.start : next_block.end] = int_to_bytes(
                n - k, next_block.length
            )
            assert len(attempt) == len(buffer)
            return self.consider_new_buffer(attempt)

        find_integer(lower)

    @defines_shrink_pass()
    def minimize_individual_blocks(self, chooser):
        """Attempt to minimize each block in sequence.

        This is the pass that ensures that e.g. each integer we draw is a
        minimum value. So it's the part that guarantees that if we e.g. do

        x = data.draw(integers())
        assert x < 10

        then in our shrunk example, x = 10 rather than say 97.

        If we are unsuccessful at minimizing a block of interest we then
        check if that's because it's changing the size of the test case and,
        if so, we also make an attempt to delete parts of the test case to
        see if that fixes it.

        We handle most of the common cases in try_shrinking_blocks which is
        pretty good at clearing out large contiguous blocks of dead space,
        but it fails when there is data that has to stay in particular places
        in the list.
        """
        block = chooser.choose(self.blocks, lambda b: not b.trivial)

        initial = self.shrink_target
        u, v = block.bounds
        i = block.index
        Lexical.shrink(
            self.shrink_target.buffer[u:v],
            lambda b: self.try_shrinking_blocks((i,), b),
            random=self.random,
            full=False,
        )

        if self.shrink_target is not initial:
            return

        lowered = (
            self.buffer[: block.start]
            + int_to_bytes(
                int_from_bytes(self.buffer[block.start : block.end]) - 1, block.length
            )
            + self.buffer[block.end :]
        )
        attempt = self.cached_test_function(lowered)
        if (
            attempt.status < Status.VALID
            or len(attempt.buffer) == len(self.buffer)
            or len(attempt.buffer) == block.end
        ):
            return

        # If it were then the lexical shrink should have worked and we could
        # never have got here.
        assert attempt is not self.shrink_target

        @self.cached(block.index)
        def first_example_after_block():
            lo = 0
            hi = len(self.examples)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                ex = self.examples[mid]
                if ex.start >= block.end:
                    hi = mid
                else:
                    lo = mid
            return hi

        ex = self.examples[
            chooser.choose(
                range(first_example_after_block, len(self.examples)),
                lambda i: self.examples[i].length > 0,
            )
        ]

        u, v = block.bounds

        buf = bytearray(lowered)
        del buf[ex.start : ex.end]
        self.incorporate_new_buffer(buf)

    @defines_shrink_pass()
    def reorder_examples(self, chooser):
        """This pass allows us to reorder the children of each example.

        For example, consider the following:

        .. code-block:: python

            import hypothesis.strategies as st
            from hypothesis import given


            @given(st.text(), st.text())
            def test_not_equal(x, y):
                assert x != y

        Without the ability to reorder x and y this could fail either with
        ``x=""``, ``y="0"``, or the other way around. With reordering it will
        reliably fail with ``x=""``, ``y="0"``.
        """
        ex = chooser.choose(self.examples)
        label = chooser.choose(ex.children).label

        group = [c for c in ex.children if c.label == label]
        if len(group) <= 1:
            return

        st = self.shrink_target
        pieces = [st.buffer[ex.start : ex.end] for ex in group]
        endpoints = [(ex.start, ex.end) for ex in group]

        Ordering.shrink(
            pieces,
            lambda ls: self.consider_new_buffer(
                replace_all(st.buffer, [(u, v, r) for (u, v), r in zip(endpoints, ls)])
            ),
            random=self.random,
        )

    def run_block_program(self, i, description, original, repeats=1):
        """Block programs are a mini-DSL for block rewriting, defined as a sequence
        of commands that can be run at some index into the blocks

        Commands are:

            * "-", subtract one from this block.
            * "X", delete this block

        If a command does not apply (currently only because it's - on a zero
        block) the block will be silently skipped over.

        This method runs the block program in ``description`` at block index
        ``i`` on the ConjectureData ``original``. If ``repeats > 1`` then it
        will attempt to approximate the results of running it that many times.

        Returns True if this successfully changes the underlying shrink target,
        else False.
        """
        if i + len(description) > len(original.blocks) or i < 0:
            return False
        attempt = bytearray(original.buffer)
        for _ in range(repeats):
            for k, d in reversed(list(enumerate(description))):
                j = i + k
                u, v = original.blocks[j].bounds
                if v > len(attempt):
                    return False
                if d == "-":
                    value = int_from_bytes(attempt[u:v])
                    if value == 0:
                        return False
                    else:
                        attempt[u:v] = int_to_bytes(value - 1, v - u)
                elif d == "X":
                    del attempt[u:v]
                else:
                    raise NotImplementedError(f"Unrecognised command {d!r}")
        return self.incorporate_new_buffer(attempt)


def shrink_pass_family(f):
    def accept(*args):
        name = "{}({})".format(f.__name__, ", ".join(map(repr, args)))
        if name not in SHRINK_PASS_DEFINITIONS:

            def run(self, chooser):
                return f(self, chooser, *args)

            run.__name__ = name
            defines_shrink_pass()(run)
        assert name in SHRINK_PASS_DEFINITIONS
        return name

    return accept


@shrink_pass_family
def block_program(self, chooser, description):
    """Mini-DSL for block rewriting. A sequence of commands that will be run
    over all contiguous sequences of blocks of the description length in order.
    Commands are:

        * ".", keep this block unchanged
        * "-", subtract one from this block.
        * "0", replace this block with zero
        * "X", delete this block

    If a command does not apply (currently only because it's - on a zero
    block) the block will be silently skipped over. As a side effect of
    running a block program its score will be updated.
    """
    n = len(description)

    """Adaptively attempt to run the block program at the current
    index. If this successfully applies the block program ``k`` times
    then this runs in ``O(log(k))`` test function calls."""
    i = chooser.choose(range(len(self.shrink_target.blocks) - n))
    # First, run the block program at the chosen index. If this fails,
    # don't do any extra work, so that failure is as cheap as possible.
    if not self.run_block_program(i, description, original=self.shrink_target):
        return

    # Because we run in a random order we will often find ourselves in the middle
    # of a region where we could run the block program. We thus start by moving
    # left to the beginning of that region if possible in order to to start from
    # the beginning of that region.
    def offset_left(k):
        return i - k * n

    i = offset_left(
        find_integer(
            lambda k: self.run_block_program(
                offset_left(k), description, original=self.shrink_target
            )
        )
    )

    original = self.shrink_target

    # Now try to run the block program multiple times here.
    find_integer(
        lambda k: self.run_block_program(i, description, original=original, repeats=k)
    )


@shrink_pass_family
def dfa_replacement(self, chooser, dfa_name):
    """Use one of our previously learned shrinking DFAs to reduce
    the current test case. This works by finding a match of the DFA in the
    current buffer that is not already minimal and attempting to replace it
    with the minimal string matching that DFA.
    """

    try:
        dfa = SHRINKING_DFAS[dfa_name]
    except KeyError:
        dfa = self.extra_dfas[dfa_name]

    matching_regions = self.matching_regions(dfa)
    minimal = next(dfa.all_matching_strings())
    u, v = chooser.choose(
        matching_regions, lambda t: self.buffer[t[0] : t[1]] != minimal
    )
    p = self.buffer[u:v]
    assert sort_key(minimal) < sort_key(p)
    replaced = self.buffer[:u] + minimal + self.buffer[v:]

    assert sort_key(replaced) < sort_key(self.buffer)

    self.consider_new_buffer(replaced)


@attr.s(slots=True, eq=False)
class ShrinkPass:
    run_with_chooser = attr.ib()
    index = attr.ib()
    shrinker = attr.ib()

    last_prefix = attr.ib(default=())
    successes = attr.ib(default=0)
    calls = attr.ib(default=0)
    shrinks = attr.ib(default=0)
    deletions = attr.ib(default=0)

    def step(self, random_order=False):
        tree = self.shrinker.shrink_pass_choice_trees[self]
        if tree.exhausted:
            return False

        initial_shrinks = self.shrinker.shrinks
        initial_calls = self.shrinker.calls
        size = len(self.shrinker.shrink_target.buffer)
        self.shrinker.engine.explain_next_call_as(self.name)

        if random_order:
            selection_order = random_selection_order(self.shrinker.random)
        else:
            selection_order = prefix_selection_order(self.last_prefix)

        try:
            self.last_prefix = tree.step(
                selection_order,
                lambda chooser: self.run_with_chooser(self.shrinker, chooser),
            )
        finally:
            self.calls += self.shrinker.calls - initial_calls
            self.shrinks += self.shrinker.shrinks - initial_shrinks
            self.deletions += size - len(self.shrinker.shrink_target.buffer)
            self.shrinker.engine.clear_call_explanation()
        return True

    @property
    def name(self):
        return self.run_with_chooser.__name__


def non_zero_suffix(b):
    """Returns the longest suffix of b that starts with a non-zero
    byte."""
    i = 0
    while i < len(b) and b[i] == 0:
        i += 1
    return b[i:]


class StopShrinking(Exception):
    pass
