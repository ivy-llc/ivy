# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import attr

from hypothesis.errors import Flaky, HypothesisException, StopTest
from hypothesis.internal.compat import int_to_bytes
from hypothesis.internal.conjecture.data import (
    ConjectureData,
    DataObserver,
    Status,
    bits_to_bytes,
)
from hypothesis.internal.conjecture.junkdrawer import IntList


class PreviouslyUnseenBehaviour(HypothesisException):
    pass


def inconsistent_generation():
    raise Flaky(
        "Inconsistent data generation! Data generation behaved differently "
        "between different runs. Is your data generation depending on external "
        "state?"
    )


EMPTY: frozenset = frozenset()


@attr.s(slots=True)
class Killed:
    """Represents a transition to part of the tree which has been marked as
    "killed", meaning we want to treat it as not worth exploring, so it will
    be treated as if it were completely explored for the purposes of
    exhaustion."""

    next_node = attr.ib()


@attr.s(slots=True)
class Branch:
    """Represents a transition where multiple choices can be made as to what
    to drawn."""

    bit_length = attr.ib()
    children = attr.ib(repr=False)

    @property
    def max_children(self):
        return 1 << self.bit_length


@attr.s(slots=True, frozen=True)
class Conclusion:
    """Represents a transition to a finished state."""

    status = attr.ib()
    interesting_origin = attr.ib()


@attr.s(slots=True)
class TreeNode:
    """Node in a tree that corresponds to previous interactions with
    a ``ConjectureData`` object according to some fixed test function.

    This is functionally a variant patricia trie.
    See https://en.wikipedia.org/wiki/Radix_tree for the general idea,
    but what this means in particular here is that we have a very deep
    but very lightly branching tree and rather than store this as a fully
    recursive structure we flatten prefixes and long branches into
    lists. This significantly compacts the storage requirements.

    A single ``TreeNode`` corresponds to a previously seen sequence
    of calls to ``ConjectureData`` which we have never seen branch,
    followed by a ``transition`` which describes what happens next.
    """

    # Records the previous sequence of calls to ``data.draw_bits``,
    # with the ``n_bits`` argument going in ``bit_lengths`` and the
    # values seen in ``values``. These should always have the same
    # length.
    bit_lengths = attr.ib(default=attr.Factory(IntList))
    values = attr.ib(default=attr.Factory(IntList))

    # The indices of of the calls to ``draw_bits`` that we have stored
    # where  ``forced`` is not None. Stored as None if no indices
    # have been forced, purely for space saving reasons (we force
    # quite rarely).
    __forced = attr.ib(default=None, init=False)

    # What happens next after observing this sequence of calls.
    # Either:
    #
    # * ``None``, indicating we don't know yet.
    # * A ``Branch`` object indicating that there is a ``draw_bits``
    #   call that we have seen take multiple outcomes there.
    # * A ``Conclusion`` object indicating that ``conclude_test``
    #   was called here.
    transition = attr.ib(default=None)

    # A tree node is exhausted if every possible sequence of
    # draws below it has been explored. We store this information
    # on a field and update it when performing operations that
    # could change the answer.
    #
    # A node may start exhausted, e.g. because it it leads
    # immediately to a conclusion, but can only go from
    # non-exhausted to exhausted when one of its children
    # becomes exhausted or it is marked as a conclusion.
    #
    # Therefore we only need to check whether we need to update
    # this field when the node is first created in ``split_at``
    # or when we have walked a path through this node to a
    # conclusion in ``TreeRecordingObserver``.
    is_exhausted = attr.ib(default=False, init=False)

    @property
    def forced(self):
        if not self.__forced:
            return EMPTY
        return self.__forced

    def mark_forced(self, i):
        """Note that the value at index ``i`` was forced."""
        assert 0 <= i < len(self.values)
        if self.__forced is None:
            self.__forced = set()
        self.__forced.add(i)

    def split_at(self, i):
        """Splits the tree so that it can incorporate
        a decision at the ``draw_bits`` call corresponding
        to position ``i``, or raises ``Flaky`` if that was
        meant to be a forced node."""

        if i in self.forced:
            inconsistent_generation()

        assert not self.is_exhausted

        key = self.values[i]

        child = TreeNode(
            bit_lengths=self.bit_lengths[i + 1 :],
            values=self.values[i + 1 :],
            transition=self.transition,
        )
        self.transition = Branch(bit_length=self.bit_lengths[i], children={key: child})
        if self.__forced is not None:
            child.__forced = {j - i - 1 for j in self.__forced if j > i}
            self.__forced = {j for j in self.__forced if j < i}
        child.check_exhausted()
        del self.values[i:]
        del self.bit_lengths[i:]
        assert len(self.values) == len(self.bit_lengths) == i

    def check_exhausted(self):
        """Recalculates ``self.is_exhausted`` if necessary then returns
        it."""
        if (
            not self.is_exhausted
            and len(self.forced) == len(self.values)
            and self.transition is not None
        ):
            if isinstance(self.transition, (Conclusion, Killed)):
                self.is_exhausted = True
            elif len(self.transition.children) == self.transition.max_children:
                self.is_exhausted = all(
                    v.is_exhausted for v in self.transition.children.values()
                )
        return self.is_exhausted


class DataTree:
    """Tracks the tree structure of a collection of ConjectureData
    objects, for use in ConjectureRunner."""

    def __init__(self):
        self.root = TreeNode()

    @property
    def is_exhausted(self):
        """Returns True if every possible node is dead and thus the language
        described must have been fully explored."""
        return self.root.is_exhausted

    def generate_novel_prefix(self, random):
        """Generate a short random string that (after rewriting) is not
        a prefix of any buffer previously added to the tree.

        The resulting prefix is essentially arbitrary - it would be nice
        for it to be uniform at random, but previous attempts to do that
        have proven too expensive.
        """
        assert not self.is_exhausted
        novel_prefix = bytearray()

        def append_int(n_bits, value):
            novel_prefix.extend(int_to_bytes(value, bits_to_bytes(n_bits)))

        current_node = self.root
        while True:
            assert not current_node.is_exhausted
            for i, (n_bits, value) in enumerate(
                zip(current_node.bit_lengths, current_node.values)
            ):
                if i in current_node.forced:
                    append_int(n_bits, value)
                else:
                    while True:
                        k = random.getrandbits(n_bits)
                        if k != value:
                            append_int(n_bits, k)
                            break
                    # We've now found a value that is allowed to
                    # vary, so what follows is not fixed.
                    return bytes(novel_prefix)
            else:
                assert not isinstance(current_node.transition, (Conclusion, Killed))
                if current_node.transition is None:
                    return bytes(novel_prefix)
                branch = current_node.transition
                assert isinstance(branch, Branch)
                n_bits = branch.bit_length

                check_counter = 0
                while True:
                    k = random.getrandbits(n_bits)
                    try:
                        child = branch.children[k]
                    except KeyError:
                        append_int(n_bits, k)
                        return bytes(novel_prefix)
                    if not child.is_exhausted:
                        append_int(n_bits, k)
                        current_node = child
                        break
                    check_counter += 1
                    # We don't expect this assertion to ever fire, but coverage
                    # wants the loop inside to run if you have branch checking
                    # on, hence the pragma.
                    assert (  # pragma: no cover
                        check_counter != 1000
                        or len(branch.children) < (2**n_bits)
                        or any(not v.is_exhausted for v in branch.children.values())
                    )

    def rewrite(self, buffer):
        """Use previously seen ConjectureData objects to return a tuple of
        the rewritten buffer and the status we would get from running that
        buffer with the test function. If the status cannot be predicted
        from the existing values it will be None."""
        buffer = bytes(buffer)

        data = ConjectureData.for_buffer(buffer)
        try:
            self.simulate_test_function(data)
            return (data.buffer, data.status)
        except PreviouslyUnseenBehaviour:
            return (buffer, None)

    def simulate_test_function(self, data):
        """Run a simulated version of the test function recorded by
        this tree. Note that this does not currently call ``stop_example``
        or ``start_example`` as these are not currently recorded in the
        tree. This will likely change in future."""
        node = self.root
        try:
            while True:
                for i, (n_bits, previous) in enumerate(
                    zip(node.bit_lengths, node.values)
                ):
                    v = data.draw_bits(
                        n_bits, forced=node.values[i] if i in node.forced else None
                    )
                    if v != previous:
                        raise PreviouslyUnseenBehaviour()
                if isinstance(node.transition, Conclusion):
                    t = node.transition
                    data.conclude_test(t.status, t.interesting_origin)
                elif node.transition is None:
                    raise PreviouslyUnseenBehaviour()
                elif isinstance(node.transition, Branch):
                    v = data.draw_bits(node.transition.bit_length)
                    try:
                        node = node.transition.children[v]
                    except KeyError as err:
                        raise PreviouslyUnseenBehaviour() from err
                else:
                    assert isinstance(node.transition, Killed)
                    data.observer.kill_branch()
                    node = node.transition.next_node
        except StopTest:
            pass

    def new_observer(self):
        return TreeRecordingObserver(self)


class TreeRecordingObserver(DataObserver):
    def __init__(self, tree):
        self.__current_node = tree.root
        self.__index_in_current_node = 0
        self.__trail = [self.__current_node]
        self.killed = False

    def draw_bits(self, n_bits, forced, value):
        i = self.__index_in_current_node
        self.__index_in_current_node += 1
        node = self.__current_node
        assert len(node.bit_lengths) == len(node.values)
        if i < len(node.bit_lengths):
            if n_bits != node.bit_lengths[i]:
                inconsistent_generation()
            # Note that we don't check whether a previously
            # forced value is now free. That will be caught
            # if we ever split the node there, but otherwise
            # may pass silently. This is acceptable because it
            # means we skip a hash set lookup on every
            # draw and that's a pretty niche failure mode.
            if forced and i not in node.forced:
                inconsistent_generation()
            if value != node.values[i]:
                node.split_at(i)
                assert i == len(node.values)
                new_node = TreeNode()
                branch = node.transition
                branch.children[value] = new_node
                self.__current_node = new_node
                self.__index_in_current_node = 0
        else:
            trans = node.transition
            if trans is None:
                node.bit_lengths.append(n_bits)
                node.values.append(value)
                if forced:
                    node.mark_forced(i)
            elif isinstance(trans, Conclusion):
                assert trans.status != Status.OVERRUN
                # We tried to draw where history says we should have
                # stopped
                inconsistent_generation()
            else:
                assert isinstance(trans, Branch), trans
                if n_bits != trans.bit_length:
                    inconsistent_generation()
                try:
                    self.__current_node = trans.children[value]
                except KeyError:
                    self.__current_node = trans.children.setdefault(value, TreeNode())
                self.__index_in_current_node = 0
        if self.__trail[-1] is not self.__current_node:
            self.__trail.append(self.__current_node)

    def kill_branch(self):
        """Mark this part of the tree as not worth re-exploring."""
        if self.killed:
            return

        self.killed = True

        if self.__index_in_current_node < len(self.__current_node.values) or (
            self.__current_node.transition is not None
            and not isinstance(self.__current_node.transition, Killed)
        ):
            inconsistent_generation()

        if self.__current_node.transition is None:
            self.__current_node.transition = Killed(TreeNode())
            self.__update_exhausted()

        self.__current_node = self.__current_node.transition.next_node
        self.__index_in_current_node = 0
        self.__trail.append(self.__current_node)

    def conclude_test(self, status, interesting_origin):
        """Says that ``status`` occurred at node ``node``. This updates the
        node if necessary and checks for consistency."""
        if status == Status.OVERRUN:
            return
        i = self.__index_in_current_node
        node = self.__current_node

        if i < len(node.values) or isinstance(node.transition, Branch):
            inconsistent_generation()

        new_transition = Conclusion(status, interesting_origin)

        if node.transition is not None and node.transition != new_transition:
            # As an, I'm afraid, horrible bodge, we deliberately ignore flakiness
            # where tests go from interesting to valid, because it's much easier
            # to produce good error messages for these further up the stack.
            if isinstance(node.transition, Conclusion) and (
                node.transition.status != Status.INTERESTING
                or new_transition.status != Status.VALID
            ):
                raise Flaky(
                    f"Inconsistent test results! Test case was {node.transition!r} "
                    f"on first run but {new_transition!r} on second"
                )
        else:
            node.transition = new_transition

        assert node is self.__trail[-1]
        node.check_exhausted()
        assert len(node.values) > 0 or node.check_exhausted()

        if not self.killed:
            self.__update_exhausted()

    def __update_exhausted(self):
        for t in reversed(self.__trail):
            # Any node we've traversed might have now become exhausted.
            # We check from the right. As soon as we hit a node that
            # isn't exhausted, this automatically implies that all of
            # its parents are not exhausted, so we stop.
            if not t.check_exhausted():
                break
