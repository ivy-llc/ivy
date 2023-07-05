import ivy
from ivy.functional.frontends.jax.func_wrapper import to_ivy_arrays_and_back


from functools import partial
import inspect
import itertools
import operator
from typing import Any, Callable, Optional, Sequence, TypeVar

import jax
import weakref
from jax._src import core
from jax._src import linear_util as lu
from jax import config  # type: ignore[no-redef]
from jax._src.core import ConcreteArray, ShapedArray, raise_to_shaped
from jax.tree_util import (
    tree_flatten,
    tree_unflatten,
    treedef_is_leaf,
    tree_map,
    tree_flatten_with_path,
    keystr,
)
from jax._src.api_util import shaped_abstractify
from jax._src.tree_util import equality_errors
from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import source_info_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import xla
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lax import windowed_reductions
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.numpy.ufuncs import logaddexp
from jax._src.traceback_util import api_boundary
from jax._src.util import (
    partition_list,
    safe_map,
    safe_zip,
    split_list,
    unzip2,
    weakref_lru_cache,
)
import numpy as np

from jax._src.lax.control_flow.common import (
    _abstractify,
    _avals_short,
    _check_tree_and_avals,
    _initial_style_jaxpr,
    _make_closed_jaxpr,
    _prune_zeros,
    _typecheck_param,
    allowed_effects,
)


# @to_ivy_arrays_and_back
# def scan(f, init, xs, length=None, reverse=False, unroll=1):
#     if length is None:
#         length = len(xs[0]) if isinstance(xs[0], ivy.Array) else len(xs)

#     if reverse:
#         xs = xs[::-1]

#     ys = []
#     carry = init
#     for x in xs:
#         for _ in range(unroll):
#             carry = f(carry, x)
#             ys.append(carry)

#     return ivy.stack(ys)

_map = safe_map
zip = safe_zip

T = TypeVar("T")
Array = Any
BooleanNumeric = Any  # A bool, or a Boolean array.

### Helper functions


def _promote_weak_typed_inputs(in_vals, in_avals, out_avals):
    """Promote weakly-typed in_vals to be compatible with out_avals.

    Args:
      in_vals : flattened list of input values.
      in_avals : corresponding list of avals.
      out_avals : list of target output avals.
    Returns:
      in_vals_new : flattened list of modified in_vals with no weak types.
      changed : bool; true if in_vals required modification.
    """
    if len(in_vals) != len(in_avals) or len(in_avals) != len(out_avals):
        # Calling function is responsible for catching this.
        return in_vals, False
    weak_mismatches = [
        i
        for i, (a1, a2) in enumerate(zip(in_avals, out_avals))
        if getattr(a1, "weak_type", False) and not core.typematch(a1, a2)
    ]
    if not weak_mismatches:
        return in_vals, False
    for i in weak_mismatches:
        new_dtype = dtypes.result_type(in_vals[i], out_avals[i])
        in_vals[i] = lax.convert_element_type(in_vals[i], new_dtype)
    return in_vals, True


### scan

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")


@api_boundary
def scan(
    f: Callable[[Carry, X], tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
) -> tuple[Carry, Y]:
    """Scan a function over leading array axes while carrying along state.

    The `Haskell-like type signature`_ in brief is

    .. code-block:: haskell

      scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

    where we use [t] here to denote the type t with an additional leading axis.
    That is, if t is an array type then [t] represents the type with an additional
    leading axis, and if t is a pytree (container) type with array leaves then [t]
    represents the type with the same pytree structure and corresponding leaves
    each with an additional leading axis.

    When the type of ``xs`` (denoted `a` above) is an array type or None, and the type
    of ``ys`` (denoted `b` above) is an array type, the semantics of :func:`~scan` are
    given roughly by this Python implementation::

      def scan(f, init, xs, length=None):
        if xs is None:
          xs = [None] * length
        carry = init
        ys = []
        for x in xs:
          carry, y = f(carry, x)
          ys.append(y)
        return carry, np.stack(ys)

    Unlike that Python version, both ``xs`` and ``ys`` may be arbitrary pytree
    values, and so multiple arrays can be scanned over at once and produce multiple
    output arrays. ``None`` is actually a special case of this, as it represents an
    empty pytree.

    Also unlike that Python version, :func:`~scan` is a JAX primitive and is
    lowered to a single WhileOp. That makes it useful for reducing
    compilation times for JIT-compiled functions, since native Python
    loop constructs in an :func:`~jax.jit` function are unrolled, leading to large
    XLA computations.

    Finally, the loop-carried value ``carry`` must hold a fixed shape and dtype
    across all iterations (and not just be consistent up to NumPy rank/shape
    broadcasting and dtype promotion rules, for example). In other words, the type
    ``c`` in the type signature above represents an array with a fixed shape and
    dtype (or a nested tuple/list/dict container data structure with a fixed
    structure and arrays with fixed shape and dtype at the leaves).

    .. note::
      :py:func:`scan` compiles ``f``, so while it can be combined with
      :py:func:`jit`, it's usually unnecessary.

    Args:
      f: a Python function to be scanned of type ``c -> a -> (c, b)``, meaning
        that ``f`` accepts two arguments where the first is a value of the loop
        carry and the second is a slice of ``xs`` along its leading axis, and that
        ``f`` returns a pair where the first element represents a new value for
        the loop carry and the second represents a slice of the output.
      init: an initial loop carry value of type ``c``, which can be a scalar,
        array, or any pytree (nested Python tuple/list/dict) thereof, representing
        the initial loop carry value. This value must have the same structure as
        the first element of the pair returned by ``f``.
      xs: the value of type ``[a]`` over which to scan along the leading axis,
        where ``[a]`` can be an array or any pytree (nested Python
        tuple/list/dict) thereof with consistent leading axis sizes.
      length: optional integer specifying the number of loop iterations, which
        must agree with the sizes of leading axes of the arrays in ``xs`` (but can
        be used to perform scans where no input ``xs`` are needed).
      reverse: optional boolean specifying whether to run the scan iteration
        forward (the default) or in reverse, equivalent to reversing the leading
        axes of the arrays in both ``xs`` and in ``ys``.
      unroll: optional positive int specifying, in the underlying operation of the
        scan primitive, how many scan iterations to unroll within a single
        iteration of a loop.

    Returns:
      A pair of type ``(c, [b])`` where the first element represents the final
      loop carry value and the second element represents the stacked outputs of
      the second output of ``f`` when scanned over the leading axis of the inputs.

    .. _Haskell-like type signature: https://wiki.haskell.org/Type_signature
    """
    if not callable(f):
        raise TypeError("lax.scan: f argument should be a callable.")
    xs_flat, xs_tree = tree_flatten(xs)

    try:
        lengths = [x.shape[0] for x in xs_flat]
    except AttributeError as err:
        msg = "scan got value with no leading axis to scan over: {}."
        raise ValueError(
            msg.format(", ".join(str(x) for x in xs_flat if not hasattr(x, "shape")))
        ) from err

    if length is not None:
        length = int(length)
        if not all(length == l for l in lengths):
            msg = (
                "scan got `length` argument of {} which disagrees with "
                "leading axis sizes {}."
            )
            raise ValueError(msg.format(length, [x.shape[0] for x in xs_flat]))
    else:
        unique_lengths = set(lengths)
        if len(unique_lengths) > 1:
            msg = "scan got values with different leading axis sizes: {}."
            raise ValueError(msg.format(", ".join(str(x.shape[0]) for x in xs_flat)))
        elif len(unique_lengths) == 0:
            msg = "scan got no values to scan over and `length` not provided."
            raise ValueError(msg)
        else:
            (length,) = unique_lengths

    if config.jax_disable_jit:
        if length == 0:
            raise ValueError(
                "zero-length scan is not supported in disable_jit() mode because the output type is unknown."
            )
        carry = init
        ys = []
        maybe_reversed = reversed if reverse else lambda x: x
        for i in maybe_reversed(range(length)):
            xs_slice = [_index_array(i, core.get_aval(x), x) for x in xs_flat]
            carry, y = f(carry, tree_unflatten(xs_tree, xs_slice))
            ys.append(y)
        stack = lambda *ys: jax.numpy.stack(ys)
        stacked_y = tree_map(stack, *maybe_reversed(ys))
        return carry, stacked_y

    xs_avals = [core.raise_to_shaped(core.get_aval(x)) for x in xs_flat]
    x_avals = [core.mapped_aval(length, 0, aval) for aval in xs_avals]

    def _create_jaxpr(init):
        init_flat, init_tree = tree_flatten(init)
        in_flat, in_tree = tree_flatten((init, xs))

        carry_avals = tuple(_map(_abstractify, init_flat))
        jaxpr, consts, out_tree = _initial_style_jaxpr(
            f, in_tree, (*carry_avals, *x_avals), "scan"
        )
        out_tree_children = out_tree.children()
        if len(out_tree_children) != 2:
            msg = "scan body output must be a pair, got {}."
            raise TypeError(msg.format(tree_unflatten(out_tree, jaxpr.out_avals)))
        carry_avals_out = jaxpr.out_avals[: out_tree_children[0].num_leaves]
        return (
            init_flat,
            carry_avals,
            carry_avals_out,
            init_tree,
            in_flat,
            jaxpr,
            consts,
            out_tree,
            out_tree_children,
        )

    # The carry input and output avals must match exactly. However, we want to account for
    # the case when init contains weakly-typed values (e.g. Python scalars), with avals that
    # may not match the output despite being compatible by virtue of their weak type.
    # To do this, we compute the jaxpr in two passes: first with the raw inputs, and if
    # necessary, a second time with modified init values.
    init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)
    new_init_flat, changed = _promote_weak_typed_inputs(
        init_flat, carry_avals, carry_avals_out
    )
    if changed:
        init = tree_unflatten(init_tree, new_init_flat)
        init_flat, carry_avals, carry_avals_out, init_tree, *rest = _create_jaxpr(init)
    in_flat, jaxpr, consts, out_tree, out_tree_children = rest

    _check_scan_carry_type(f, init, out_tree_children[0], carry_avals_out)
    disallowed_effects = allowed_effects.filter_not_in(jaxpr.effects)
    if disallowed_effects:
        raise NotImplementedError(
            f"Effects not supported in `scan`: {disallowed_effects}"
        )

    out = scan_p.bind(
        *consts,
        *in_flat,
        reverse=reverse,
        length=length,
        jaxpr=jaxpr,
        num_consts=len(consts),
        num_carry=len(init_flat),
        linear=(False,) * (len(consts) + len(in_flat)),
        unroll=unroll,
    )
    return tree_unflatten(out_tree, out)


# Now call the scan function with some example arguments to test if it works
print(scan(lambda carry, x: carry + x, 0, [np.ndarray([1, 2, 3]), np.ndarray([4, 5, 6])]))
# print(scan(lambda carry, x: carry + x, 0, [ivy.array([1, 2, 3]), ivy.array([4, 5, 6])]))
