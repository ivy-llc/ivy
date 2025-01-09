# global
import sys
import weakref
from typing import Optional, Tuple, Type
from random import randint

# local
from .conversion import is_array
from .helpers import _is_tracked_np_proxy, _is_tracked_variable
from . import tracked_var_proxy as tvp
from . import globals as glob
import ivy


class Param:
    """A class for parameters in the graph. Parameters can be native arrays
    such as torch tensors or something else tracked in the graph such as
    native shapes, ints etc.

    Attributes
    ----------
    ptype
        the type of the parameter e.g. <class 'torch.Tensor'>.
    is_var
        whether the parameter is a variable.
    shape
        the shape of the parameter, if one exists.
    """

    def __init__(
        self,
        ptype: Type[ivy.NativeArray],
        is_var: bool = False,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        self.ptype = ptype
        self.is_var = is_var
        self.shape = tuple(shape) if ivy.exists(shape) else None

    def __repr__(self):
        return "<Param, type={}>".format(self.ptype)


# Tracking Parameters #
# ------------------- #


def is_parameter(x, with_numpy=False):
    return (
        is_array(x, with_numpy=with_numpy) or
        _is_tracked_variable(x) or
        _is_tracked_np_proxy(x)
    )


def get_types(parameters):
    """
    Get types of the parameters, for labelling the
    edges of the visualised graph.
    """
    return [p.__class__ for p in parameters]


def get_shapes(parameters):
    """
    Get parameter shapes, so they can
    be displayed in the visualised graph.
    """
    return [tuple(p.shape) if hasattr(p, "shape") else None for p in parameters]


def get_var_flags(parameters):
    """
    Returns whether each parameter is a variable, as variables
    will be coloured differently in the visualised graph.
    """
    return [ivy.current_backend().is_variable(p, exclusive=True) for p in parameters]


def get_ids(parameters, to_ivy):
    parameters = [ivy.to_native(p) if to_ivy else p for p in parameters]
    return [_get_unique_id(p) for p in parameters]


def _find_parameter_indexes(nest, with_numpy, stateful_classes):
    """Find the indexes of the parameters in the args and kwargs."""
    return ivy.nested_argwhere(
        nest,
        lambda x: is_parameter(x, with_numpy=with_numpy)
        or isinstance(x, stateful_classes),
        check_nests=True,
        to_ignore=tvp.get_types_to_ignore(),
    )


def record_parameters_info(
    args, to_ivy, with_numpy, stateful_classes=(), stateful_idxs=[]
):
    indexes = (
        _find_parameter_indexes(args, with_numpy, stateful_classes) + stateful_idxs
    )
    parameters = ivy.multi_index_nest(args, indexes)
    ids = get_ids(parameters, to_ivy)
    types = get_types(parameters)
    var_flags = get_var_flags(parameters)
    shapes = get_shapes(parameters)
    return indexes, parameters, ids, types, var_flags, shapes


# Unique ID Handling #
# ------------------ #


def _generate_id() -> int:
    """
    Generates a parameter id which will be a positive integer less than sys.maxsize
    which is 2^31-1 or 2^63-1, for 32-bit and 64-bit systems, respectively.
    """
    return randint(0, sys.maxsize)


def store_unique_id(x: ivy.NativeArray, graph):
    """Ensure each parameter in the graph has a unique id. For example
    an inplace function will have the same return id as one of the inputs,
    but we require them to be unique, so we generate a new unique id for
    the output, storing it in `raw_id_to_unique_id`, so future `_get_unique_id`
    calls can find the new id.
    """
    orig_id = id(x)
    new_id = _generate_id()
    glob.raw_id_to_unique_id[glob.current_trace_mode][orig_id] = new_id
    if orig_id in graph._stateful_clone_id_dict:
        graph._stateful_clone_id_dict[new_id] = graph._stateful_clone_id_dict[orig_id]


def _get_unique_id(x: ivy.NativeArray) -> int:
    """Returns the unique id for the parameter x, which is just id(x)
    unless x has appeared as multiple edges in the graph, in which case
    it will be some new id generated in `store_unique_id`.
    """
    id_ = id(x)
    if id_ in glob.raw_id_to_weakref and not ivy.exists(glob.raw_id_to_weakref[id_]()):
        # cpython sometimes reusues ids for objects with non-overlapping lifetimes
        # this has occurred if we arrive here, so we need to generate a unqiue id for x
        glob.raw_id_to_unique_id[glob.current_trace_mode][id_] = _generate_id()

    unique_id = (
        glob.raw_id_to_unique_id[glob.current_trace_mode][id_]
        if id_ in glob.raw_id_to_unique_id[glob.current_trace_mode]
        else id_
    )
    try:
        glob.raw_id_to_weakref[id(x)] = weakref.ref(x)
    except TypeError:
        glob.raw_id_to_weakref[id(x)] = lambda: x
    return unique_id


# Deleting Parameters #
# ------------------- #


def delete_parameter(x: ivy.NativeArray, graph) -> Optional[ivy.NativeArray]:
    """Returns the input x if it isn't a parameter and caching is enabled, since
    then the arg needs to be retained and cached in the graph. Otherwise returns
    None, having the effect of deleting the parameter.
    """
    x = ivy.to_native(x) if graph._to_ivy else x
    id_ = _get_unique_id(x)

    # TODO: this is a temporary fix for a problem with tracing both train and eval branches
    # of a keras model containing lstm. For some reason the second branch to be traced doesn't
    # connect part of the graph properly, and a 0-d tensor parameter has already been deleted
    # so it can't be cached. This change allows it to be cached.
    if hasattr(x, "shape") and len(x.shape) == 0:
        return x

    if id_ not in glob.dependent_ids[glob.current_trace_mode] and graph._array_caching:
        return x
    else:
        return None
