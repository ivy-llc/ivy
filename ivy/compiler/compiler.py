# global
import ivy
import copy

# local
from ivy.compiler.graph import Graph
from ivy.compiler import globals as glob
# noinspection PyProtectedMember
from ivy.compiler.op_logging import _wrap_methods_for_op_logging, _unwrap_methods_from_op_logging


def _create_graph(fn, *args, stateful=None, arg_stateful_idxs=None, kwarg_stateful_idxs=None,
                  output_connected_only=True, include_generators=True, with_array_caching=True, name='graph', **kwargs):

    # extra stateful instances modified in the graph
    stateful = ivy.default(stateful, [])
    arg_stateful_idxs = ivy.default(arg_stateful_idxs, [])
    stateful_args = ivy.multi_index_nest(args, arg_stateful_idxs)
    kwarg_stateful_idxs = ivy.default(kwarg_stateful_idxs, [])
    stateful_kwargs = ivy.multi_index_nest(kwargs, kwarg_stateful_idxs)
    all_stateful = stateful + stateful_args + stateful_kwargs

    # extract the associated stateful classes
    all_stateful_classes = [s.__class__ for s in all_stateful]

    # copy the states for resetting after forward pass and compilation
    all_state_copies = list()
    for s in all_stateful:
        state_copy = copy.deepcopy(s.__dict__)
        if isinstance(s, dict):
            state_copy = {**state_copy, **s}
        all_state_copies.append(state_copy)

    # construct the graph
    graph = Graph(name, fn, *args, **kwargs, stateful=stateful, arg_stateful_idxs=arg_stateful_idxs,
                  kwarg_stateful_idxs=kwarg_stateful_idxs, include_generators=include_generators,
                  with_array_caching=with_array_caching)

    # wrap all methods for operation logging
    _wrap_methods_for_op_logging(graph, all_stateful_classes)

    # forward pass through the graph, logging all operations
    try:
        graph.log_all_ops()
    except Exception as e:
        _unwrap_methods_from_op_logging(all_stateful_classes)
        # noinspection PyBroadException
        try:
            graph.show(save_to_disk=True, output_connected_only=False, fname='graph_at_point_of_failure.png')
        except Exception:
            pass
        raise e

    # unwrap all methods, now all operations have been logged
    _unwrap_methods_from_op_logging(all_stateful_classes)

    # reset the stateful objects to their initial state, prior to compilation
    for s, sc in zip(all_stateful, all_state_copies):
        for k in list(s.__dict__.keys()):
            if k not in sc:
                del s.__dict__[k]
                continue
            s.__dict__[k] = sc[k]
        if isinstance(s, dict):
            for k in list(s.keys()):
                if k not in sc:
                    del s[k]
                    continue
                s[k] = sc[k]

    # connect graph
    graph.connect(output_connected_only)

    # reset all global compiler variables, just to be sure
    glob.wrapping_paused = False
    glob.op_logging = False
    glob.wrapped_stack.clear()
    glob.raw_pids_to_weakrefs = dict()
    glob.raw_pids_to_unique_pids = dict()
    glob.dependent_pids = set()

    # return graph
    return graph


def compile_graph(fn, *args, stateful=None, arg_stateful_idxs=None, kwarg_stateful_idxs=None, include_generators=True,
                  with_array_caching=True, return_graph=False, time_chronological=True, time_inference=False,
                  timing_fname=None, name='graph', **kwargs):

    # set time inference flag
    glob.time_inference = time_inference
    glob.timing_fname = timing_fname

    # create graph
    graph = _create_graph(
        fn, *args, stateful=stateful, arg_stateful_idxs=arg_stateful_idxs, kwarg_stateful_idxs=kwarg_stateful_idxs,
        include_generators=include_generators, with_array_caching=with_array_caching, name=name, **kwargs)

    # compile the graph forward pass into an executable function
    comp_fn = graph.compiled(time_chronological)

    # return
    if return_graph:
        return comp_fn, graph
    return comp_fn


def show_graph(fn, *args, stateful=None, arg_stateful_idxs=None, kwarg_stateful_idxs=None, randomness_factor=0.1,
               save_to_disk=False, with_edge_labels=True, with_arg_labels=True, with_output_labels=True,
               output_connected_only=True, include_generators=True, with_array_caching=True, highlight_subgraph=None,
               fname=None, return_graph=False, name='graph', **kwargs):

    # create graph
    graph = _create_graph(
        fn, *args, stateful=stateful, arg_stateful_idxs=arg_stateful_idxs, kwarg_stateful_idxs=kwarg_stateful_idxs,
        output_connected_only=output_connected_only, include_generators=include_generators,
        with_array_caching=with_array_caching, name=name, **kwargs)

    # show the compiled graph
    graph.show(save_to_disk, with_edge_labels, with_arg_labels, with_output_labels, output_connected_only,
               randomness_factor, highlight_subgraph, fname=fname)

    # return
    if return_graph:
        return graph
