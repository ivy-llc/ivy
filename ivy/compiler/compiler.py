# global
import ivy
import copy

# local
import ivy.compiler.globals as glob
from ivy.compiler.graph import Graph
from ivy.compiler.op_logging import _wrap_methods_for_op_logging, _unwrap_methods_from_op_logging


def _create_graph(fn, *args, stateful=None, num_workers=1, output_connected_only=True, **kwargs):

    # extra stateful instances modified in the graph
    stateful = ivy.default(stateful, [])

    # extract the associated stateful classes
    stateful_classes = [s.__class__ for s in stateful]

    # copy the states for resetting after forward pass and compilation
    state_copies = [copy.deepcopy(s.__dict__) for s in stateful]

    # construct the graph
    graph = Graph(fn, *args, **kwargs, stateful=stateful, num_workers=num_workers)

    # wrap all methods for operation logging
    _wrap_methods_for_op_logging(graph, stateful_classes)

    # forward pass through the graph, logging all operations
    graph.log_all_ops()

    # unwrap all methods, now all operations have been logged
    _unwrap_methods_from_op_logging(stateful_classes)

    # reset the stateful objects to their initial state, prior to compilation
    for s, sc in zip(stateful, state_copies):
        for k in list(s.__dict__.keys()):
            if k not in sc:
                del s.__dict__[k]
                continue
            s.__dict__[k] = sc[k]

    # connect graph
    graph.connect(output_connected_only)

    # reset all global compiler variables, just to be sure
    glob.wrapping_paused = False
    glob.op_logging = False
    glob.wrapped_stack.clear()

    # return graph
    return graph


def compile_graph(fn, *args, stateful=None, num_workers=1, **kwargs):

    # create graph
    graph = _create_graph(fn, *args, stateful=stateful, num_workers=num_workers, **kwargs)

    # compile the graph forward pass into an executable function
    return graph.compiled()


def show_graph(fn, *args, stateful=None, num_workers=1, save_to_disk=False, with_edge_labels=True, with_arg_labels=True,
               output_connected_only=True, **kwargs):

    # create graph
    graph = _create_graph(fn, *args, stateful=stateful, num_workers=num_workers,
                          output_connected_only=output_connected_only, **kwargs)

    # show the compiled graph
    graph.show(save_to_disk, with_edge_labels, with_arg_labels, output_connected_only)
