"""
Logic for applying truncations (replacing many nodes with a single node) to a Graph. Truncations are applied after
the first trace pass of the transpiler, and the replacement node will always be an ivy function - so it can be used
when transpiling to any backend alongside the frontend functions.
"""

import ivy
import networkx as nx
from networkx.algorithms import isomorphism
import tracer.globals as glob
from tracer.graph import Graph
from tracer.wrapping import Node
from typing import (
    Any,
    Callable,
    List,
    Sequence,
    Tuple,
    Union,
)


# Classes #
# ------- #


class ParamLocation:
    """
    Defines the location of a parameter which needs to be passed to the target node as an argument

    Parameters
    ----------
    identifier
        the unique identifier of the truncation node the parameter originates from
    initial_key
        the key (if this is a kwarg) or idx (if it's an arg) of the parameter with the arguments of the source node
    target_key
        the key/idx of the parameter in the target node arguments
    within_subgraph
        the name of the function in the subgraph this parameter originates from, in the case the
        param is within the callback subgraph of a higher order funcion

        otherwise `None`
    sequence_target_key
        if multiple parameters are passed into a single argument as a sequence (list or tuple),
        this specifies the index of this parameter within that sequence

        otherwise `None`
    """

    def __init__(
        self,
        identifier: str,
        initial_key: Union[int, str],
        target_key: Union[int, str],
        within_subgraph: Union[str, None] = None,
        sequence_target_key: Union[int, None] = None,
    ):
        self.identifier = identifier
        self.initial_key = initial_key
        self.target_key = target_key

        # defines the argument to find this param from within the subgraph,
        # if the arg needs to be collected from the subgraph
        self.within_subgraph = within_subgraph

        # the location within a sequence of parameters which are
        # passed to the target as a single argument
        self.sequence_target_key = sequence_target_key


class ParamConstant:
    """
    Holds a constant value along with its location in the arguments of the
    target node (idx/key within the node args/kwargs)
    """

    def __init__(
        self,
        target_key: Union[int, str],
        constant: Any,
    ):
        self.target_key = target_key
        self.constant = constant


class Truncation:
    """
    Defines a single possible truncation - the graph that can be truncated, the target
    node it can be truncated to, the locations of all the required arguments and outputs
    """

    def __init__(
        self,
        truncation_graph: nx.DiGraph,
        target: Callable,
        arg_locations: Sequence[Union[ParamLocation, ParamConstant, Tuple]] = [],
        output_locations: Sequence[ParamLocation] = [],
    ):
        self.truncation_graph = truncation_graph
        self.target = target
        self.arg_locations = arg_locations
        self.output_locations = output_locations


# Helpers #
# ------- #


def _get_arg(graph_node: Node, location: ParamLocation, id_idx: int):
    """
    Collect the arg, arg_param_id and arg_tracked_idx from a specified location
    (arg position) within the graph_node
    """

    is_arg = isinstance(location.initial_key, int)
    args = graph_node.args if is_arg else graph_node.kwargs
    arg_param_ids = graph_node.arg_param_ids if is_arg else graph_node.kwarg_param_ids

    arg = args[location.initial_key]
    if arg is None:
        arg_param_id = arg_param_ids[id_idx]
        arg_tracked_idx = [location.target_key]
        id_idx += 1
    elif isinstance(arg, (list, tuple)):
        # get ids and idxs for all elements of the arg
        arg_param_id = []
        for element in arg:
            if element is None:
                tmp_arg_param_id = arg_param_ids[id_idx]
                arg_param_id.append(tmp_arg_param_id)
                id_idx += 1
        arg_tracked_idx = [
            [location.target_key, i] for i, a in enumerate(arg) if a is None
        ]

    return arg, arg_param_id, arg_tracked_idx, id_idx


def _get_output_ids_recursive(graph_node: Node, output: Union[List, Tuple], i: int):
    """
    Recursively retrieve the param ids and tracked idxs from the graph node,
    so they can be used in the target node
    """

    output_param_ids = []
    output_tracked_idxs = []
    if isinstance(output, (list, tuple)):
        for output_param in output:
            ids, idxs, i = _get_output_ids_recursive(graph_node, output_param, i)
            output_param_ids += ids
            output_tracked_idxs += idxs
    else:
        # TODO: check that this works correctly when there are multiple nodes in the
        # truncation graph which connect to the rest of the graph
        output_param_ids.append(graph_node.output_param_ids[i])
        output_tracked_idxs.append(graph_node.output_tracked_idxs[i])
        i += 1
    return output_param_ids, output_tracked_idxs, i


def _get_subgraph_arg(
    networkx_subgraphs: List[nx.DiGraph],
    truncation_graph: nx.DiGraph,
    location: ParamLocation,
):
    """
    Collect the arg, arg_param_id and arg_tracked_idx from a specified location (node & arg position)
    within the callback subgraph of a higher order node within the graph.
    """

    arg = None
    arg_param_id = None
    arg_tracked_idx = None
    is_arg = isinstance(location.initial_key, int)

    networkx_subgraph = networkx_subgraphs[
        0
    ]  # TODO: generalise for all subgraphs, not just [0]
    truncation_inner_subgraph = truncation_graph.graph["subgraphs"][
        location.identifier
    ][0]

    inner_graph_matcher = isomorphism.DiGraphMatcher(
        networkx_subgraph, truncation_inner_subgraph
    )
    matching_inner_truncation_subgraphs = inner_graph_matcher.isomorphisms_iter()

    networkx_subgraph_nodes = sorted(networkx_subgraph.nodes, key=lambda x: x.__name__)
    networkx_subgraph_ids = [n.id_ for n in networkx_subgraph_nodes]

    for inner_truncation_subgraph in matching_inner_truncation_subgraphs:
        inner_truncation_nodes = sorted(
            inner_truncation_subgraph.keys(), key=lambda x: x.__name__
        )
        inner_truncation_ids = [n.id_ for n in inner_truncation_nodes]

        # only collect the args if the inner truncation subgraph is identically ordered to the
        # networkx subgraph, otherwise args can be placed in the wrong position
        if inner_truncation_ids == networkx_subgraph_ids:
            for (
                subgraph_node,
                matched_subgraph_node,
            ) in inner_truncation_subgraph.items():
                # if the function within the subgraph is the one that contains the required arg
                if location.within_subgraph == matched_subgraph_node:
                    args = subgraph_node.args if is_arg else subgraph_node.kwargs
                    arg_param_ids = (
                        subgraph_node.arg_param_ids
                        if is_arg
                        else subgraph_node.kwarg_param_ids
                    )

                    arg = args[location.initial_key]
                    if arg is None:
                        arg_param_id = arg_param_ids[location.initial_key]
                        arg_tracked_idx = [
                            location.target_key,
                            location.sequence_target_key,
                        ]

    return arg, arg_param_id, arg_tracked_idx


def _is_valid_truncation(target_node: Node, truncation_subgraph: dict, graph: Graph):
    """
    Run through some sanity checks to ensure the truncation is valid before we apply it
    """

    # check that any ids being removed from the graph are not required by any remaining functions
    all_arg_param_ids = []
    for f in graph._functions:
        if f not in truncation_subgraph:
            all_arg_param_ids += f.arg_param_ids
    for graph_node in truncation_subgraph.keys():
        if any(
            [
                out_id in all_arg_param_ids
                and out_id not in target_node.output_param_ids
                for out_id in graph_node.output_param_ids
            ]
        ):
            return False

    # if an output parameter of the whole graph is in a node to be truncated,
    # ensure that the output parameter is also produced by the truncated function
    for output_param_id in graph._output_param_ids[glob.current_trace_mode]:
        for graph_node in truncation_subgraph.keys():
            if output_param_id in graph_node.output_param_ids:
                if output_param_id not in target_node.output_param_ids:
                    return False
    return True


def _node_match(n1, n2):
    """
    Decides whether two nodes match based on whether they share the same name,
    args, and whether any subgraphs they may have are isomorphic
    """

    if "name" in n1 and "name" in n2 and "args" in n1 and "args" in n2:
        # if any kwargs which are required to not be tracked for this
        # truncation to apply are present, the node does not match
        if "non_tracked_kwargs" in n2:
            for key in n2["non_tracked_kwargs"]:
                if [key] in n1["tracked_kwargs"]:
                    return False

        if len(n2["args"]) == 0 and "kwargs" not in n2:
            # if args/kwargs are not specified, only match on name
            return n1["name"] == n2["name"]
        else:
            if len(n1["args"]) != len(n2["args"]):
                # if args are not the same, the nodes do not match
                return False

            if "subgraphs" in n1:
                if "subgraphs" not in n2:
                    return False
                # checks that any callback subgraphs of the node are isomorphic
                for subgraph1, subgraph2 in zip(n1["subgraphs"], n2["subgraphs"]):
                    if not nx.is_isomorphic(subgraph1, subgraph2):
                        return False

            # checks that the node args are from the same function, or are the same
            # style, as those required by the truncation - ie.  ["add", "cached/initial", "any"]
            # means the first arg must come from an `add` function, the second arg must be cached
            # (or not originate from another function), and the third arg can be anything
            args_eq = all(
                [a1 == a2 or a2 == "any" for a1, a2 in zip(n1["args"], n2["args"])]
            )

            # same check for kwargs (although providing kwargs is optional)
            if "kwargs" in n1 and "kwargs" in n2:
                same_keys = True
                same_values = True
                for key1, key2 in zip(n1["kwargs"].keys(), n2["kwargs"].keys()):
                    if key1 != key2:
                        same_keys = False
                for value1, value2 in zip(n1["kwargs"].keys(), n2["kwargs"].keys()):
                    if value1 != value2 and value2 != "any":
                        same_values = False
                kwargs_eq = same_keys and same_values
            else:
                kwargs_eq = True

            return n1["name"] == n2["name"] and args_eq and kwargs_eq
    return False


# Main #
# ---- #


def truncate_graph(graph: Graph):
    """
    Searches the graph to see if there are any possible know truncations present - wherever
    this is the case, the relevant nodes will be truncated into a single ivy function
    """

    truncations = TRUNCATIONS[ivy.current_backend_str()]()
    modes = (
        ["train", "eval"]
        if graph._traced_train_modes == "all"
        else [graph._traced_train_modes]
    )

    for mode in modes:
        # apply truncations to all modes that have been traced
        if mode == "train":
            graph._train()
        else:
            graph._eval()

        for truncation in truncations:

            # convert traced graph to networkx, so we can search for the presence of possible truncations
            networkx_graph = graph._to_networkx(ignore_skippable=True)
            truncation_graph = truncation.truncation_graph
            arg_locations = truncation.arg_locations
            output_locations = truncation.output_locations

            # find any subgraphs of the traced graph which match the specified truncation
            graph_matcher = isomorphism.DiGraphMatcher(
                networkx_graph, truncation_graph, node_match=_node_match
            )
            matching_truncation_subgraphs = graph_matcher.subgraph_isomorphisms_iter()

            for truncation_subgraph in matching_truncation_subgraphs:
                # for each subgraph that matches the truncation, we need to
                # collect the correct attributes for the replacement node
                target_arg_param_ids = []
                target_arg_tracked_idxs = []
                target_args = []
                target_kwarg_param_ids = []
                target_kwarg_tracked_idxs = []
                target_kwargs = {}
                target_output = []
                target_output_param_ids = []
                target_output_tracked_idxs = []
                nodes_to_replace = []

                for graph_node, matched_node in truncation_subgraph.items():
                    nodes_to_replace.append(graph_node)
                    arg_id_idx = 0
                    kwarg_id_idx = 0

                    for arg_location in arg_locations:
                        # each arg_location specifies the function and position within the arguments
                        # of that function of an argument which is required by the target node

                        if isinstance(arg_location, ParamLocation):
                            arg = None
                            arg_param_id = None
                            arg_tracked_idx = None
                            if arg_location.identifier == matched_node:
                                if arg_location.within_subgraph is not None:
                                    # locate the arg, arg_param_id and arg_tracked_idx within the callback subgraph
                                    networkx_subgraphs = networkx_graph.graph[
                                        "subgraphs"
                                    ][graph_node.id_]
                                    (
                                        arg,
                                        arg_param_id,
                                        arg_tracked_idx,
                                    ) = _get_subgraph_arg(
                                        networkx_subgraphs,
                                        truncation_graph,
                                        arg_location,
                                    )
                                else:
                                    # locate the arg, arg_param_id and arg_tracked_idx whether they are in
                                    # the arg or kwargs of the initial node
                                    if isinstance(arg_location.initial_key, int):
                                        (
                                            arg,
                                            arg_param_id,
                                            arg_tracked_idx,
                                            arg_id_idx,
                                        ) = _get_arg(
                                            graph_node, arg_location, arg_id_idx
                                        )
                                    else:
                                        (
                                            arg,
                                            arg_param_id,
                                            arg_tracked_idx,
                                            kwarg_id_idx,
                                        ) = _get_arg(
                                            graph_node, arg_location, kwarg_id_idx
                                        )

                                    # add the arg, arg_param_id and arg_tracked_idx to the arg or kwargs of the target node
                                    if isinstance(arg_location.target_key, int):
                                        target_args.append(arg)
                                        if (
                                            arg_param_id is not None
                                            and arg_tracked_idx is not None
                                        ):
                                            if isinstance(arg_param_id, (list, tuple)):
                                                for param_id in arg_param_id:
                                                    target_arg_param_ids.append(
                                                        param_id
                                                    )
                                                for tracked_idx in arg_tracked_idx:
                                                    target_arg_tracked_idxs.append(
                                                        tracked_idx
                                                    )
                                            else:
                                                target_arg_param_ids.append(
                                                    arg_param_id
                                                )
                                                target_arg_tracked_idxs.append(
                                                    arg_tracked_idx
                                                )
                                    else:
                                        target_kwargs[arg_location.target_key] = arg
                                        if (
                                            arg_param_id is not None
                                            and arg_tracked_idx is not None
                                        ):
                                            if isinstance(arg_param_id, (list, tuple)):
                                                for param_id in arg_param_id:
                                                    target_kwarg_param_ids.append(
                                                        param_id
                                                    )
                                                for tracked_idx in arg_tracked_idx:
                                                    target_kwarg_tracked_idxs.append(
                                                        tracked_idx
                                                    )
                                            else:
                                                target_kwarg_param_ids.append(
                                                    arg_param_id
                                                )
                                                target_kwarg_tracked_idxs.append(
                                                    arg_tracked_idx
                                                )

                        elif isinstance(arg_location, ParamConstant):
                            # if this argument of the target node should always be the same
                            if isinstance(arg_location.target_key, int):
                                target_args.append(arg_location.constant)
                            else:
                                target_kwargs[arg_location.target_key] = (
                                    arg_location.constant
                                )

                        elif isinstance(arg_location, (list, tuple)):
                            # the required target node argument is a sequence of tracked/cached parameters from different fns/locations

                            # initially fill the lists with empty positions - this allows us to ensure the elements will be ordered correctly
                            seq_arg = ["empty_position" for _ in arg_location]
                            seq_arg_param_ids = ["empty_position" for _ in arg_location]
                            seq_arg_tracked_idxs = [
                                "empty_position" for _ in arg_location
                            ]
                            target_key = arg_location[0].target_key

                            for sub_location in arg_location:
                                # if the required argument is contained within a callback subgraph
                                if sub_location.within_subgraph is not None:
                                    # collect the arg, arg_param_id, arg_tracked_idx from within the subgraph
                                    networkx_subgraphs = networkx_graph.graph[
                                        "subgraphs"
                                    ][graph_node.id_]
                                    arg, arg_param_id, arg_tracked_idx = (
                                        _get_subgraph_arg(
                                            networkx_subgraphs,
                                            truncation_graph,
                                            sub_location,
                                        )
                                    )

                                    # add the collected arg to `seq_arg` - the sequence of arguments to be given to the target node
                                    seq_arg[sub_location.sequence_target_key] = arg
                                    if arg is None:
                                        seq_arg_param_ids[
                                            sub_location.sequence_target_key
                                        ] = arg_param_id
                                        seq_arg_tracked_idxs[
                                            sub_location.sequence_target_key
                                        ] = arg_tracked_idx
                                else:
                                    args = (
                                        graph_node.args
                                        if isinstance(sub_location.initial_key, int)
                                        else graph_node.kwargs
                                    )
                                    seq_arg[sub_location.sequence_target_key] = args[
                                        sub_location.initial_key
                                    ]

                            def _position_used(x):
                                return (not isinstance(x, str)) or x != "empty_position"

                            # remove any unused positions from the arg attributes
                            seq_arg = [x for x in seq_arg if _position_used(x)]
                            seq_arg_param_ids = [
                                x for x in seq_arg_param_ids if _position_used(x)
                            ]
                            seq_arg_tracked_idxs = [
                                x for x in seq_arg_tracked_idxs if _position_used(x)
                            ]

                            # add the collected arg, ids, etc to the target lists, which will ultimately be added to the target node
                            if isinstance(target_key, int):
                                target_args.append(seq_arg)
                                for p_id, p_idx in zip(
                                    seq_arg_param_ids, seq_arg_tracked_idxs
                                ):
                                    target_arg_param_ids.append(p_id)
                                    target_arg_tracked_idxs.append(p_idx)
                            else:
                                target_kwargs[target_key] = seq_arg
                                for p_id, p_idx in zip(
                                    seq_arg_param_ids, seq_arg_tracked_idxs
                                ):
                                    target_kwarg_param_ids.append(p_id)
                                    target_kwarg_tracked_idxs.append(p_idx)

                    # get the target node output, output_param_ids and output_tracked_idxs
                    for output_location in output_locations:
                        if output_location.identifier == matched_node:
                            output = graph_node.output[output_location.initial_key]
                            target_output.append(output)

                    t_output_param_ids, t_output_tracked_idxs, _ = (
                        _get_output_ids_recursive(graph_node, output, 0)
                    )
                    target_output_param_ids += t_output_param_ids
                    target_output_tracked_idxs += t_output_tracked_idxs

                # create the replacement node
                target_node = Node()
                target_node.backend_fn = truncation.target
                target_node.__name__ = truncation.target.__name__
                target_node.id_ = id(target_node)
                target_node.args = target_args
                target_node.arg_param_ids = target_arg_param_ids
                target_node.arg_tracked_idxs = target_arg_tracked_idxs
                target_node.kwargs = target_kwargs
                target_node.kwarg_param_ids = target_kwarg_param_ids
                target_node.kwarg_tracked_idxs = target_kwarg_tracked_idxs
                target_node.output = target_output
                target_node.output_param_ids = target_output_param_ids
                target_node.output_tracked_idxs = target_output_tracked_idxs
                target_node.inplace_fn = False
                target_node.from_tracked_var = False
                target_node.with_tracked_slices = []
                target_node.is_ivy = True
                target_node.fns_in = [
                    graph._id_to_function[glob.current_trace_mode][param_id]
                    for param_id in target_arg_param_ids + target_kwarg_param_ids
                    if param_id in graph._id_to_function[glob.current_trace_mode]
                ]
                target_node.fns_out = [
                    graph._id_to_function[glob.current_trace_mode][out_id]
                    for out_id in target_output_param_ids
                    if out_id in graph._id_to_function[glob.current_trace_mode]
                ]

                # apply the truncation to the graph if it passes sanity checks to ensure its validity
                if _is_valid_truncation(target_node, truncation_subgraph, graph):
                    graph.replace_nodes(nodes_to_replace, target_node)
                    graph.contains_truncations = True


# Truncations #
# ----------- #


def get_jax_truncations():
    return []


def get_numpy_truncations():
    return []


def get_paddle_truncations():
    return []


def get_tensorflow_truncations():

    # LSTM WITHOUT BIAS
    tf_lstm_no_bias_step_subgraph = nx.DiGraph()
    # NOTE: can also include `kwargs`, but is not necessary
    tf_lstm_no_bias_step_subgraph.add_node(
        "dot_1", name="dot", args=["any", "any"], kwargs={}, subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "dot_2", name="dot", args=["any", "any"], subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "rep_1", name="rep_method", args=["dot", "dot"], subgraphs=[]
    )  # `add`
    tf_lstm_no_bias_step_subgraph.add_node(
        "split_1", name="split", args=["add", "cached/initial"], subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "sigmoid_1", name="sigmoid", args=["split"], subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "sigmoid_2", name="sigmoid", args=["split"], subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "rep_2", name="rep_method", args=["sigmoid", "cached/initial"], subgraphs=[]
    )  # `mul`
    tf_lstm_no_bias_step_subgraph.add_node(
        "tanh_1", name="tanh", args=["split"], subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "rep_3", name="rep_method", args=["sigmoid", "tanh"], subgraphs=[]
    )  # `mul`
    tf_lstm_no_bias_step_subgraph.add_node(
        "rep_4", name="rep_method", args=["rep_method", "rep_method"], subgraphs=[]
    )  # `add`
    tf_lstm_no_bias_step_subgraph.add_node(
        "sigmoid_3", name="sigmoid", args=["split"], subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "tanh_2", name="tanh", args=["rep_method"], subgraphs=[]
    )
    tf_lstm_no_bias_step_subgraph.add_node(
        "rep_5", name="rep_method", args=["sigmoid", "tanh"], subgraphs=[]
    )  # `mul`

    tf_lstm_no_bias_step_subgraph.add_edge(
        "dot_1", "rep_1", origin="dot", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "dot_2", "rep_1", origin="dot", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "rep_1", "split_1", origin="rep_method", target="split"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "split_1", "sigmoid_1", origin="split", target="sigmoid"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "split_1", "sigmoid_2", origin="split", target="sigmoid"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "split_1", "tanh_1", origin="split", target="tanh"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "split_1", "sigmoid_3", origin="split", target="sigmoid"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "sigmoid_2", "rep_2", origin="sigmoid", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "sigmoid_1", "rep_3", origin="sigmoid", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "sigmoid_3", "rep_5", origin="sigmoid", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "rep_2", "rep_4", origin="rep_method", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "rep_3", "rep_4", origin="rep_method", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "rep_4", "tanh_2", origin="rep_method", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "tanh_1", "rep_3", origin="tanh", target="rep_method"
    )
    tf_lstm_no_bias_step_subgraph.add_edge(
        "tanh_2", "rep_5", origin="tanh", target="rep_method"
    )

    tf_lstm_no_bias_graph = nx.DiGraph()
    tf_lstm_no_bias_graph.add_node(
        "rnn_1",
        name="rnn",
        args=["subgraph", "any", "list"],
        subgraphs=[tf_lstm_no_bias_step_subgraph],
        non_tracked_kwargs=[
            "mask",
            "unroll",
        ],  # defines any kwargs which must not be tracked for the truncation to be applicable
    )
    tf_lstm_no_bias_graph.graph["subgraphs"] = {
        "rnn_1": [tf_lstm_no_bias_step_subgraph],
    }

    tf_lstm_no_bias = Truncation(
        tf_lstm_no_bias_graph,
        ivy.lstm,
        arg_locations=[
            # keep these ordered
            ParamLocation(
                "rnn_1", 1, 0
            ),  # this is saying tensorflow.keras.backend.rnn argument[1] needs to become argument[0] of ivy.lstm
            ParamLocation("rnn_1", 2, 1),
            [
                ParamLocation(
                    "rnn_1", 1, 2, within_subgraph="dot_1", sequence_target_key=0
                ),
                ParamLocation(
                    "rnn_1", 1, 2, within_subgraph="dot_2", sequence_target_key=1
                ),
            ],  # weights
            ParamConstant(
                3, 1
            ),  # num layers will always be one, because tf.keras.layers.LSTM represents a single lstm layer
            ParamConstant(4, 0.0),  # dropout
            ParamConstant(5, False),  # training
            ParamConstant(6, False),  # bidirectional
            ParamConstant(7, True),  # batch_first
            ParamConstant("has_ih_bias", False),
            ParamConstant("has_hh_bias", False),
            ParamConstant("weights_transposed", True),
        ],
        output_locations=[
            ParamLocation("rnn_1", 0, 0),
        ],
    )

    # LSTM WITH BIAS
    tf_lstm_w_bias_step_subgraph = nx.DiGraph()
    tf_lstm_w_bias_step_subgraph.add_node(
        "dot_1", name="dot", args=["any", "any"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "dot_2", name="dot", args=["any", "any"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "rep_1", name="rep_method", args=["dot", "dot"], subgraphs=[]
    )  # `add`
    tf_lstm_w_bias_step_subgraph.add_node(
        "bias_add_1",
        name="bias_add",
        args=["rep_method", "cached/initial"],
        subgraphs=[],
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "split_1", name="split", args=["bias_add", "cached/initial"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "sigmoid_1", name="sigmoid", args=["split"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "sigmoid_2", name="sigmoid", args=["split"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "rep_2", name="rep_method", args=["sigmoid", "cached/initial"], subgraphs=[]
    )  # `mul`
    tf_lstm_w_bias_step_subgraph.add_node(
        "tanh_1", name="tanh", args=["split"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "rep_3", name="rep_method", args=["sigmoid", "tanh"], subgraphs=[]
    )  # `mul`
    tf_lstm_w_bias_step_subgraph.add_node(
        "rep_4", name="rep_method", args=["rep_method", "rep_method"], subgraphs=[]
    )  # `add`
    tf_lstm_w_bias_step_subgraph.add_node(
        "sigmoid_3", name="sigmoid", args=["split"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "tanh_2", name="tanh", args=["rep_method"], subgraphs=[]
    )
    tf_lstm_w_bias_step_subgraph.add_node(
        "rep_5", name="rep_method", args=["sigmoid", "tanh"], subgraphs=[]
    )  # `mul`

    tf_lstm_w_bias_step_subgraph.add_edge(
        "dot_1", "rep_1", origin="dot", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "dot_2", "rep_1", origin="dot", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "rep_1", "bias_add_1", origin="rep_method", target="bias_add"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "bias_add_1", "split_1", origin="bias_add", target="split"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "split_1", "sigmoid_1", origin="split", target="sigmoid"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "split_1", "sigmoid_2", origin="split", target="sigmoid"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "split_1", "tanh_1", origin="split", target="tanh"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "split_1", "sigmoid_3", origin="split", target="sigmoid"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "sigmoid_2", "rep_2", origin="sigmoid", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "sigmoid_1", "rep_3", origin="sigmoid", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "sigmoid_3", "rep_5", origin="sigmoid", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "rep_2", "rep_4", origin="rep_method", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "rep_3", "rep_4", origin="rep_method", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "rep_4", "tanh_2", origin="rep_method", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "tanh_1", "rep_3", origin="tanh", target="rep_method"
    )
    tf_lstm_w_bias_step_subgraph.add_edge(
        "tanh_2", "rep_5", origin="tanh", target="rep_method"
    )

    tf_lstm_w_bias_graph = nx.DiGraph()
    tf_lstm_w_bias_graph.add_node(
        "rnn_1",
        name="rnn",
        args=["subgraph", "any", "list"],
        subgraphs=[tf_lstm_w_bias_step_subgraph],
        non_tracked_kwargs=["mask", "unroll"],
    )
    tf_lstm_w_bias_graph.graph["subgraphs"] = {
        "rnn_1": [tf_lstm_w_bias_step_subgraph],
    }

    tf_lstm_w_bias = Truncation(
        tf_lstm_w_bias_graph,
        ivy.lstm,
        arg_locations=[
            ParamLocation("rnn_1", 1, 0),
            ParamLocation("rnn_1", 2, 1),
            [
                ParamLocation(
                    "rnn_1", 1, 2, within_subgraph="dot_1", sequence_target_key=0
                ),
                ParamLocation(
                    "rnn_1", 1, 2, within_subgraph="dot_2", sequence_target_key=1
                ),
                ParamLocation(
                    "rnn_1", 1, 2, within_subgraph="bias_add_1", sequence_target_key=2
                ),
            ],  # weights
            ParamConstant(3, 1),  # num layers
            ParamConstant(4, 0.0),  # dropout
            ParamConstant(5, False),  # training
            ParamConstant(6, False),  # bidirectional
            ParamConstant(7, True),  # batch_first
            ParamConstant("has_ih_bias", True),
            ParamConstant("has_hh_bias", False),
            ParamConstant("weights_transposed", True),
        ],
        output_locations=[
            ParamLocation("rnn_1", 0, 0),
        ],
    )

    # truncations are in order largest -> smallest
    return [
        tf_lstm_w_bias,
        tf_lstm_no_bias,
    ]


def get_torch_truncations():
    return []


# all possible truncations for each framework
TRUNCATIONS = {
    "jax": get_jax_truncations,
    "numpy": get_numpy_truncations,
    "paddle": get_paddle_truncations,
    "tensorflow": get_tensorflow_truncations,
    "torch": get_torch_truncations,
}
