import functools

from ..param import _generate_id
from .. import globals as glob

# -------- #
# Wrapping
# -------- #


def add_incoming_subgraph_fns(graph, fn, arg_param_ids):
    # add information about incoming functions
    if fn.__name__ == "vectorized_fn":  # for handling nested vmaps
        subgraph_dict = graph._tmp_subgraph_id_to_function[-2]
    else:
        subgraph_dict = graph._subgraph_id_to_function
    fns_in = [subgraph_dict[id_] for id_ in arg_param_ids if id_ in subgraph_dict]
    return fns_in


def add_subgraph_fns_to_dict(graph, fn, graph_fn, output_param_ids):
    if fn.__name__ == "vectorized_fn":  # for handling nested vmaps
        graph._subgraph_id_to_function = graph._tmp_subgraph_id_to_function.pop()

    # add this function to the graph for each output pid
    for id_ in output_param_ids:
        if id_ not in graph._subgraph_id_to_function:
            graph._subgraph_id_to_function[id_] = graph_fn


def process_vmap_fn(graph, fn, args, kwargs):
    from tracer.wrapping import _wrap_function_for_op_logging

    new_args = list(args)
    scalar_fn = new_args[0]

    if isinstance(scalar_fn, functools.partial):
        scalar_fn_args = scalar_fn.args
        scalar_fn_kwargs = scalar_fn.keywords
        orig_fn = scalar_fn.func
        scalar_fn = lambda *scalar_fn_args: orig_fn(*scalar_fn_args, **scalar_fn_kwargs)

    # if scalar_fn is itself a jax function, create a new wrapped scalar_function
    if hasattr(scalar_fn, "wrapped_for_tracing"):

        def _wrapped_scalar_fn(*args, **kwargs):
            return scalar_fn(*args, **kwargs)

    else:
        _wrapped_scalar_fn = scalar_fn

    # change the name to track in the logging stack
    setattr(_wrapped_scalar_fn, "__name__", "scalar_fn")
    # _wrapped_scalar_fn.args = scalar_fn_args
    # _wrapped_scalar_fn.kwargs = scalar_fn_kwargs

    graph.vmap_args = new_args[1:]
    graph.vmap_kwargs = kwargs
    graph.vmap_fn = fn
    graph.vmap_node_ids.append(_generate_id())

    # wrap the scalar fn so that it can be traced
    wrapped_scalar_fn = _wrap_function_for_op_logging(
        _wrapped_scalar_fn, graph, to_ivy=graph._to_ivy
    )

    # replace the original scalar fn with the wrapped one
    new_args[0] = wrapped_scalar_fn
    args = tuple(new_args)
    return args, kwargs


def process_scalar_fn(graph, fn, args, output):
    # create the subgraph now that we have returned from the scalar function
    from tracer.graph import Graph

    subgraph_ids = {
        **graph._tmp_subgraph_id_to_function[-1],
        **graph._subgraph_id_to_function,
    }

    subgraph = Graph(fn, *args, to_ivy=graph._to_ivy)
    subgraph._id_to_function[glob.current_trace_mode] = subgraph_ids

    subgraph._register_output(output)
    subgraph.connect()
    # temporarily storing the subgraph. This will later be replaced
    # when building the vmap (i.e. vectorized fn) node
    graph._sub_graphs[graph.vmap_node_ids[-1]] = subgraph

    # reset subgraph objs
    graph._subgraph_id_to_function = {}


def process_vectorized_fn(graph, graph_fn):
    # pop the old subgraph and replace it with a new key to the vmap node
    vmap_id = graph.vmap_node_ids.pop()
    orig_subgraph = graph._sub_graphs.pop(vmap_id)
    orig_subgraph._sub_graphs = dict(graph._sub_graphs)
    graph._sub_graphs[id(graph_fn)] = orig_subgraph

    # add additional attributes to the vmap function
    graph_fn.vmap_args = graph.vmap_args
    graph_fn.vmap_kwargs = graph.vmap_kwargs

    return graph_fn


# -----------------
# Source generation
# -----------------


def get_vmap_args_kwargs_str(vmap_args, vmap_kwargs):
    vmap_args = ["'" + v + "'" if isinstance(v, str) else v for v in vmap_args]
    vmap_args_str = ", ".join(map(str, vmap_args)) if vmap_args else ""
    vmap_kwargs_str = (
        ", ".join(
            f"{k}={v}"
            for k, v in vmap_kwargs.items()
            if k in ("in_axes", "out_axes", "in_dims", "out_dims")
        )
        if vmap_kwargs
        else ""
    )
    return vmap_args_str, vmap_kwargs_str


def generate_vmap_subgraph(
    sg, graph, f, frontend=None, indent=0, nested_fn_body="", fn_header=""
):
    from tracer.source_gen import join_args_n_kwargs

    subgraph = graph._sub_graphs[f.id_]
    sg.count_references(subgraph)
    scalar_fn_name = "scalar_fn_" + str(id(f))
    scalar_fn = sg.generate_source(
        frontend=frontend, graph=subgraph, fn_name=scalar_fn_name, indent=indent + 4
    )

    nested_fn_body += scalar_fn + "\n\n"
    if sg.graph._to_ivy:
        vmap_fn = "ivy.vmap"
    elif sg.graph._transpiling and not sg.graph._to_ivy:
        if indent == 0 and not sg.vmap_imported and sg.graph.backend == "torch":
            fn_header = "import functorch\n" + fn_header
            sg.vmap_imported = True

        if sg.graph.backend == "torch":
            vmap_fn = "functorch.vmap"
        else:
            vmap_fn = "jax.vmap"

    else:
        if sg.graph.backend == "torch":
            vmap_fn = "torch.func.vmap"
        else:
            vmap_fn = "jax.vmap"

    vmap_args_str, vmap_kwargs_str = get_vmap_args_kwargs_str(
        f.vmap_args, f.vmap_kwargs
    )
    args_n_kwargs_str = join_args_n_kwargs(vmap_args_str, vmap_kwargs_str)
    vectorized_fn_str = (
        f"vectorized_fn = {vmap_fn}({scalar_fn_name}, {args_n_kwargs_str})\n"
    )

    return vectorized_fn_str, nested_fn_body, fn_header


# -------------
# Visualization
# -------------


def _handle_vmap_nodes(main_graph, subgraphs, all_functions):
    all_graphs = [main_graph] + subgraphs
    vmap_tuples = []
    all_nodes = []

    def _get_vmap_tuple(subgraph):
        return (list(subgraph.nodes)[1], list(subgraph.nodes)[-1])

    for graph in all_graphs:
        all_nodes += [
            (node, node_attr)
            for (node, node_attr) in graph.nodes(data=True)
            if "input" not in node_attr["label"] and "output" not in node_attr["label"]
        ]

    fn_names = [fn.__name__ for fn in all_functions]

    # Create a dictionary with the name as key and the corresponding index as value
    order_dict = {name: i for i, name in enumerate(fn_names)}

    # Sort the unordered nodes based on the index of the name in the order list, breaking ties with the original position
    ordered_nodes = sorted(
        all_nodes, key=lambda x: (order_dict[x[1]["label"]], all_nodes.index(x))
    )

    # filter for nodes that contain the label "vmap"
    vmap_nodes = [node for node in ordered_nodes if "vmap" in node[1]["label"]]

    for i, node in enumerate(vmap_nodes):
        inp_node, out_node = _get_vmap_tuple(subgraphs[i])
        vmap_tuples.append((inp_node, node[0], out_node))

    return vmap_tuples


def _remove_input_subgraph_nodes(subgraph_nodes):
    outputs = []
    other_nodes = []

    for node, attrs in subgraph_nodes:
        if "input" in attrs["label"]:
            continue
        elif "output" in attrs["label"]:
            outputs.append((node, attrs))
        else:
            other_nodes.append((node, attrs))

    return other_nodes + outputs


def _get_all_graph_functions(graph, all_graph_functions=None):
    if all_graph_functions is None:
        all_graph_functions = []
    for fn in graph._functions:
        if fn.__name__ == "vmap":
            all_graph_functions.append(fn)
            subgraph = graph._sub_graphs[id(fn)]
            _get_all_graph_functions(subgraph, all_graph_functions)
        else:
            all_graph_functions.append(fn)
    return all_graph_functions
