# global
import sys
import importlib
import importlib.util
from typing import Callable, List, Tuple
from types import EllipsisType
import textwrap
import re
import os
import builtins
import keyword
import math
import linecache

# local
import ivy
from . import helpers
from .param import _get_unique_id
from . import globals as glob
from .special_ops.vmap_helpers import generate_vmap_subgraph

BINARY_OPERATORS = {
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__matmul__": "@",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__mod__": "%",
    "__pow__": "**",
    "__lshift__": "<<",
    "__rshift__": ">>",
    "__and__": "&",
    "__xor__": "^",
    "__or__": "|",
    "__lt__": "<",
    "__le__": "<=",
    "__eq__": "==",
    "__ne__": "!=",
    "__gt__": ">",
    "__ge__": ">=",
}


class EvalCacheLoader(object):
    def __init__(self):
        self.eval_cache = {}
        self.next_id = 0

    def cache(self, src: str, globals):
        """Store the source for the scripted call in a private cache, and add
        a lazy entry in linecache that allows the source to be retrieved by 'filename'.
        """

        key = self._get_key()
        self.eval_cache[key] = src

        # Don't mutate globals so that this loader is only used
        # to populate linecache, and doesn't interact with other modules
        # that might check `__loader__`
        globals_copy = globals.copy()
        globals_copy["__file__"] = key
        globals_copy["__name__"] = key
        globals_copy["__loader__"] = self
        linecache.lazycache(key, globals_copy)

        return key

    def get_source(self, module_name):
        if module_name in self.eval_cache:
            return self.eval_cache[module_name]
        return None

    def _get_key(self):
        key = f"scripted_call.{self.next_id}"
        self.next_id += 1
        return key


_loader = EvalCacheLoader()


class SourceGenerator:
    def __init__(self, graph):
        self.graph = graph
        # Constants cached during graph creation, these are passed to the traced call.
        self.constants = {}
        # Count of references to intermediate variables
        self.reference_count = {}
        self.count_references(graph)
        # Count of fns
        self.fn_counter = {}
        # Count of memory id and their styled var names
        self.id_name_db = {}
        # store information from the generated source code of subgraphs to avoid possible duplication
        for subgraph_list in graph._id_to_subgraphs[glob.current_trace_mode].values():
            for _, subgraph in subgraph_list:
                self.id_name_db.update(subgraph._eager_graph._id_name_db)
                for name in subgraph._eager_graph._id_name_db.values():
                    remove_idx_pattern = re.compile(r"\[.*\]$")
                    name_no_idx = re.sub(remove_idx_pattern, "", name)
                    remove_num_pattern = re.compile(r"_[0-9]+$")
                    standardized_name = re.sub(remove_num_pattern, "", name_no_idx)
                    if standardized_name in self.fn_counter:
                        self.fn_counter[standardized_name] += 1
                    else:
                        self.fn_counter[standardized_name] = 1
        self.vmap_imported = False
        # maps eg `[unstack[0], unstack[1]]` to `unstack`
        self.unpacked_to_original = {}
        # imports to add for special cases listed in `glob.FN_PATH_TO_IMPORT_PATH`
        self.special_imports = set()

    def count_references(self, graph=None):
        for f in graph._functions:
            for a in f.arg_param_ids:
                self.reference_count[a] = self.reference_count.get(a, 0) + 1
            for kw in f.kwarg_param_ids:
                self.reference_count[kw] = self.reference_count.get(kw, 0) + 1
        for o in graph._output_param_ids[glob.current_trace_mode]:
            self.reference_count[o] = self.reference_count.get(o, 0) + 1

    def get_constants(self) -> dict:
        return self.constants

    def generate_vars_from(self, from_: str, graph=None) -> List[str]:
        """Creates references to the tracked params to extract their values from the args and kwargs
        of the original function. A list of lines will be returned, where each line will have the format:

        "p{tracked_param_id} = {args|kwargs}[i1][i2]["keyword"][...]"
        """
        if from_ not in ["args", "kwargs"]:
            raise ValueError("from_ must be one of 'args' or 'kwargs'.")
        extracted_params = []
        param_ids = graph._arg_param_ids if from_ == "args" else graph._kwarg_param_ids
        tracked_idxs = (
            graph._arg_tracked_idxs if from_ == "args" else graph._kwarg_tracked_idxs
        )
        key_from = lambda x: "'" + x + "'"
        # handles edge case where graph._fn_signature.keys() is empty
        fn_signs = list(graph._fn_signature.keys())
        if (
            not fn_signs
            or set(fn_signs) <= set(["args", "kwargs"])
            or len(param_ids) > len(fn_signs)
        ):  # handles ivy.container as input e.g. test_cont_with_NDArray
            fn_signs = [f"input_{i}" for i in range(len(param_ids))]

        if glob.tracing_subgraph:
            # include a prefix to the variables to distinguish from main graph vars in the same scope
            sign_prefix = (
                "lamdba" if graph._fn.__name__ == "<lambda>" else graph._fn.__name__
            )
            fn_signs = [sign_prefix + "_" + f_s for f_s in fn_signs]

        # using zip controls args and kwargs according to param_ids pulled by from_ value
        for i, (mem_id, idx) in enumerate(zip(param_ids, tracked_idxs)):
            indices = ']['.join(key_from(i) if isinstance(i, str) else str(i) for i in idx)
            indices_str = f"[{indices}]"
            if from_ == "args":
                tracked_name = fn_signs[i]
                if tracked_name == "args":
                    tracked_name = "input_arg"
                # avoid name collision with **kwargs
                if tracked_name == "kwargs":
                    tracked_name = "input_kwargs"
            else:
                tracked_name = "k_" + re.sub(r"[^a-zA-Z0-9_]+", "_", indices_str).strip(
                    "_"
                )
            extracted_var = f"{tracked_name} = {from_}{indices_str}\n    "
            extracted_params.append(extracted_var)
            self.id_name_db[mem_id] = tracked_name
            self.fn_counter[tracked_name] = self.fn_counter.get(tracked_name, 0) + 1
        return extracted_params

    def generate_fn_output(self, graph=None) -> Tuple[str, str]:
        """Generates the final line of the traced function. This will have the format:

        "return p1234, p4321, ..."
        """
        output_reprs = graph._output[glob.current_trace_mode]
        ids = graph._output_param_ids[glob.current_trace_mode]
        idxs = graph._output_tracked_idxs[glob.current_trace_mode]

        # replace tracked parameters with their ids
        output_reprs = ivy.set_nest_at_indices(output_reprs, idxs, ids, shallow=False)
        # Replace non tracked vars with a variable that stores their value
        output_reprs = self.replace_nontracked_from_reprs(output_reprs, ids, idxs)
        # Replace tracked args with the corresponding intermediate variable
        output_reprs = ivy.map_nest_at_indices(
            output_reprs,
            idxs,
            lambda x: f"c{x}" if f"c{x}" in self.constants else self.id_name_db[x],
            shallow=False,
        )

        # store subclasses as context variables
        output_reprs_list = [output_reprs]
        subclass_idx = ivy.nested_argwhere(
            output_reprs_list,
            lambda x: isinstance(x, (list, tuple, dict))
            and type(x) not in [list, tuple, dict],
            check_nests=True,
        )
        subclass_values = ivy.multi_index_nest(output_reprs_list, subclass_idx)
        subclass_names = [f"init_{type(x).__name__}" for x in subclass_values]
        subclass_inits = [type(x) for x in subclass_values]
        # replace args with the new variables
        output_reprs = nest_to_str(output_reprs, subclass_names, subclass_inits)
        output_reprs = self.avoid_sequence_unpacking(output_reprs)
        # for hf transformers, when transpiling to jax,
        # we want to convert the subclass output to be jax.jit compatible here
        subclass_reprs = ""
        if graph._transpiling and graph.backend == "jax" and len(subclass_idx) > 0:
            jsubclass_names = ['_'.join(x.split('_')[1:]) for x in subclass_names]
            jsubclass_names = [
                f"Flax{x}" for x in subclass_names
            ]
            # modify output reprs to represent the new subclasses
            for i, x in enumerate(subclass_names):
                output_reprs = output_reprs.replace(x, jsubclass_names[i])
            # generate source str corresponding to the class definitions
            subclass_reprs += f"import collections\n"
            jsubclass_reprs = [
                generate_subclass_init(jsubclass_names[i], x)
                for i, x in enumerate(subclass_values)
            ]
            jsubclass_reprs = "\n\n".join(jsubclass_reprs)
            subclass_reprs += jsubclass_reprs
        else:
            # add variables to function context
            self.constants.update(dict(zip(subclass_names, subclass_inits)))
        return f"return {output_reprs}", subclass_reprs

    def generate_source(
        self,
        frontend: str = None,
        graph=None,
        fn_name: str = "traced_fn",
        indent: int = 0,
    ) -> str:
        """Generates a string containing the source file for the traced call."""
        from tracer.graph import LazyGraph, SubGraph

        self.constants.update(graph.constants_in_output)
        # source file header
        fn_header = ""
        if frontend is not None and graph is not None and graph.contains_truncations:
            fn_header += "import ivy\n"
            fn_header += f"from ivy.functional.frontends.{frontend}.func_wrapper import to_ivy_arrays_and_back\n"
        frontend_header = "import ivy.functional.frontends.<fw> as <fw>\n"
        if indent == 0:
            if frontend:
                # ToDo: Only update the header if we are transpiling
                fn_header += frontend_header.replace("<fw>", frontend)
            elif not graph._to_ivy:
                fn_header += f"import {graph.backend}\n"
                if graph.backend == "jax":
                    fn_header += f"import jaxlib\n"
            else:
                fn_header += f"import ivy\n"
            if graph._with_numpy:
                if frontend:
                    fn_header += frontend_header.replace("<fw>", "numpy")
                else:
                    fn_header += f"import numpy\n"
            if glob.check_frontend_vs_original_differences:
                fn_header += "import ivy\n"
                fn_header += f"import tracer.globals as glob\n"
            if eval(os.getenv("CHECK_TRANSPILER_OVERHEAD", "False")):
                fn_header += f"from time import perf_counter\n"
                fn_header += f"import tracer.globals as glob\n"
        # function signature

        fn_signature = f"\ndef {fn_name}(*args, **kwargs):\n"
        # function variables
        fn_variables = ""
        # graph args and kwargs
        tracked_in_args = self.generate_vars_from("args", graph)
        tracked_in_kwargs = self.generate_vars_from("kwargs", graph)
        if graph._is_subgraph:
            # include ids which originate in the main graph named as
            # themselves -these will be converted to the correct names
            # during source_gen of the main graph
            for var_id in graph._greater_scope_ids:
                if var_id not in self.id_name_db:
                    self.id_name_db[var_id] = str(var_id)
        # add the tracked params from iterables in args
        for tracked_var in tracked_in_args + tracked_in_kwargs:
            fn_variables += tracked_var
        # stateful
        stateful_params = []
        for i, sid in enumerate(graph._stateful_param_ids):
            name = f"stateful_{i}"
            stateful_params.append(name)
            self.id_name_db[sid] = name
        self.constants.update(
            dict(
                zip(
                    stateful_params,
                    graph._stateful,
                )
            )
        )
        # function body
        fn_body = ""
        nested_fn_body = ""
        # Iterate over every registered fn
        for f in graph._functions:
            if f.__name__ == "vmap":
                vectorized_fn_str, nested_fn_body, fn_header = generate_vmap_subgraph(
                    self, graph, f, frontend, indent, nested_fn_body, fn_header
                )
                inner_fn_body = self.generate_inner_fn(
                    f, frontend=frontend, graph=graph
                )
                inner_fn_body = get_unindented_stmts(inner_fn_body)
                fn_body += vectorized_fn_str + inner_fn_body + "\n"
            else:
                inner_fn_body = self.generate_inner_fn(
                    f, frontend=frontend, graph=graph
                )
                get_unindented_stmts(inner_fn_body)
                fn_body += inner_fn_body

        # function output
        fn_output = ""
        graph_output_str, subclass_output_str = self.generate_fn_output(graph)
        fn_output += graph_output_str
        self.clean_constants()

        callback_fns = ""
        if len(graph._id_to_subgraphs[glob.current_trace_mode]) > 0:
            # loop through the constants here, looking for instances of LazyGraph, where one exists
            # (and is initialized), get its sourcecode and add to callback_fns (which is added to body later)
            # then add a reference to this rather than a const and remove it from the constants
            constants_to_add = []
            constants_to_delete = []

            for c_id, const_val in self.constants.items():
                if (
                    isinstance(const_val, LazyGraph)
                    and const_val._initialized
                    and isinstance(const_val._eager_graph, SubGraph)
                ):
                    for k, v in const_val._eager_graph.constants.items():
                        constants_to_add.append((k, v))

                    # name the generated function after the original; index incase of duplicates
                    callback_fn_name = const_val._eager_graph.__name__
                    callback_fn_name = (
                        "callback"
                        if callback_fn_name == "<lambda>"
                        else callback_fn_name
                    )
                    self.fn_counter[callback_fn_name] = (
                        self.fn_counter.get(callback_fn_name, 0) + 1
                    )
                    callback_fn_name = self.generate_styled_o_name(callback_fn_name, 0)

                    # add the generated function to the string of all generated callbacks
                    callback_fns += (
                        "\ndef "
                        + callback_fn_name
                        + const_val._eager_graph.obtain_sourcecode()[0].split(
                            "def traced_fn"
                        )[-1]
                        + "\n"
                    )

                    # keep a reference of the id to callback fn name,
                    # so it can be directly referenced in the source code
                    self.id_name_db[c_id] = callback_fn_name
                    fn_body = fn_body.replace(str(c_id), callback_fn_name)
                    constants_to_delete.append(c_id)

            # delete any callbacks from the constants
            for c_id in constants_to_delete:
                del self.constants[c_id]

            # add callback subgraph constants to the main graph constants
            for k, v in constants_to_add:
                self.constants[k] = v

        if callback_fns:
            callback_fns += "\n"

        if indent == 0:  # no need to add args & kwargs in scalar fn
            for c_id in self.constants:
                if not graph._is_subgraph:
                    fn_variables += f"{c_id} = kwargs['{c_id}']\n    "

                # replace any remaining ids within the callback functions with their
                # constant version defined within the scope of the main graph
                raw_id = c_id.replace("c", "")
                if raw_id in callback_fns and c_id not in callback_fns:
                    callback_fns = callback_fns.replace(raw_id, c_id)

        # fn_str stores the traced function definition
        fn_variables = get_unindented_stmts(fn_variables)
        fn_output = get_unindented_stmts(fn_output)
        fn_body = get_unindented_stmts(fn_body)

        if "scipy" in fn_body:
            fn_header += f"import scipy\n"
        if "math" in fn_body:
            fn_header += f"import math\n"
        if "torchvision" in fn_body:
            fn_header += (
                frontend_header.replace("<fw>", "torchvision")
                if frontend
                else f"import torchvision\n"
            )
        if "jax._src" in fn_body and frontend is None:
            fn_header += "import jax._src as _src\n"
            fn_body = fn_body.replace("jax._src", "_src")
        if subclass_output_str:
            fn_header += subclass_output_str + "\n"
        if callback_fns:
            for identifier, name in self.id_name_db.items():
                # replace any ids with their names, where applicable
                callback_fns = callback_fns.replace(str(identifier), name)
        for special_import in self.special_imports:
            fn_header += special_import + "\n"

        # ensure that while loop indentation is correct
        fn_body = fn_body.replace("== True:\n", "== True:\n    ")

        fn_str = (
            fn_header
            + fn_signature
            + textwrap.indent(
                nested_fn_body + fn_variables + callback_fns + fn_body + fn_output,
                " " * 4,
            )
        )

        if graph._is_subgraph:
            graph._id_name_db = self.id_name_db

        return fn_str

    def clean_constants(self):
        """Removes integers and certain strings from the constants dictionary"""
        for k, v in list(self.constants.items()):
            if should_remove_constant(v) or (isinstance(k, str) and "'" in k):
                del self.constants[k]

    def generate_inner_fn(self, fn: Callable, frontend: str = None, graph=None) -> str:
        """Generates a string which corresponds to a single functional node of the graph."""
        if fn.args is None and fn.kwargs is None:
            return ""

        # function name -> needs to be correctly formatted
        if fn.__name__ == "vmap":
            fn_path = "vectorized_fn"
        else:
            fn_path = get_fn_name(fn, backend=graph.backend)

        # special cases where directly importing the module doesn't work
        # e.g. tensorflow.python.ops.resource_variable etc.
        if frontend is None:
            fn_path_split = fn_path.rsplit(".", 2)
            if (
                len(fn_path_split) == 3
                and fn_path_split[0] in glob.FN_PATH_TO_IMPORT_PATH
            ):
                import_path = glob.FN_PATH_TO_IMPORT_PATH[fn_path_split[0]]
                import_path = import_path.replace("<placeholder>", fn_path_split[1])
                self.special_imports.add(import_path)
                fn_path = ".".join(fn_path_split[1:])
        fn_name = fn_path
        if frontend:
            paths_to_replace = glob.NATIVE_TO_FRONTEND_PATH
            if any([p in fn_name for p in paths_to_replace]):
                old_path, new_path = [
                    (k, v) for k, v in paths_to_replace.items() if k in fn_name
                ][0]
                fn_name = fn_name.replace(old_path, new_path)

        # function type -> can be a method or a function
        is_getattr = fn_name in ["__getattr__", "__getattribute__"]
        is_input_to_output = fn_name.split(".")[-1] == "input_to_output"
        is_method = (
            (fn.inplace_fn or fn_name[:2] == "__" or fn.from_tracked_var)
            and not is_input_to_output
            and "tvp" not in fn.__name__
        )

        # convert a function call to a method call where possible, such as
        # numpy.ndarray.astype(x, c4350497972) -> x.astype(c4350497972)
        if not frontend and any([
            prefix + "." in fn_name for prefix in glob.FUNCTION_TO_METHOD_CONVERSIONS
        ]):
            is_method = True
            fn_name = fn_name.split(".")[-1]

        is_binary_operator = fn_name.split(".")[-1] in BINARY_OPERATORS
        is_inplace = fn.inplace_fn
        is_pythonic_while_loop = (
            fn.__name__ == "while_loop" and ivy.current_backend_str() in ["torch"]
        )  # TODO: extend this for numpy and paddle

        # args -> we may have to remove the first one if calling a method
        args_str = self.generate_inner_fn_args(fn, from_="args")
        kwargs_str = self.generate_inner_fn_args(fn, from_="kwargs")
        args_n_kwargs_str = join_args_n_kwargs(args_str, kwargs_str)

        # generate a while loop in a pythonic rather than functional way
        if is_pythonic_while_loop:
            cond_fn_str = args_str.split(", ")[0]
            body_fn_str = args_str.split(", ")[1]
            loop_args_str = args_str.replace("[", "").replace("]", "").split(", ")[2:]
            loop_args_str = ", ".join(loop_args_str)
            output_str = self.generate_inner_fn_output(fn, fn_name)

            fn_str = (
                f"\n{output_str} = {loop_args_str}\n"
                f"while {cond_fn_str}(*{output_str}) == True:\n"
                f"{output_str} = {body_fn_str}(*{output_str})\n\n"
            )
            return fn_str

        if is_binary_operator and not frontend:
            op = BINARY_OPERATORS[fn_name.split(".")[-1]]
            fn_name = f".{fn_name}"
        elif (
            is_method or is_inplace
        ) and fn.__name__ in glob.INPLACE_FUNCTIONS_WITHOUT_RET[graph.backend]:
            instance, _ = method_args_from(args_n_kwargs_str)
        elif is_method or is_inplace:
            instance, args_n_kwargs_str = method_args_from(args_n_kwargs_str)
            fn_name = f"{instance}.{fn_name}"
        if is_input_to_output:
            args_n_kwargs_str = args_n_kwargs_str.split(",")[0]
        # output -> can be inplace or not
        output_str = self.generate_inner_fn_output(fn, fn_name)
        # delete any intermediate var that won't be used again
        del_statement = ""
        to_delete = []
        for a in fn.arg_param_ids + fn.kwarg_param_ids:
            self.reference_count[a] = self.reference_count[a] - 1
            if self.reference_count[a] == 0:
                arg_repr = self.id_name_db[a]
                if "[" in arg_repr and "]" in arg_repr:
                    continue
                if (
                    self.graph._is_subgraph and a not in self.graph._greater_scope_ids
                ) or (
                    not self.graph._is_subgraph
                    and a not in self.graph._callback_required_ids
                ):
                    # delete the parameter from the main graph if it is not used within a
                    # callback function or delete the parameter from the subgraph if it
                    # does not stem from the main graph
                    to_delete.append(self.id_name_db[a])
        if to_delete:
            del_statement = "del " + ", ".join(to_delete) + "\n    "
        # return the function string
        if is_binary_operator and not frontend:
            args = args_n_kwargs_str.split(",", 1)
            fn_str = f"{output_str} = {args[0]} {op} {args[1].strip()}\n    "
        elif hasattr(fn, "is_builtin_callable") and fn.is_builtin_callable:
            fn_str = f"{output_str} = {fn_name}({args_n_kwargs_str})\n    "
        elif is_inplace:
            fn_str = f"{fn_name}({args_n_kwargs_str})\n    "
            fn_str += f"{output_str} = {instance}\n    "
        elif is_getattr:
            fn_str = f"{output_str} = getattr({instance}, {args_n_kwargs_str})\n    "
        elif is_input_to_output:
            fn_str = f"{output_str} = {args_n_kwargs_str}\n    "
        elif hasattr(fn, "is_ivy") and fn.is_ivy:
            fn_str = f"{fn_name} = to_ivy_arrays_and_back(ivy.{fn_name})\n    {output_str} = {fn_name}({args_n_kwargs_str})\n    "
        else:
            fn_str = f"{output_str} = {fn_name}({args_n_kwargs_str})\n    "
        if glob.check_frontend_vs_original_differences:
            if frontend:
                fn_str += f"cloned = ivy.copy_array({output_str}.ivy_array, to_ivy_array=False) if hasattr({output_str}, 'dtype') else {output_str}\n    "
                fn_str += f"glob.frontend_results.append(cloned)\n    "
            else:
                fn_str += f"cloned = ivy.copy_array({output_str}, to_ivy_array=False) if hasattr({output_str}, 'dtype') else {output_str}\n    "
                fn_str += f"glob.original_results.append(cloned)\n    "
        if eval(os.getenv("CHECK_TRANSPILER_OVERHEAD", "False")):
            if not self.graph._transpiling:
                if not frontend:
                    fn_str = "s = perf_counter()\n    " + fn_str
                    fn_str += (
                        f"glob.times[{(fn_path, fn.id_)}] = [perf_counter() - s]\n    "
                    )
                else:
                    fn_str = (
                        f"glob.node_expansion[{(fn_path, fn.id_)}] = []\n    " + fn_str
                    )
                    fn_str = (
                        f"glob.current_frontend = {(fn_path, fn.id_)}\n    " + fn_str
                    )
            elif graph.backend != "ivy":
                fn_str = "s = perf_counter()\n    " + fn_str
                fn_str += f"glob.transpiled_times[{fn.from_}].append(perf_counter() - s)\n    "
        return fn_str + del_statement

    def avoid_sequence_unpacking(self, reprs):
        """Avoid source like `[unstack[0], unstack[1]]` by replacing it with `unstack`."""
        for k, v in self.unpacked_to_original.items():
            reprs = reprs.replace(k, v)
        return reprs

    def generate_inner_fn_args(self, f: Callable, from_: str) -> str:
        """Generates a string which contains the args of the specified function.
        This function also stores any constant variable in the self.constant dict."""
        if from_ not in ["args", "kwargs"]:
            raise ValueError("from_ must be one of 'args' or 'kwargs'.")
        from_args = from_ == "args"
        args = f.args if from_args else f.kwargs
        if args is None:
            return ""
        idxs = f.arg_tracked_idxs if from_args else f.kwarg_tracked_idxs
        ids = f.arg_param_ids if from_args else f.kwarg_param_ids
        # Replace tracked args with their ids
        args_reprs = ivy.set_nest_at_indices(args, idxs, ids, shallow=False)
        # Replace non tracked args with a variable that stores their value
        args_reprs = self.replace_nontracked_from_reprs(args_reprs, ids, idxs)
        # Replace tracked args with the corresponding intermediate variable
        args_reprs = ivy.map_nest_at_indices(
            args_reprs, idxs, lambda x: self.id_name_db[x]
        )
        # store subclasses as context variables
        subclass_idx = ivy.nested_argwhere(
            args_reprs,
            lambda x: isinstance(x, (list, tuple, dict))
            and type(x) not in [list, tuple, dict],
            check_nests=True,
        )
        subclass_values = ivy.multi_index_nest(args_reprs, subclass_idx)
        subclass_names = [f"init_{type(x).__name__}" for x in subclass_values]
        subclass_types = [type(x) for x in subclass_values]
        # add variables to function context
        self.constants.update(dict(zip(subclass_names, subclass_values)))
        # Fabricate the args string
        _slice_idxs = f.with_tracked_slices
        args_reprs = nest_to_str(
            args_reprs, subclass_names, subclass_types, _slice_idxs
        )
        args_reprs = self.avoid_sequence_unpacking(args_reprs)
        return args_reprs

    def replace_nontracked_from_reprs(self, reprs, param_ids, tracked_idxs):
        # get indices of the values in args that are not tracked
        nontracked_idxs = ivy.nested_argwhere(
            reprs,
            lambda x: type(x) is not int or x not in param_ids,
            check_nests=True,
        )
        nestables_idxs = ivy.nested_argwhere(
            reprs,
            lambda x: isinstance(x, (list, tuple, dict)),
            check_nests=True,
        )
        # remove nestables from constants if not all inner values are constants
        nontracked_idxs = remove_non_constant_nests_idxs(
            nontracked_idxs, nestables_idxs, tracked_idxs
        )
        # get values at said indices
        nontracked_values = ivy.multi_index_nest(reprs, nontracked_idxs)
        # generate a unique id for each value
        nontracked_ids = [
            (
                str(x)
                if should_remove_constant(x)
                else f"'{x}'" if isinstance(x, str) else f"c{_get_unique_id(x)}"
            )
            for x in nontracked_values
        ]
        # add variables to function context
        context = dict(zip(nontracked_ids, nontracked_values))
        self.constants.update(context)
        # replace reprs with the new variables
        reprs = ivy.set_nest_at_indices(reprs, nontracked_idxs, nontracked_ids)
        return reprs

    @staticmethod
    def generate_styled_fn_name(fn_name):
        """
        Generates styled names for functions. Appends "_var" when
        name belongs to builtin python names

        Examples
        --------
        >>> generate_styled_fn_name('torch.Tensor.__rmul__')
        rmul

        >>> generate_styled_fn_name('torch.add')
        add
        """
        match = re.search(r"\.__(.*?)__", fn_name)
        if match:
            fn_name_styled = match.group(1)
        elif "." not in fn_name:
            fn_name_styled = "_" + fn_name
        else:
            fn_name_styled = fn_name.split(".")[-1]
        if fn_name_styled in keyword.kwlist + dir(builtins) + [
            "torch",
            "tensorflow",
            "numpy",
            "jax",
            "scipy",
            "paddle",
        ]:
            fn_name_styled = f"{fn_name_styled}_var"
        return fn_name_styled

    def generate_styled_o_name(self, fn_name_styled, output_idx):
        """
        Generates styled names for outputs with counter to be inline
        with torch.fx convention. Counter is added only when var name is
        used more than once.

        Examples
        --------
        # First torch.add
        >>> generate_styled_o_name('add', _)
        add

        # Second torch.add - can't have same output name as before
        >>> generate_styled_o_name('add', _)
        add_1
        """
        if self.fn_counter[fn_name_styled] == 1:
            name = f"{fn_name_styled}"
        else:
            name = f"{fn_name_styled}_{self.fn_counter[fn_name_styled]-1}"
        if output_idx == 0:
            return name
        else:
            return name + f"_o{output_idx}"

    def generate_inner_fn_output(self, f: Callable, fn_name: str) -> str:
        fn_name_styled = self.generate_styled_fn_name(fn_name)
        # check for nestables in the output reprs
        nestables_idxs = ivy.nested_argwhere(
            f.output,
            lambda x: isinstance(x, (list, tuple, dict)),
            check_nests=True,
        )
        if nestables_idxs:

            def _generate_indexing_reprs(indices):
                ret = ""
                for idx, index in enumerate(indices):
                    if idx == 0:
                        continue
                    if not str(index).isnumeric():
                        ret += f'["{index}"]'
                    else:
                        ret += f"[{index}]"
                return ret

            self.fn_counter[fn_name_styled] = self.fn_counter.get(fn_name_styled, 0) + 1
            name = self.generate_styled_o_name(fn_name_styled, 0)
            tracked_idx_reprs = [
                _generate_indexing_reprs(tracked_idx)
                for tracked_idx in f.output_tracked_idxs
            ]
            full_nest = ()
            for idx, idx_repr in enumerate(tracked_idx_reprs):
                assert name + idx_repr not in self.id_name_db.values()
                self.id_name_db[f.output_param_ids[idx]] = name + idx_repr
                full_nest += (name + idx_repr,)
            full_nest_list = "[" + ", ".join(full_nest) + "]"
            full_nest_tuple = "(" + ", ".join(full_nest) + ")"
            self.unpacked_to_original[full_nest_list] = name
            self.unpacked_to_original[full_nest_tuple] = name
            return name

        output_reprs = ivy.set_nest_at_indices(
            f.output, f.output_tracked_idxs, f.output_param_ids, shallow=False
        )
        # output_reprs can be nested --> [[...], [...]] while
        # output_param_ids will always be flat --> [<param_id_1>, <param_id_2>]
        # so need to compare the two lists keeping the nesting in mind
        reprs = []
        flat_reprs = helpers.flatten(output_reprs)
        for i, o in enumerate(flat_reprs):
            self.fn_counter[fn_name_styled] = self.fn_counter.get(fn_name_styled, 0) + 1
            if o in f.output_param_ids:
                name = self.generate_styled_o_name(fn_name_styled, i)
                # currently old deleted var names aren't reused so following
                # assertion acts as a sanity check but can be removed for efficieny
                assert name not in self.id_name_db.values()
                self.id_name_db[o] = name
                reprs.append(name)
            else:
                reprs.append("_")
        output_reprs = ", ".join(reprs) if reprs else "_"
        return output_reprs


# Helpers


def get_unindented_stmts(text):
    lines = text.splitlines()
    unindented_lines = [line.lstrip() for line in lines]
    unindented_text = "\n".join(unindented_lines)
    return unindented_text


def join_args_n_kwargs(args_str: str, kwargs_str: str) -> str:
    """Generates a string which contains args and kwargs correctly formatted to script a function.
    Parameters
    ----------
    args_str : str
        String containing the arguments of a function. (i.e. "x1, x2").
    kwargs_str : str
        String containing the keyword arguments of a function. (i.e. "kw1=v1, kw2=v2").
    Returns
    -------
    str
        Correctly formatted arguments and keyword arguments.
        (i.e. "x1, x2, kw1=v1, kw2=v2").
    """
    valid_args_n_kwargs = [i for i in [args_str, kwargs_str] if i]
    return ", ".join(valid_args_n_kwargs)


def nest_to_str(
    nest, _inits=None, _init_types=None, _slice_idxs=None, _base=True, _index=None
):
    """Takes a nestable which holds strings and integers and correctly formats the nested structure into
    a final string."""
    # all arguments should have been already converted to strings
    _inits = ivy.default(_inits, [])
    _init_types = ivy.default(_init_types, [])
    _slice_idxs = ivy.default(_slice_idxs, [])
    _index = [] if _base else _index
    is_subclass_init = type(nest) in _init_types
    is_slice = _index in _slice_idxs
    if isinstance(nest, (tuple, list)):
        if isinstance(nest, list):
            opening, closing = "[", "]"
        elif isinstance(nest, tuple) and not hasattr(nest, "_fields"):
            opening, closing = "(", ")"
        _args = [
            nest_to_str(item, _inits, _init_types, _slice_idxs, False, _index + [i])
            for i, item in enumerate(nest)
        ]
        if len(_args) == 1 and not _base:
            _args[0] += ","
    elif isinstance(nest, dict):
        opening, closing = "{", "}"
        union = ": " if not _base else "="
        union = "=" if is_subclass_init else union
        _args = {
            k: nest_to_str(v, _inits, _init_types, _slice_idxs, False, _index + [k])
            for k, v in nest.items()
        }
        if isinstance(nest, ivy.Container):  # ToDo: Move to _inits?
            opening, closing = "ivy.Container(", ")"
            union = "="
        # Wrap raw dict keys in string quotations
        _args = [
            (
                f"'{k}'{union}{v}"
                if isinstance(k, str) and union == ": "
                else f"{k}{union}{v}"
            )
            for k, v in _args.items()
        ]
    else:
        if not isinstance(nest, str):
            return str(nest)
        return nest
    if _base:
        opening, closing = "", ""
    if is_slice:
        opening, closing = "slice(", ")"
    if is_subclass_init:
        idx = _init_types.index(type(nest))
        subclass_name = _inits[idx]
        opening, closing = f"{subclass_name}(", ")"
        # TODO: Add check for normal namedtuples
        if "torch.return_types" in str(type(nest)):
            opening = opening + "["
            closing = "]" + closing
    formatted_args = ', '.join(_args)
    repr = f"{opening}{formatted_args}{closing}"
    return repr


def get_fn_name(fn: Callable, backend: str) -> str:
    "Gets the correct function name from a given function."
    if fn.__name__ in [
        "__getattribute__",
        "__getattr__",
        "__getitem__",
        "__setitem__",
    ]:
        return fn.__name__
    elif hasattr(fn, "is_builtin_callable") and fn.is_builtin_callable:
        return fn.path if hasattr(fn, "path") else fn.__name__
    if fn.inplace_fn and fn.__name__ not in glob.INPLACE_FUNCTIONS_WITHOUT_RET[backend]:
        return fn.backend_fn.__name__
    if hasattr(fn, "path"):
        return fn.path
    else:
        fn_path = fn.backend_fn.__name__
    if "tvp__" in fn_path:
        fn_path = fn_path[5:-2]
    elif fn.from_tracked_var:
        return fn.__name__
    return fn_path


def method_args_from(args_n_kwargs):
    args = args_n_kwargs.split(", ")
    instance = args[0]
    method_args = ", ".join(args[1:])
    return instance, method_args


def _exec_with_source(src: str, globals):
    key = _loader.cache(src, globals)
    exec(compile(src, key, "exec"), globals)


def load_fn_from_str(source_str):
    """Executes the source code passed as arguments and returns the defined "traced_fn" """
    namespace = {}
    _exec_with_source(source_str, namespace)
    namespace["traced_fn"].source_code = source_str[source_str.find("def traced") :]
    traced_fn = namespace["traced_fn"]
    return traced_fn


def load_fn_from_file(source_str):
    """Saves the generated source code into a file and imports said file as a module.
    This allows the user to step into the (scripted) traced function."""
    # ToDo: fix path of intermediate file
    file_path = "ivy_temp_script.py"
    module_name = "ivy_traced_fn"
    with open(file_path, "w") as f:
        f.write(source_str)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module at {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        traced_fn = module.__dict__["traced_fn"]
    except:
        raise ImportError("Error while loading the traced function as a module.")
    return traced_fn


def remove_non_constant_nests_idxs(nontracked_idxs, nestables_idxs, tracked_idxs):
    # remove any nested types idxs from nontracked if there is any tracked param inside it
    to_remove = []
    for nest_idx in nestables_idxs:
        for idx in tracked_idxs:
            if idx[: len(nest_idx)] == nest_idx:
                to_remove.append(nest_idx)
                break
    for nidx in to_remove:
        nontracked_idxs.remove(nidx)
        nestables_idxs.remove(nidx)
    # remove any inner idxs from nontracked if its outer nest is constant
    redundant = []
    for nidx in nontracked_idxs:
        for nest_idx in nestables_idxs:
            if nidx != nest_idx and nidx[: len(nest_idx)] == nest_idx:
                redundant.append(nidx)
                break
    [nontracked_idxs.remove(nidx) for nidx in redundant]
    return nontracked_idxs


def generate_subclass_init(clsname, instance):
    # generate code string for making HF transformers jax.jit compatible
    from dataclasses import fields

    if isinstance(instance, dict):
        # OrderedDict
        code_string = "import flax\nimport typing\n\n\n"
        code_string += f"@flax.struct.dataclass\n"
        code_string += f"class {clsname}(collections.OrderedDict):"
        for field in fields(instance):
            code_string += "\n"
            code_string += f"    {field.name}: typing.Any = None"
        code_string += "\n"
    elif isinstance(instance, tuple) and hasattr(instance, "_fields"):
        # NamedTuple
        code_string = (
            f"\n\n{clsname} = collections.namedtuple('{clsname}', {instance._fields})\n"
        )
    return code_string


def should_remove_constant(constant):
    """Whether a constant can be removed from the constants dictionary. We remove the
    constant if the representation of constant itself is the same as the code that
    would generate it.

    For example, the integer object `1` would give `True` as the representation of `1`
    i.e. `str(1)` gives `"1"`, which is the correct code to produce the integer `1`.
    """
    if isinstance(constant, slice) and any(
        not isinstance(x, (int, type(None)))
        for x in [constant.start, constant.stop, constant.step]
    ):
        # do not remove slices if they contain none int/None values, such as tensors
        return False

    return (
        isinstance(constant, (bool, int, slice, EllipsisType, type(None)))
        or (
            isinstance(constant, float)
            and not (math.isinf(constant) or math.isnan(constant))
        )
        or (
            type(constant) in (tuple, list)
            and all(should_remove_constant(x) for x in constant)
        )
    )
