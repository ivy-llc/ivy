# global
import warnings
import re
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import sys


warnings.filterwarnings("ignore", category=DeprecationWarning)
import pytest

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers


# function that trims white spaces from docstrings
def trim(*, docstring):
    """Trim function from PEP-257."""
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Current code/unittests expects a line return at
    # end of multiline docstrings
    # workaround expected behavior from unittests
    if "\n" in docstring:
        trimmed.append("")

    return "\n".join(trimmed)


def check_docstring_examples_run(
    *, fn, from_container=False, from_array=False, num_sig_fig=2
):
    """Performs docstring tests for a given function.

    Parameters
    ----------
    fn
        Callable function to be tested.
    from_container
        if True, check docstring of the function as a method of an Ivy Container.
    from_array
        if True, check docstring of the function as a method of an Ivy Array.
    num_sig_fig
        Number of significant figures to check in the example.

    Returns
    -------
    None if the test passes, else marks the test as failed.
    """
    """
    Functions skipped as their output dependent on outside factors:

    random_normal, random_uniform, shuffle, num_gpus, current_backend,
    get_backend
    """
    to_skip = [
        "random_normal",
        "random_uniform",
        "randint",
        "shuffle",
        "beta",
        "gamma",
        "dev",
        "num_gpus",
        "current_backend",
        "get_backend",
        "namedtuple",
        "invalid_dtype",
        "DType",
        "NativeDtype",
        "Dtype",
        "multinomial",
        "num_cpu_cores",
        "get_all_ivy_arrays_on_dev",
        "num_ivy_arrays_on_dev",
        "total_mem_on_dev",
        "used_mem_on_dev",
        "percent_used_mem_on_dev",
        "function_supported_dtypes",
        "function_unsupported_dtypes",
        "randint",
        "unique_counts",
        "unique_all",
        "dropout",
        "dropout1d",
        "dropout2d",
        "dropout3d",
        "total_mem_on_dev",
        "supports_inplace_updates",
        "get",
        "deserialize",
        "set_split_factor",
    ]
    # the temp skip list consists of functions
    # which have an issue with their implementation
    skip_list_temp = [
        "outer",  # Failing only torch backend as inputs must be 1-D.
        "pool",  # Maximum recursion depth exceeded ivy.pool
        "put_along_axis",  # Depends on scatter_nd for numpy.
        "result_type",  # Different ouput coming for diff backends in 1st example.
        "scaled_dot_product_attention",  # Different backends giving different answers.
        "eigh_tridiagonal",  # Failing only for TF backend
        "dct",
        "choose",  # Maximum recurion depth exceeded (No backend choose fn).
        "idct",  # Function already failing for all 5 backends.
        "set_item",  # Different errors for diff backends (jax, torch)
        "l1_normalize",  # Function already failing for all 5 backends.
        "histogram",  # Failing for TF, Torch backends (TODO's left)
        "value_and_grad",  # Failing only for Torch backend. (Requires_grad=True)
        "layer_norm",  # Failing only for Torch backend.
        "eigvalsh",  # Failing only Jax Backend + only for Native Array Example.
        "conv2d_transpose",  # Function already failing for all 5 backends.
        "solve",
        "one_hot",  # One small example failing for all backends except torch.
        "scatter_flat",  # Function Already failing for 3 backends
        "scatter_nd",  #
        "execute_with_gradients",  # Function Already failing for 4 backends.
        "gather",
        "multiprocessing",
        "if_else",
        "trace_graph",  # SystemExit: Please sign up for free pilot access.
        "dill",
        "smooth_l1_loss",  # Function already failing for all 5 backends.
        "cummax",  # Function already failing for all 5 backends.
        "insert_into_nest_at_index",
        "while_loop",
        "argmax",
        "native_array",
    ]

    # skip list for array and container docstrings
    skip_arr_cont = [
        # generates different results due to randomization
        "cumprod",
        "supports_inplace_updates",
        "shuffle",
        "dropout",
        "dropout1d",
        "dropout2d",
        "dropout3",
        "svd",
        "unique_all",
        # exec and self run generates diff results
        "dev",
        "scaled_dot_product_attention",
        # temp list for array/container methods
        "einops_reduce",
        "array_equal",
        "batched_outer",
        "huber_loss",
        "softshrink",
        "tt_matrix_to_tensor",
        "unsorted_segment_mean",
        "array_equal",
        "batched_outer",
        "huber_loss",
        "kl_div",
        "soft_margin_loss",
        "threshold",
    ]

    # comment out the line below in future to check for the functions in temp skip list
    to_skip += skip_list_temp  # + currently_being_worked_on

    if not hasattr(fn, "__name__"):
        return True
    fn_name = fn.__name__
    if fn_name not in ivy.utils.backend.handler.ivy_original_dict:
        return True

    if from_container:
        docstring = getattr(
            ivy.utils.backend.handler.ivy_original_dict["Container"], fn_name
        ).__doc__
    elif from_array:
        docstring = getattr(
            ivy.utils.backend.handler.ivy_original_dict["Array"], fn_name
        ).__doc__
    else:
        docstring = ivy.utils.backend.handler.ivy_original_dict[fn_name].__doc__
    if docstring is None:
        return True
    if fn_name in to_skip:
        return True
    if (from_container or from_array) and fn_name in skip_arr_cont:
        return True

    # removing extra new lines and trailing white spaces from the docstrings
    trimmed_docstring = trim(docstring=docstring)
    trimmed_docstring = trimmed_docstring.split("\n")
    # end_index: -1, if print statement is not found in the docstring
    end_index = -1

    # parsed_output is set as an empty string to manage functions with multiple inputs
    parsed_output = ""

    # parsing through the docstrings to find lines with print statement
    # following which is our parsed output
    sub = ">>> print("
    for index, line in enumerate(trimmed_docstring):
        if sub in line:
            for i, s in enumerate(trimmed_docstring[index + 1 :]):
                if s.startswith(">>>") or s.lower().startswith(
                    ("with", "#", "instance")
                ):
                    end_index = index + i + 1
                    break
            else:
                end_index = len(trimmed_docstring)
            p_output = trimmed_docstring[index + 1 : end_index]
            p_output = "".join(p_output).replace(" ", "")
            p_output = p_output.replace("...", "")
            if parsed_output != "":
                parsed_output += ","
            parsed_output += p_output

    if end_index == -1:
        return True

    executable_lines = []

    for line in trimmed_docstring:
        if line.startswith(">>>"):
            executable_lines.append(line.split(">>>")[1][1:])
            is_multiline_executable = True
        if line.startswith("...") and is_multiline_executable:
            executable_lines[-1] += line.split("...")[1][1:]
        if ">>> print(" in line:
            is_multiline_executable = False

    # noinspection PyBroadException
    f = StringIO()
    with redirect_stdout(f):
        for line in executable_lines:
            # noinspection PyBroadException
            try:
                if f.getvalue() != "" and f.getvalue()[-2] != ",":
                    print(",")
                exec(line)
            except Exception as e:
                print(e, " ", ivy.current_backend_str(), " ", line)

    output = f.getvalue()
    output = output.rstrip()
    output = output.replace(" ", "").replace("\n", "")
    output = output.rstrip(",")

    # handling cases when the stdout contains ANSI colour codes
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(
        r"""
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
    """,
        re.VERBOSE,
    )

    output = ansi_escape.sub("", output)

    # print("Output: ", output)
    # print("Putput: ", parsed_output)

    # assert output == parsed_output, "Output is unequal to the docstrings output."
    sig_fig = float(f"1e-{str(num_sig_fig)}")
    atol = sig_fig / 10000
    numeric_pattern = re.compile(
        r"""
            [\{\}\(\)\[\]\<>]|\w+:
        """,
        re.VERBOSE,
    )

    num_output = output.replace("ivy.array", "").replace("ivy.Shape", "")
    num_parsed_output = parsed_output.replace("ivy.array", "").replace("ivy.Shape", "")
    num_output = numeric_pattern.sub("", num_output)
    num_parsed_output = numeric_pattern.sub("", num_parsed_output)
    num_output = num_output.split(",")
    num_parsed_output = num_parsed_output.split(",")
    docstr_result = True
    for doc_u, doc_v in zip(num_output, num_parsed_output):
        try:
            docstr_result = np.allclose(
                np.nan_to_num(complex(doc_u)),
                np.nan_to_num(complex(doc_v)),
                rtol=sig_fig,
                atol=atol,
            )
        except Exception:
            if str(doc_u) != str(doc_v):
                docstr_result = False
        if not docstr_result:
            print(
                "output for ",
                fn_name,
                " on run: ",
                output,
                "\noutput in docs :",
                parsed_output,
                "\n",
                doc_u,
                " != ",
                doc_v,
                "\n",
            )
            ivy.warn(
                f"Output is unequal to the docstrings output: {fn_name}",
                stacklevel=0,
            )
            break
    return docstr_result


@pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow", "torch"])
def test_docstrings(backend):
    ivy.set_default_device("cpu")
    ivy.set_backend(backend)
    failures = []
    success = True

    for k, v in ivy.__dict__.copy().items():
        if k == "Array":
            for method_name in dir(v):
                method = getattr(ivy.Array, method_name)
                if hasattr(ivy.functional, method_name):
                    if helpers.gradient_incompatible_function(
                        fn=getattr(ivy.functional, method_name)
                    ) or check_docstring_examples_run(fn=method, from_array=True):
                        continue
                elif helpers.gradient_incompatible_function(
                    fn=method
                ) or check_docstring_examples_run(fn=method, from_array=True):
                    continue
                failures.append(f"Array.{method_name}")
                success = False
        elif k == "Container":
            for method_name in dir(v):
                method = getattr(ivy.Container, method_name)
                if hasattr(ivy.functional, method_name):
                    if helpers.gradient_incompatible_function(
                        fn=getattr(ivy.functional, method_name)
                    ) or check_docstring_examples_run(fn=method, from_container=True):
                        continue
                elif helpers.gradient_incompatible_function(
                    fn=method
                ) or check_docstring_examples_run(fn=method, from_container=True):
                    continue
                failures.append(f"Container.{method_name}")
                success = False
        else:
            if check_docstring_examples_run(
                fn=v
            ) or helpers.gradient_incompatible_function(fn=v):
                continue
            success = False
            failures.append(k)
    if not success:
        assert (
            success
        ), "\nThe following methods had failing docstrings:\n\n{}\n".format(
            "\n".join(failures)
        )

    ivy.previous_backend()
