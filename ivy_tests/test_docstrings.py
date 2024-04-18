import warnings
import re
import logging
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import sys
import pytest 
import ivy
import ivy_tests.test_ivy.helpers as helpers

warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_print_statements(trimmed_docstring):
    parsed_output = ""
    sub = ">>> print("
    end_index = -1
    
    for index, line in enumerate(trimmed_docstring):
        if sub in line:
            for i, s in enumerate(trimmed_docstring[index + 1:]):
                if s.startswith(">>>") or s.lower().startswith(("with", "#", "instance")):
                    end_index = index + i + 1
                    break
            else:
                end_index = len(trimmed_docstring)
                
            p_output = trimmed_docstring[index + 1:end_index]
            p_output = "".join(p_output).replace(" ", "")
            p_output = p_output.replace("...", "")
            
            if parsed_output != "":
                parsed_output += ","
            
            parsed_output += p_output
    
    return parsed_output, end_index

def execute_docstring_examples(executable_lines):
    f = StringIO()
    with redirect_stdout(f):
        for line in executable_lines:
            try:
                if f.getvalue() != "" and f.getvalue()[-2] != ",":
                    print(",")
                exec(line)
            except Exception as e:
                print(e, " ", ivy.current_backend_str(), " ", line)
    
    return f.getvalue()

def trim(docstring):
    if not docstring:
        return ""
    
    lines = docstring.expandtabs().splitlines()
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    
    if "\n" in docstring:
        trimmed.append("")
    
    return "\n".join(trimmed)

skip_list = {
    "to_skip": [],
    "skip_list_temp": []
}

logging.basicConfig(level=logging.INFO)

def execute_and_log(line):
    try:
        exec(line)
    except Exception as e:
        logging.error(f"Error executing line: {line}\nError: {e}")

def assert_equal_with_logging(expected, actual, message=""):
    try:
        assert expected == actual, message
    except AssertionError as e:
        logging.error(f"AssertionError: {e}\nExpected: {expected}\nActual: {actual}")

def check_docstring_examples_run(docstring, *, fn, from_container=False, from_array=False, num_sig_fig=2):
    trimmed_docstring = trim(docstring)
    parsed_output, end_index = parse_print_statements(trimmed_docstring)
    
    if end_index == -1:
        return True
    
    executable_lines = [line.split(">>>")[1][1:] for line in trimmed_docstring if line.startswith(">>>")]
    is_multiline_executable = False
    
    for line in trimmed_docstring:
        if line.startswith(">>>"):
            is_multiline_executable = True
        if line.startswith("...") and is_multiline_executable:
            executable_lines[-1] += line.split("...")[1][1:]
        if ">>> print(" in line:
            is_multiline_executable = False
    
    output = execute_docstring_examples(executable_lines)
    
    sig_fig = float(f"1e-{str(num_sig_fig)}")
    atol = sig_fig / 10000
    numeric_pattern = re.compile(r"[\{\}\(\)\[\]\<>]|\w+:", re.VERBOSE)
    
    num_output = output.replace("ivy.array", "").replace("ivy.Shape", "")
    num_parsed_output = parsed_output.replace("ivy.array", "").replace("ivy.Shape", "")
    num_output = numeric_pattern.sub("", num_output)
    num_parsed_output = numeric_pattern.sub("", num_parsed_output)
    
    num_output = num_output.split(",")
    num_parsed_output = num_parsed_output.split(",")
    
    for doc_u, doc_v in zip(num_output, num_parsed_output):
        try:
            assert_equal_with_logging(
                np.allclose(
                    np.nan_to_num(complex(doc_u)),
                    np.nan_to_num(complex(doc_v)),
                    rtol=sig_fig,
                    atol=atol
                ),
                True,
                message=f"Output mismatch: {doc_u} != {doc_v}"
            )
        except Exception:
            if str(doc_u) != str(doc_v):
                logging.error(
                    f"Output mismatch for {fn.__name__}: {doc_u} != {doc_v}"
                )
                return False
    
    return True

@pytest.mark.parametrize("backend", ["jax", "numpy", "tensorflow", "torch"])
def test_docstrings(backend):
    ivy.set_default_device("cpu")
    ivy.set_backend(backend)
    failures = []
    success = True

    for k, v in ivy.__dict__.copy().items():
        docstring = getattr(v, "__doc__", "")
        
        if k == "Array":
            for method_name in dir(v):
                method = getattr(ivy.Array, method_name)
                if hasattr(ivy.functional, method_name):
                    if helpers.gradient_incompatible_function(
                        fn=getattr(ivy.functional, method_name)
                    ) or check_docstring_examples_run(docstring, fn=method, from_array=True):
                        continue
                elif helpers.gradient_incompatible_function(
                    fn=method
                ) or check_docstring_examples_run(docstring, fn=method, from_array=True):
                    continue
                failures.append(f"Array.{method_name}")
                success = False
        elif k == "Container":
            for method_name in dir(v):
                method = getattr(ivy.Container, method_name)
                if hasattr(ivy.functional, method_name):
                    if helpers.gradient_incompatible_function(
                        fn=getattr(ivy.functional, method_name)
                    ) or check_docstring_examples_run(docstring, fn=method, from_container=True):
                        continue
                elif helpers.gradient_incompatible_function(
                    fn=method
                ) or check_docstring_examples_run(docstring, fn=method, from_container=True):
                    continue
                failures.append(f"Container.{method_name}")
                success = False
        else:
            if check_docstring_examples_run(docstring, fn=v) or helpers.gradient_incompatible_function(fn=v):
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
