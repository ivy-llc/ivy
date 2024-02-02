import numpy as np
from datetime import timedelta
import nbformat
import json
import os
import re


class OutputMessage:
    _set_data = True

    def __init__(self):
        self.output_type = None
        self.text = None
        self.execution_count = None

        # error attributes
        self.traceback = None
        self.evalue = None
        self.ename = None

    def set_error_attributes(self, attrs):
        self.ename = attrs["ename"]
        self.evalue = attrs["evalue"]
        self.traceback = attrs["traceback"]

    def as_dict(self):
        return {key: value for key, value in vars(self).items() if value is not None}

    @property
    def set_data(self):
        return self._set_data


def fetch_notebook_and_configs(path):
    with open(path) as f:
        notebook = nbformat.reads(f.read(), nbformat.current_nbformat)
    path_tree = path.split(os.sep)
    module, name = path_tree[-2:]
    root_path = os.sep.join(path_tree[:-2])
    module_config = json.load(
        open(os.path.join(root_path, "tests", "config.json"))
    )[module]
    config = dict()
    if name in module_config:
        config = module_config[name]
    if "install" in config:
        for setup_cell_number in config["install"]:
            for setup_line in notebook.cells[setup_cell_number - 1].source.split("\n"):
                if "exit()" in setup_line:
                    continue
                setup_command = setup_line.strip("!")
                print(f"RUN {setup_command}")
                os.system(setup_command)
    return notebook, config


def process_stream_content(out, content):
    # assuming there are no warning messages in the notebook
    if content["name"] == "stderr":
        out._set_data = False
        return
    setattr(out, "name", content.get("name"))
    out.text = content["text"]


def process_error(out, err_attrs):
    out.set_error_attributes(err_attrs)


def process_display_data(out, content):
    for mime, data in content["data"].items():
        mime_type = (
            mime.split("/")[-1].lower().replace("+xml", "").replace("plain", "text")
        )
        target_obj = out if out is not None else content
        setattr(target_obj, mime_type, data)


def record_output(msg, outs, execution_count):
    msg_type = msg["header"]["msg_type"]
    content = msg["content"]

    processing_functions = {
        "stream": process_stream_content,
        "display_data": process_display_data,
        "pyout": process_display_data,
        "execute_result": process_display_data,
        "pyerr": process_error,
        "error": process_error,
    }
    out = OutputMessage()
    processing_function = processing_functions.get(msg_type)
    if processing_function:
        processing_function(out, content)
        if out.set_data:
            out.output_type = msg_type
            out.execution_count = execution_count
            outs.append(out.as_dict())


def consolidate(outputs):
    """
    Consolidate the results from a stream into a list of lines

    Args:
        outputs:

    Returns:

    """
    output_str = ""
    execution_count = None

    for element in outputs:
        if (
            element.get("name") == "stdout"
            or element.get("output_type") == "execute_result"
        ):
            if "data" in element:
                output_str += element["data"].get("text/plain", "")
            else:
                output_str += element.get("text", "")
                execution_count = element.get("execution_count")
    return output_str, execution_count


def assert_equal(value1, value2, message):
    assert value1 == value2, message


def assert_all_close(
    *,
    test_tensor=None,
    cell_tensor=None,
    original_test_output=None,
    original_cell_output=None,
    rtol=1e-05,
    atol=1e-08,
):
    assert np.allclose(
        np.nan_to_num(cell_tensor), np.nan_to_num(test_tensor), rtol=rtol, atol=atol
    ), (
        " the results from notebook "
        "and runtime "
        f"do not match\n {original_cell_output}!={original_test_output} \n\n"
    )


def benchmarking_helper(exec_fn, exec_comp):
    pattern = r"([\d.]+) ([µmns]+)"
    match_1 = re.search(pattern, exec_fn)
    match_2 = re.search(pattern, exec_comp)
    print(f"match_1 {exec_fn} {match_1}")
    print(f"match_2 {exec_comp} {match_2}")

    if match_1 and match_2:
        execution_time_1, time_unit_1 = float(match_1.group(1)), match_1.group(2)
        execution_time_2, time_unit_2 = float(match_2.group(1)), match_2.group(2)

        time_unit_mapping = {
            "µs": "microseconds",
            "ms": "milliseconds",
            "ns": "nanoseconds",
        }

        if time_unit_1 in time_unit_mapping and time_unit_2 in time_unit_mapping:
            time_delta_1 = timedelta(
                **{time_unit_mapping[time_unit_1]: execution_time_1}
            )
            time_delta_2 = timedelta(
                **{time_unit_mapping[time_unit_2]: execution_time_2}
            )

            speedup = abs(time_delta_1 / time_delta_2)
            return speedup

    return None


def value_test(
    *,
    test_obj,
    test_output=None,
    test_execution_count=None,
    cell_output=None,
    cell_execution_count=None,
    config=None,
    next_test_output=None,
    next_cell_output=None,
):
    test_obj.assertEqual(
        cell_execution_count, test_execution_count, "Asynchronous execution failed!"
    )

    benchmark_threshold = None
    if re.search(r"i\d$", config):
        test_output = "\n".join(test_output.split("\n")[1:])
        cell_output = "\n".join(cell_output.split("\n")[1:])
        config = config[:-3]
        print(f"final config {config}")
    elif re.search(r"x\d$", config):
        benchmark_threshold = int(config[-1])
        config = config[:-3]
        print(f"final config {config}")

    if config == "value":
        test_obj.assertEqual(cell_output, test_output, f"Values don't match")

    elif config == "array_values":
        pattern_for_numbers = r"(\d*)(\.?)(\d+)"
        test_numbers = [
            "".join(occurrence)
            for occurrence in re.findall(pattern_for_numbers, test_output)
        ]
        cell_numbers = [
            "".join(occurrence)
            for occurrence in re.findall(pattern_for_numbers, cell_output)
        ]
        test_tensor = eval("[" + ", ".join(test_numbers) + "]")
        cell_tensor = eval("[" + ", ".join(cell_numbers) + "]")
        test_obj.assertEqual(
            len(test_tensor),
            len(cell_tensor),
            f"Length of outputs doesn't match {cell_output} != {test_output}",
        )
        print(f"test_tensor {test_tensor}")
        print(f"cell_tensor {cell_tensor}")
        assert_all_close(
            test_tensor=test_tensor,
            cell_tensor=cell_tensor,
            original_test_output=test_output,
            original_cell_output=cell_output,
        )

    elif config == "benchmark":
        speedup_cell = benchmarking_helper(cell_output, next_cell_output)
        speedup_test = benchmarking_helper(test_output, next_test_output)
        print(f"speedup_cell {speedup_cell}")
        print(f"speedup_test {speedup_test}")
        if benchmark_threshold:
            test_obj.assertLessEqual(benchmark_threshold, speedup_test)
        else:
            test_obj.assertLessEqual(speedup_cell, speedup_test)
