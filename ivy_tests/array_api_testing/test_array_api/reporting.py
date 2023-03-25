from array_api_tests.dtype_helpers import dtype_to_name
from array_api_tests import _array_module as xp
from array_api_tests import __version__

from collections import Counter
from types import BuiltinFunctionType, FunctionType
import dataclasses
import json
import warnings

from hypothesis.strategies import SearchStrategy

from pytest import mark, fixture

try:
    import pytest_jsonreport  # noqa
except ImportError:
    raise ImportError("pytest-json-report is required to run the array API tests")


def to_json_serializable(o):
    if o in dtype_to_name:
        return dtype_to_name[o]
    if isinstance(o, (BuiltinFunctionType, FunctionType, type)):
        return o.__name__
    if dataclasses.is_dataclass(o):
        return to_json_serializable(dataclasses.asdict(o))
    if isinstance(o, SearchStrategy):
        return repr(o)
    if isinstance(o, dict):
        return {to_json_serializable(k): to_json_serializable(v) for k, v in o.items()}
    if isinstance(o, tuple):
        if hasattr(o, "_asdict"):  # namedtuple
            return to_json_serializable(o._asdict())
        return tuple(to_json_serializable(i) for i in o)
    if isinstance(o, list):
        return [to_json_serializable(i) for i in o]

    # Ensure everything is JSON serializable. If this warning is issued, it
    # means the given type needs to be added above if possible.
    try:
        json.dumps(o)
    except TypeError:
        warnings.warn(
            f"{o!r} (of type {type(o)}) is not JSON-serializable. Using the repr instead."
        )
        return repr(o)

    return o


@mark.optionalhook
def pytest_metadata(metadata):
    """
    Additional global metadata for --json-report.
    """
    metadata["array_api_tests_module"] = xp.mod_name
    metadata["array_api_tests_version"] = __version__


@fixture(autouse=True)
def add_extra_json_metadata(request, json_metadata):
    """
    Additional per-test metadata for --json-report
    """

    def add_metadata(name, obj):
        obj = to_json_serializable(obj)
        json_metadata[name] = obj

    test_module = request.module.__name__
    if test_module.startswith("array_api_tests.meta"):
        return

    test_function = request.function.__name__
    assert test_function.startswith("test_"), "unexpected test function name"

    if test_module == "array_api_tests.test_has_names":
        array_api_function_name = None
    else:
        array_api_function_name = test_function[len("test_") :]

    add_metadata("test_module", test_module)
    add_metadata("test_function", test_function)
    add_metadata("array_api_function_name", array_api_function_name)

    if hasattr(request.node, "callspec"):
        params = request.node.callspec.params
        add_metadata("params", params)

    def finalizer():
        # TODO: This metadata is all in the form of error strings. It might be
        # nice to extract the hypothesis failing inputs directly somehow.
        if hasattr(request.node, "hypothesis_report_information"):
            add_metadata(
                "hypothesis_report_information",
                request.node.hypothesis_report_information,
            )
        if hasattr(request.node, "hypothesis_statistics"):
            add_metadata("hypothesis_statistics", request.node.hypothesis_statistics)

    request.addfinalizer(finalizer)


@mark.optionalhook
def pytest_json_modifyreport(json_report):
    # Deduplicate warnings. These duplicate warnings can cause the file size
    # to become huge. For instance, a warning from np.bool which is emitted
    # every time hypothesis runs (over a million times) causes the warnings
    # JSON for a plain numpy namespace run to be over 500MB.

    # This will lose information about what order the warnings were issued in,
    # but that isn't particularly helpful anyway since the warning metadata
    # doesn't store a full stack of where it was issued from. The resulting
    # warnings will be in order of the first time each warning is issued since
    # collections.Counter is ordered just like dict().
    counted_warnings = Counter([frozenset(i.items()) for i in json_report["warnings"]])
    deduped_warnings = [
        {**dict(i), "count": counted_warnings[i]} for i in counted_warnings
    ]

    json_report["warnings"] = deduped_warnings
