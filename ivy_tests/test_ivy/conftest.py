# global
import os
import pytest
from typing import Dict
import sys
import multiprocessing as mp

# for enabling numpy's bfloat16 behavior
from packaging import version
from .helpers.globals import mod_backend, mod_frontend

multiprocessing_flag = False  # multiversion


# local
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy import set_exception_trace_mode
from ivy_tests.test_ivy.helpers import globals as test_globals
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks  # noqa
from ivy_tests.test_ivy.helpers.multiprocessing import backend_proc, frontend_proc
from ivy_tests.test_ivy.helpers.pipeline_helper import (
    BackendHandler,
    BackendHandlerMode,
)

GENERAL_CONFIG_DICT = {}
UNSET_TEST_CONFIG = {"list": [], "flag": []}
UNSET_TEST_API_CONFIG = {"list": [], "flag": []}

TEST_PARAMS_CONFIG = []
SKIP_GROUND_TRUTH = False
UNSUPPORTED_FRAEMWORK_DEVICES = {"numpy": ["gpu", "tpu"]}
if "ARRAY_API_TESTS_MODULE" not in os.environ:
    os.environ["ARRAY_API_TESTS_MODULE"] = "ivy.functional.backends.numpy"


def default_framework_mapper(fw, fw_path="/opt/fw/", set_too=False):
    # do a path search, get the latest
    # so that we can get the highest version
    # available dynamically and set that for
    # use by the rest of the code
    # eg: torch/1.11.0 and torch/1.12.0
    # this will map to torch/1.12.0
    try:
        versions = os.listdir(f"/opt/fw/{fw}")
    except FileNotFoundError:
        # if no version exists return None
        return None
    versions = [version.parse(v) for v in versions]
    versions.sort()
    if set_too:
        sys.path.insert(1, f"{fw_path}{fw}/{str(versions[-1])}")
    return str(versions[-1])


def pytest_report_header(config):
    return [
        f"backend(s): {config.getoption('backend')}",
        f"device: {config.getoption('device')}",
        f"number of Hypothesis examples: {config.getoption('num_examples')}",
    ]


def pytest_configure(config):
    global available_frameworks
    global multiprocessing_flag
    if config.getoption("--set-backend"):
        BackendHandler._update_context(BackendHandlerMode.SetBackend)

    # Ivy Exception traceback
    set_exception_trace_mode(config.getoption("--ivy-tb"))

    # Pytest traceback
    config.option.tbstyle = config.getoption("--tb")

    # device
    raw_value = config.getoption("--device")
    if raw_value == "all":
        devices = ["cpu", "gpu:0", "tpu:0"]
    else:
        devices = raw_value.split(",")

    # framework
    raw_value = config.getoption("--backend")
    if raw_value == "all":
        backend_strs = available_frameworks
    else:
        backend_strs = raw_value.split(",")
    no_mp = config.getoption("--no-mp")

    if not no_mp:
        # we go multiprocessing, if  multiversion
        known_backends = {"tensorflow", "torch", "jax"}
        found_backends = set()
        for fw in backend_strs:
            if "/" in fw:
                # set backend to be used
                BackendHandler._update_context(BackendHandlerMode.SetBackend)
                multiprocessing_flag = True
                # spin up multiprocessing
                # build mp process, queue, initiation etc
                input_queue = mp.Queue()
                output_queue = mp.Queue()
                proc = mp.Process(target=backend_proc, args=(input_queue, output_queue))
                # start the process so that it loads the framework
                input_queue.put(fw)
                proc.start()

                # we have the process running, the framework imported within,
                # we now pack the queue and the process and store it in dict
                # for future access
                fwrk, ver = fw.split("/")
                mod_backend[fwrk] = (proc, input_queue, output_queue)
                # set the latest version for the rest of the test code and move on
                default_framework_mapper(fwrk, set_too=False)
                found_backends.add(fwrk)

        if found_backends:
            # we know it's multiversion+multiprocessing
            # spin up processes for other backends that
            # were not found in --backend flag
            left_frameworks = known_backends.difference(found_backends)
            for fw in left_frameworks:
                # spin up multiprocessing
                # build mp process, queue, initiation etc
                # find the latest version of this framework
                # and set it in the path for rest of the code
                # to access
                version = default_framework_mapper(fw, set_too=False)
                # spin up process only if a version was found else don't
                if version:
                    input_queue = mp.Queue()
                    proc = mp.Process(
                        target=backend_proc, args=(input_queue, output_queue)
                    )
                    # start the process so that it loads the framework
                    input_queue.put(f"{fw}/{version}")
                    proc.start()

                # we have the process running, the framework imported within,
                # we now pack the queue and the process and store it in dict
                # for future access
                mod_backend[fw] = (proc, input_queue, output_queue)

    else:
        # no multiprocessing if multiversion
        for fw in backend_strs:
            if "/" in fw:
                multiprocessing_flag = True
                # multiversion, but without multiprocessing
                sys.path.insert(1, f"/opt/fw/{fw}")

    # frontend
    frontend = config.getoption("--frontend")
    if frontend:
        frontend_strs = frontend.split(",")
        # if we are passing a frontend flag, it has to have a version with it
        for frontend in frontend_strs:
            # spin up multiprocessing
            fw, ver = frontend.split("/")
            # build mp process, queue, initiation etc
            queue = mp.Queue()
            proc = mp.Process(target=frontend_proc, args=(queue,))
            # start the process so that it loads the framework
            proc.start()
            queue.put(f"{fw}/{ver}")
            # we have the process running, the framework imported within,
            # we now pack the queue and the process and store it in dict
            # for future access
            mod_frontend[fw] = (proc, queue)

    # trace_graph
    raw_value = config.getoption("--trace_graph")
    if raw_value == "both":
        trace_modes = [True, False]
    elif raw_value == "true":
        trace_modes = [True]
    else:
        trace_modes = [False]

    # implicit
    raw_value = config.getoption("--with_implicit")
    if raw_value == "true":
        implicit_modes = [True, False]
    else:
        implicit_modes = [False]

    # create test configs
    for backend_str in backend_strs:
        for device in devices:
            if "/" in backend_str:
                backend_str = backend_str.split("/")[0]
            if (
                backend_str in UNSUPPORTED_FRAEMWORK_DEVICES
                and device.partition(":")[0]
                in UNSUPPORTED_FRAEMWORK_DEVICES[backend_str]
            ):
                continue
            for trace_graph in trace_modes:
                for implicit in implicit_modes:
                    TEST_PARAMS_CONFIG.append(
                        (
                            device,
                            backend_str,
                            trace_graph,
                            implicit,
                        )
                    )

    process_cl_flags(config)


@pytest.fixture(autouse=True)
def run_around_tests(request, on_device, backend_fw, trace_graph, implicit):
    try:
        test_globals.setup_api_test(
            backend_fw,
            (
                request.function.ground_truth_backend
                if hasattr(request.function, "ground_truth_backend")
                else None
            ),
            on_device,
            (
                request.function.test_data
                if hasattr(request.function, "test_data")
                else None
            ),
        )

    except Exception as e:
        test_globals.teardown_api_test()
        raise RuntimeError(f"Setting up test for {request.function} failed.") from e

    yield
    test_globals.teardown_api_test()


def pytest_generate_tests(metafunc):
    # Skip backend test against groud truth backend
    # This redundant and wastes resources, as we going to be comparing
    # The backend against it self
    global SKIP_GROUND_TRUTH
    if hasattr(metafunc.function, "ground_truth_backend"):
        test_paramters = TEST_PARAMS_CONFIG.copy()
        # Find the entries that contains the ground truth backend as it's backend
        for entry in test_paramters.copy():
            # Entry 1 is backend_fw
            if entry[1] == metafunc.function.ground_truth_backend and SKIP_GROUND_TRUTH:
                test_paramters.remove(entry)
        metafunc.parametrize(
            "on_device,backend_fw,trace_graph,implicit", test_paramters
        )
    else:
        metafunc.parametrize(
            "on_device,backend_fw,trace_graph,implicit", TEST_PARAMS_CONFIG
        )


def process_cl_flags(config) -> Dict[str, bool]:
    getopt = config.getoption
    no_extra_testing = getopt("--no-extra-testing")

    tmp_config = {
        "as_variable": (
            getopt("--skip-variable-testing"),
            getopt("--with-variable-testing"),
        ),
        "native_array": (
            getopt("--skip-native-array-testing"),
            getopt("--with-native-array-testing"),
        ),
        "with_out": (
            getopt("--skip-out-testing"),
            getopt("--with-out-testing"),
        ),
        "container": (
            getopt("--skip-nestable-testing"),
            getopt("--with-nestable-testing"),
        ),
        "instance_method": (
            getopt("--skip-instance-method-testing"),
            getopt("--with-instance-method-testing"),
        ),
        "test_gradients": (
            getopt("--skip-gradient-testing"),
            getopt("--with-gradient-testing"),
        ),
        "test_trace": (
            getopt("--skip-trace-testing"),
            getopt("--with-trace-testing"),
        ),
        "test_trace_each": (
            getopt("--skip-trace-testing-each"),
            getopt("--with-trace-testing-each"),
        ),
        "transpile": (
            False,
            getopt("--with-transpile"),
        ),
        "test_cython_wrapper": (
            getopt("--skip-cython-wrapper-testing"),
            getopt("--with-cython-wrapper-testing"),
        ),
    }

    # whether to skip gt testing or not
    # global SKIP_GROUND_TRUTH
    # SKIP_GROUND_TRUTH = not tmp_config["transpile"][1]

    # final mapping for hypothesis value generation
    for k, v in tmp_config.items():
        # when both flags are true
        if v[0] and v[1]:
            raise Exception(
                f"--skip-{k}--testing and --with-{k}--testing flags cannot be used "
                "together"
            )
        if v[1] and no_extra_testing:
            raise Exception(
                f"--with-{k}--testing and --no-extra-testing flags cannot be used "
                "together"
            )
        # skipping a test
        if v[0] or no_extra_testing:
            pf.build_flag(k, False)
        # extra testing
        if v[1]:
            pf.build_flag(k, True)


def pytest_addoption(parser):
    parser.addoption("--no-mp", action="store", default=None)
    parser.addoption(
        "--set-backend",
        action="store_true",
        default=False,
        help="Force the testing pipeline to use ivy.set_backend for backend setting",
    )

    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("-B", "--backend", action="store", default="all")
    parser.addoption("--trace_graph", action="store_true")
    parser.addoption("--with_implicit", action="store_true")
    parser.addoption("--frontend", action="store", default=None)
    parser.addoption("--env", action="store", default=None)
    parser.addoption("--ground_truth", action="store", default=None)
    parser.addoption("--skip-variable-testing", action="store_true")
    parser.addoption("--skip-native-array-testing", action="store_true")
    parser.addoption("--skip-out-testing", action="store_true")
    parser.addoption("--skip-nestable-testing", action="store_true")
    parser.addoption("--skip-instance-method-testing", action="store_true")
    parser.addoption("--skip-gradient-testing", action="store_true")
    parser.addoption("--skip-trace-testing", action="store_true")
    parser.addoption("--skip-trace-testing-each", action="store_true")

    parser.addoption("--with-variable-testing", action="store_true")
    parser.addoption("--with-native-array-testing", action="store_true")
    parser.addoption("--with-out-testing", action="store_true")
    parser.addoption("--with-nestable-testing", action="store_true")
    parser.addoption("--with-instance-method-testing", action="store_true")
    parser.addoption("--with-gradient-testing", action="store_true")
    parser.addoption("--with-trace-testing", action="store_true")
    parser.addoption("--with-trace-testing-each", action="store_true")
    parser.addoption("--with-transpile", action="store_true")
    parser.addoption("--no-extra-testing", action="store_true")
    parser.addoption(
        "--my_test_dump",
        action="store",
        default=None,
        help="Print test items in my custom format",
    )
    parser.addoption("--skip-cython-wrapper-testing", action="store_true")
    parser.addoption("--with-cython-wrapper-testing", action="store_true")


def pytest_collection_finish(session):
    # Make sure we're not accidentally accessing it during test
    global TEST_PARAMS_CONFIG
    del TEST_PARAMS_CONFIG
    if session.config.option.my_test_dump is not None:
        for item in session.items:
            item_path = os.path.relpath(item.path)
            print(f"{item_path}::{item.name}")

        for backend in mod_backend:
            proc = mod_backend[backend]
            proc.terminate()
        pytest.exit("Done!")
