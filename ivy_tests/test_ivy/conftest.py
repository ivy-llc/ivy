# global
import os
import sys
import multiprocessing as mp
import pytest
from typing import Dict
import importlib

mod_frontend = {
    "tensorflow": None,
    "numpy": None,
    "jax": None,
    "torch": None,
    "mindspore": None,
    "scipy": None,
    "paddle": None,
}  # multiversion
mod_backend = {
    "numpy": None,
    "jax": None,
    "tensorflow": None,
    "torch": None,
    "paddle": None,
    "mxnet": None,
}  # multiversion

ground_backend = None  # multiversion


def backend_proc(queue):
    # first argument is going to be the framework and its path

    framework = queue.get()
    path = "/opt/fw/" + framework
    sys.path.insert(1, path)
    framework = framework.split("/")[0]
    framework = importlib.import_module(framework)
    print(framework.__version__)
    while True:
        # subsequent arguments will be passed
        data = queue.get()
        if not data:
            break
        # process the data


def frontend_proc(queue):
    # first argument is going to be the framework and its path
    framework = queue.get()
    sys.path.insert(1, f"/opt/fw/{framework}")
    importlib.import_module(framework.split("/")[0])
    while True:
        # subsequent arguments will be passed
        data = queue.get()
        if not data:
            break
        # process the data


# local
import ivy_tests.test_ivy.helpers.test_parameter_flags as pf
from ivy import DefaultDevice
from ivy import set_exception_trace_mode
from ivy_tests.test_ivy.helpers import globals as test_globals
from ivy_tests.test_ivy.helpers.available_frameworks import available_frameworks

available_frameworks = available_frameworks()
GENERAL_CONFIG_DICT = {}
UNSET_TEST_CONFIG = {"list": [], "flag": []}
UNSET_TEST_API_CONFIG = {"list": [], "flag": []}

TEST_PARAMS_CONFIG = []
UNSUPPORTED_FRAEMWORK_DEVICES = {"numpy": ["gpu", "tpu"]}

if "ARRAY_API_TESTS_MODULE" not in os.environ:
    os.environ["ARRAY_API_TESTS_MODULE"] = "ivy.functional.backends.numpy"


def pytest_report_header(config):
    return [
        f"backend(s): {config.getoption('backend')}",
        f"device: {config.getoption('device')}",
        f"number of Hypothesis examples: {config.getoption('num_examples')}",
    ]


def default_framework_mapper(fw, set_too=False):
    # do a path search, get the latest one
    versions = os.listdir(f"/opt/fw/{fw}")
    versions.sort()
    if set_too:
        sys.path.insert(1, f"/opt/fw/{fw}/{versions[-1]}")
    return versions[-1]


def pytest_configure(config):
    global available_frameworks

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
        for fw in backend_strs:
            if "/" in fw:
                # spin up multiprocessing
                # build mp process, queue, initiation etc
                print(fw, "here")
                queue = mp.Queue()
                proc = mp.Process(target=backend_proc, args=(queue,))
                # start the process so that it loads the framework
                queue.put(fw)
                proc.start()

                # we have the process running, the framework imported within,
                # we now pack the queue and the process and store it in dict
                # for future access
                mod_backend[fw] = (proc, queue)
            else:
                # this is usual testing, just set the latest version and move on
                default_framework_mapper(fw, set_too=True)
    else:
        # no multiprocessing if multiversion
        for fw in backend_strs:
            if "/" in fw:
                # multiversion, but without multiprocessing
                sys.path.insert(1, f"/opt/fw/{fw}")
            else:
                # usual testing, set the latest version, move on
                default_framework_mapper(fw, set_too=True)

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
            queue.put(fw + "/" + ver)
            # we have the process running, the framework imported within,
            # we now pack the queue and the process and store it in dict
            # for future access
            mod_frontend[fw] = (proc, queue)

    # compile_graph
    raw_value = config.getoption("--compile_graph")
    if raw_value == "both":
        compile_modes = [True, False]
    elif raw_value == "true":
        compile_modes = [True]
    else:
        compile_modes = [False]

    # implicit
    raw_value = config.getoption("--with_implicit")
    if raw_value == "true":
        implicit_modes = [True, False]
    else:
        implicit_modes = [False]

    # create test configs
    for backend_str in backend_strs:
        for device in devices:
            if (
                backend_str in UNSUPPORTED_FRAEMWORK_DEVICES.keys()
                and device.partition(":")[0]
                in UNSUPPORTED_FRAEMWORK_DEVICES[backend_str]
            ):
                continue
            for compile_graph in compile_modes:
                for implicit in implicit_modes:
                    if "/" in backend_str:
                        TEST_PARAMS_CONFIG.append(
                            (
                                device,
                                test_globals.FWS_DICT[backend_str.split("/")[0]](),
                                compile_graph,
                                implicit,
                            )
                        )
                    else:
                        TEST_PARAMS_CONFIG.append(
                            (
                                device,
                                test_globals.FWS_DICT[backend_str](),
                                compile_graph,
                                implicit,
                            )
                        )

    process_cl_flags(config)


@pytest.fixture(autouse=True)
def run_around_tests(request, on_device, backend_fw, compile_graph, implicit):
    ivy_test = hasattr(request.function, "_ivy_test")
    if ivy_test:
        try:
            if ground_backend:
                test_globals.setup_api_test(
                    backend_fw.backend,
                    ground_backend,
                    on_device,
                    (
                        request.function.test_data
                        if hasattr(request.function, "test_data")
                        else None
                    ),
                )
            else:
                test_globals.setup_api_test(
                    backend_fw.backend,
                    request.function.ground_truth_backend,
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
    with backend_fw.use:
        with DefaultDevice(on_device):
            yield
    if ivy_test:
        test_globals.teardown_api_test()


def pytest_generate_tests(metafunc):
    # Skip backend test against groud truth backend
    # This redundant and wastes resources, as we going to be comparing
    # The backend against it self
    if hasattr(metafunc.function, "ground_truth_backend"):
        test_paramters = TEST_PARAMS_CONFIG.copy()
        # Find the entries that contains the ground truth backend as it's backend
        for entry in test_paramters.copy():
            if entry[1].backend == metafunc.function.ground_truth_backend:
                test_paramters.remove(entry)
        metafunc.parametrize(
            "on_device,backend_fw,compile_graph,implicit", test_paramters
        )
    else:
        metafunc.parametrize(
            "on_device,backend_fw,compile_graph,implicit", TEST_PARAMS_CONFIG
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
    }

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
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("-B", "--backend", action="store", default="all")
    parser.addoption("--compile_graph", action="store_true")
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

    parser.addoption("--with-variable-testing", action="store_true")
    parser.addoption("--with-native-array-testing", action="store_true")
    parser.addoption("--with-out-testing", action="store_true")
    parser.addoption("--with-nestable-testing", action="store_true")
    parser.addoption("--with-instance-method-testing", action="store_true")
    parser.addoption("--with-gradient-testing", action="store_true")
    parser.addoption("--no-extra-testing", action="store_true")
    parser.addoption(
        "--my_test_dump",
        action="store",
        default=None,
        help="Print test items in my custom format",
    )


def pytest_collection_finish(session):
    # Make sure we're not accidentally accessing it during test
    global TEST_PARAMS_CONFIG
    del TEST_PARAMS_CONFIG
    if session.config.option.my_test_dump is not None:
        for item in session.items:
            item_path = os.path.relpath(item.path)
            print("{}::{}".format(item_path, item.name))
        pytest.exit("Done!")
