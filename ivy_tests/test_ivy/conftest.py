# global
import os
import pytest
from typing import Dict, Union, Tuple
from hypothesis import settings

# local
from ivy import clear_backend_stack, DefaultDevice
from ivy_tests.test_ivy import helpers


MAX_EXAMPLES: int

FW_STRS = ["numpy", "jax", "tensorflow", "torch"]

TEST_BACKENDS: Dict[str, callable] = {
    "numpy": lambda: helpers.get_ivy_numpy(),
    "jax": lambda: helpers.get_ivy_jax(),
    "tensorflow": lambda: helpers.get_ivy_tensorflow(),
    "torch": lambda: helpers.get_ivy_torch(),
    "": lambda: None,
}

CONFIG_DICT: Dict[str, Union[Tuple[bool, bool], None, bool]] = {
    "as_variable": None,
    "native_array": None,
    "with_out": None,
    "container": None,
    "instance_method": None,
}

TEST_PARAMS_CONFIG = []

if "ARRAY_API_TESTS_MODULE" not in os.environ:
    os.environ["ARRAY_API_TESTS_MODULE"] = "ivy.functional.backends.numpy"


def pytest_configure(config):
    num_examples = config.getoption("--num-examples")
    deadline = config.getoption("--deadline")
    profile_settings = {}
    if num_examples:
        profile_settings["max_examples"] = num_examples
    if deadline:
        profile_settings[deadline] = deadline

    settings.register_profile("test-profile", **profile_settings, print_blob=True)
    settings.load_profile("test-profile")

    # device
    raw_value = config.getoption("--device")
    if raw_value == "all":
        devices = ["cpu", "gpu:0", "tpu:0"]
    else:
        devices = raw_value.split(",")

    # framework
    raw_value = config.getoption("--backend")
    if raw_value == "all":
        backend_strs = FW_STRS
    else:
        backend_strs = raw_value.split(",")

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
            for compile_graph in compile_modes:
                for implicit in implicit_modes:
                    TEST_PARAMS_CONFIG.append(
                        (
                            device,
                            TEST_BACKENDS[backend_str](),
                            compile_graph,
                            implicit,
                            backend_str,
                        )
                    )


@pytest.fixture(autouse=True)
def run_around_tests(device, f, compile_graph, fw, implicit):
    clear_backend_stack()
    if f is not None:
        with f.use:
            with DefaultDevice(device):
                yield
    else:
        with DefaultDevice(device):
            yield


def pytest_generate_tests(metafunc):
    metafunc.parametrize("device,f,compile_graph,implicit,fw", TEST_PARAMS_CONFIG)


@pytest.fixture(scope="session")
def fixt_frontend_str():  # ToDo, temporary till handle test decorator is updated.
    return None


@pytest.fixture(scope="session")
def get_command_line_flags(request) -> Dict[str, bool]:
    getopt = request.config.getoption
    no_extra_testing = getopt("--no-extra-testing")

    CONFIG_DICT["as_variable"] = (
        getopt("--skip-variable-testing"),
        getopt("--with-variable-testing"),
    )
    CONFIG_DICT["native_array"] = (
        getopt("--skip-native-array-testing"),
        getopt("--with-native-array-testing"),
    )
    CONFIG_DICT["with_out"] = (
        getopt("--skip-out-testing"),
        getopt("--with-out-testing"),
    )
    CONFIG_DICT["container"] = (
        getopt("--skip-nestable-testing"),
        getopt("--with-nestable-testing"),
    )
    CONFIG_DICT["instance_method"] = (
        getopt("--skip-instance-method-testing"),
        getopt("--with-instance-method-testing"),
    )
    CONFIG_DICT["test_gradients"] = (
        getopt("--skip-gradient-testing"),
        getopt("--with-gradient-testing"),
    )

    # final mapping for hypothesis value generation
    for k, v in CONFIG_DICT.items():
        # when both flags are true
        if v[0] and v[1]:
            raise Exception(
                f"--skip-{k}--testing and --with-{k}--testing flags cannot be used \
                    together"
            )
        if v[1] and no_extra_testing:
            raise Exception(
                f"--with-{k}--testing and --no-extra-testing flags cannot be used \
                    together"
            )
        # skipping a test
        if v[0] or no_extra_testing:
            CONFIG_DICT[k] = False
        # extra testing
        if v[1]:
            CONFIG_DICT[k] = True
        # default
        if not v[0] ^ v[1]:
            CONFIG_DICT[k] = None

    return CONFIG_DICT


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("--backend", action="store", default="")
    parser.addoption("--compile_graph", action="store_true")
    parser.addoption("--with_implicit", action="store_true")

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
        "--num-examples",
        action="store",
        default=5,
        type=int,
        help="set max examples generated by Hypothesis",
    )
    parser.addoption(
        "--deadline",
        action="store",
        default=500000,
        type=int,
        help="set deadline for testing one example",
    )
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
