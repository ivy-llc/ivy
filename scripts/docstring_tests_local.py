"""The script for running the docstring tests locally and easily."""
# global
import argparse

# local
import ivy
from ivy_tests.test_docstrings import check_docstring_examples_run


# parse all arguments
def _parse_testing_arguments(*, backend: str = "all", fn: str = "abs"):
    """
    Parse the arguments for the docstring tests.

    Parameters
    ----------
    backend
        The backend to use for testing. Any supported backends
        Default is "all".
    fn
        The function name to test.

    Returns
    -------
    ret
        A tuple of the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run docstring tests.")
    parser.add_argument(
        "--backend", default=backend, help="The backend to use for testing."
    )
    parser.add_argument("--fn", default=fn, help="The function name to test.")
    args, unknown = parser.parse_known_args()
    if args.backend == "all":
        args.backend = ["jax", "tensorflow", "numpy", "torch", "paddle"]
    else:
        args.backend = [args.backend]
    return args.backend, args.fn


def _test_docstring():
    backends, function_name = _parse_testing_arguments()
    for backend in backends:
        print(
            "Running the docstring tests for the function {} with the backend {}."
            .format(function_name, backend)
        )
        ivy.set_backend(backend)
        imported_function = ivy.__dict__[function_name]
        # assertion for the docstring test
        assert check_docstring_examples_run(fn=imported_function) is True, (
            "The docstring check for the function {} failed the test for the"
            " backend {}.".format(function_name, backend)
        )
        ivy.unset_backend()
    print(
        "The docstring check for the function {} passed the test for the backends {}."
        .format(function_name, backends)
    )


if __name__ == "__main__":
    _test_docstring()
