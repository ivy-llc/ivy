# global
from typing import Sequence
from packaging.version import parse
import sys

# local
from ivy.transpiler.exceptions import exceptions


def _check_framework_installed(
    framework: str,
    exception: exceptions.SourceToSourceTranslatorException,
):
    """
    Checks that a framework is installed and usable locally, throws a invalid source/target exception if it's not.
    """

    try:
        exec(f"import {framework}; {framework}.__version__")
    except:
        raise exception(
            f"Unable to find '{framework}' installed locally. Ensure it has been installed with `pip install {framework}` or similar.",
            propagate=True,
        )

    if framework == "jax":
        try:
            import flax
            assert parse(flax.__version__) >= parse("0.8.0")
        except (ImportError, AssertionError):
            raise exception(
                f"Transpiling to JAX requires the Flax package (version >= '0.8.0').",
                propagate=True,
            )


def assert_valid_source(source: str, valid_sources: Sequence[str]):
    if source not in valid_sources:
        raise exceptions.InvalidSourceException(
            f"'source' must be one of {valid_sources}.",
            propagate=True,
        )
    if source in ["jax", "numpy", "tensorflow", "torch"]:
        _check_framework_installed(source, exceptions.InvalidSourceException)


def assert_valid_target(target: str, valid_targets: Sequence[str]):
    if target not in valid_targets:
        raise exceptions.InvalidTargetException(
            f"'target' must be one of {valid_targets}.",
            propagate=True,
        )
    if target in ["jax", "numpy", "tensorflow", "torch"]:
        _check_framework_installed(target, exceptions.InvalidTargetException)
