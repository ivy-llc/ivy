from typing import List
import operator
import requests
import sys

PYPI_URL = "https://pypi.python.org/pypi/"

# Define comparison operators
COMPARISON_OPS = {
    "~=": lambda x, y: x[: len(y)] == y,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

# Define a list of comparison operators that represent "lower than"
LOWER_OPS = ["<=", "<"]


def compare_versions(x, y, op, op1):
    """
    Compares two version numbers using the given comparison operators.
    Returns the higher or lower version number, depending on which operator was used.
    """
    if len(x) != len(y):
        x, y = make_versions_equal_length(x, y)

    x = tuple(map(convert_to_int_if_possible, x))
    y = tuple(map(convert_to_int_if_possible, y))

    if COMPARISON_OPS[op](x, y):
        if op in LOWER_OPS:
            return min(x, y)
        else:
            return max(x, y)

    if COMPARISON_OPS[op1](y, x):
        if op1 in LOWER_OPS:
            return min(y, x)
        else:
            return max(y, x)

    return None


def make_versions_equal_length(x, y):
    """
    Make the given version numbers equal in length by padding with zeroes.
    """
    if len(x) < len(y):
        x = list(x) + ["0"] * (len(y) - len(x))
    else:
        y = list(y) + ["0"] * (len(x) - len(y))
    return x, y


def convert_to_int_if_possible(value):
    """
    Convert a string to an int if possible, otherwise return the original string.
    """
    try:
        return int(value)
    except ValueError:
        return value


def parse_version_constraints(version_constraints):
    """
    Parse the version constraints string for a package into a dictionary of operator-version tuples.
    """
    version_constraints = (
        version_constraints.replace("(", "")
        .replace(")", "")
        .replace(" ", "")
        .split(",")
    )
    constraints_dict = {}
    for constraint in version_constraints:
        op, version = split_constraint(constraint)
        constraints_dict[op] = tuple(version.split("."))
    return constraints_dict


def split_constraint(constraint):
    """
    Split a version constraint string into the comparison operator and version number.
    """
    for i in range(len(constraint)):
        if constraint[i] not in COMPARISON_OPS:
            continue
        return constraint[:i], constraint[i + 1 :]
    raise ValueError(f"Invalid version constraint: {constraint}")


def get_package_version_constraints(package_name):
    """
    Retrieve the version constraints for the given package from PyPI.
    """
    package_info = requests.get(PYPI_URL + package_name + "/json").json()
    requires_python = ["python " + str(package_info["info"]["requires_python"])]
    requires_dist = package_info["info"]["requires_dist"] or []
    return parse_version_constraints(requires_python + requires_dist)


def resolve_version_conflicts(pkg1, pkg2):
    """
    Given two package names, resolve any version conflicts between them.
    Returns a dictionary of package name-version tuples representing the resolved versions.
    """
    pkg1_constraints = get_package_version_constraints(pkg1)
    pkg2_constraints = get_package_version_constraints
