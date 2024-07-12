# Assert All Dependencies are Importable and Correctly Versioned #
# ---------------------------------------------------------------#

import os
import argparse
import termcolor
import importlib
import faulthandler
from packaging import version

faulthandler.enable()
ERROR = False
ERROR_MSG = "\n"
WARN = False
WARN_MSG = "\n"
PRINT_MSG = "\n"


def parse(str_in):
    str_in = str_in.replace("\n", "")
    import_ops = ["==", "<", "<=", ">", ">="]
    if "mod_name=" in str_in:
        mod_name = str_in.split("mod_name=")[-1].split(" ")[0].split(",")[0]
    else:
        mod_name = str_in.split("=")[0].split(" ")[0]
    expected_version, expected_op = None, None
    for import_op in import_ops:
        if import_op in str_in:
            lib_name, expected_version = str_in.split(import_op)
            if "mod_name=" not in str_in:
                mod_name = lib_name
            expected_version = expected_version.split(" ")[0].split(",")[0]
            expected_op = import_op
    return mod_name, expected_version, expected_op


def compare(version1, version2, operator):
    version1 = version.parse(version1)
    version2 = version.parse(version2)
    if operator == "==":
        return version1 == version2
    elif "<" in operator:
        if operator == "<=":
            return version1 <= version2
        return version1 < version2
    else:
        if operator == ">=":
            return version1 >= version2
        return version1 > version2


def test_imports(fname, assert_version, update_versions):
    global ERROR
    global ERROR_MSG
    global WARN
    global WARN_MSG
    global PRINT_MSG
    versions_to_update = {}
    msg = f"\nasserting imports work for: {fname}\n\n"
    PRINT_MSG += msg
    ERROR_MSG += msg
    WARN_MSG += msg
    with open(fname, "r") as f:
        file_lines = f.readlines()
    mod_names_n_versions = [parse(req) for req in file_lines]
    for line_num, (mod_name, expected_version, expected_op) in enumerate(
        mod_names_n_versions
    ):
        # noinspection PyBroadException
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            ERROR = True
            msg = f"{mod_name} could not be imported: {e}\n"
            ERROR_MSG += msg
            PRINT_MSG += msg
            continue
        # noinspection PyBroadException
        try:
            # noinspection PyUnresolvedReferences
            detected_version = mod.__version__
        except AttributeError:
            try:
                detected_version = ".".join([str(n) for n in mod.VERSION])
            except AttributeError:
                continue
        except Exception:
            detected_version = None
        if detected_version and expected_version:
            if compare(detected_version, expected_version, expected_op):
                msg = f"{mod_name} detected correct version: {detected_version}\n"
            else:
                msg = (
                    f"expected version {expected_version} for module {mod_name}, but"
                    f" detected version {detected_version}\n"
                )
                versions_to_update[line_num] = {
                    "expected": expected_version,
                    "detected": detected_version,
                }
                if assert_version:
                    ERROR = True
                    ERROR_MSG += msg
                else:
                    WARN = True
                    WARN_MSG += msg
            PRINT_MSG += msg
        else:
            if detected_version:
                msg = (
                    f"{mod_name} detected version: {detected_version}, but no expected"
                    " version provided\n"
                )
            elif expected_version:
                msg = (
                    f"{mod_name} expected version: {expected_version}, but unable to"
                    " detect version\n"
                )
            else:
                msg = (
                    "no expected version provided, and unable to detect version for"
                    f" {mod_name}\n"
                )
            WARN = True
            PRINT_MSG += msg
            WARN_MSG += msg
    if not update_versions:
        return
    for line_num, versions in versions_to_update.items():
        orig_str = file_lines[line_num]
        new_str = orig_str.replace(versions["expected"], versions["detected"])
        file_lines[line_num] = new_str
    with open(fname, "w") as f:
        f.writelines(file_lines)


def main(filepaths, assert_matching_versions, update_versions):
    for filepath in filepaths.replace(" ", "").split(","):
        assert os.path.isfile(filepath)
        test_imports(
            filepath,
            assert_version=assert_matching_versions,
            update_versions=update_versions,
        )
    print(PRINT_MSG)
    if WARN:
        print(termcolor.colored("WARNING\n" + WARN_MSG, "red"))
    if ERROR:
        raise Exception(ERROR_MSG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fp",
        "--filepaths",
        type=str,
        required=True,
        help=(
            "Comma separated filepaths of all text files to check. Spaces are ignored."
        ),
    )
    parser.add_argument(
        "-amv",
        "--assert_matching_versions",
        action="store_true",
        help=(
            "Whether to assert that all module versions match those lists in the "
            "requirements.txt and optional.txt files."
        ),
    )
    parser.add_argument(
        "-uv",
        "--update_versions",
        action="store_true",
        help="Whether to update the versions in the installation files.",
    )
    parsed_args = parser.parse_args()
    main(
        parsed_args.filepaths,
        parsed_args.assert_matching_versions,
        parsed_args.update_versions,
    )
