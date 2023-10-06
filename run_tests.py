# Run Tests
import os
import sys
from pymongo import MongoClient
import requests
import json
import old_run_test_helpers as old_helpers
from run_tests_CLI.get_all_tests import BACKENDS


def get_latest_package_version(package_name):
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        response.raise_for_status()
        package_info = response.json()
        return package_info["info"]["version"]
    except requests.exceptions.RequestException:
        print(f"Error: Failed to fetch package information for {package_name}.")
        return None


def get_submodule_and_function_name(test_path, is_frontend_test=False):
    submodule_test = test_path.split("/")[-1]
    submodule, test_function = submodule_test.split("::")
    submodule = submodule.replace("test_", "").replace(".py", "")
    function_name = test_function[5:]
    if is_frontend_test:
        with open(test_path.split("::")[0]) as test_file:
            test_file_content = test_file.read()
            test_name = test_function.split(",")[0]
            test_function_idx = test_file_content.find(f"def {test_name}")
            function_name = test_file_content[
                test_file_content[:test_function_idx].rfind('fn_tree="') + 9 :
            ].split('"')[0]
    return submodule, function_name


if __name__ == "__main__":
    redis_url = sys.argv[1]
    redis_pass = sys.argv[2]
    mongo_key = sys.argv[3]
    version_flag = sys.argv[4]
    gpu_flag = sys.argv[5]
    workflow_id = sys.argv[6]
    priority_flag = sys.argv[7]

    if len(sys.argv) > 8 and sys.argv[8] != "null":
        run_id = sys.argv[8]
    else:
        run_id = f"https://github.com/unifyai/ivy/actions/runs/{workflow_id}"

    device = "cpu"
    if gpu_flag == "true":
        device = "gpu"

    status = True
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["ci_dashboard"]

    # old
    if priority_flag == "true":
        priority_flag = True
    else:
        priority_flag = False
    failed = False
    old_db = cluster["Ivy_tests_multi_gpu"]
    old_db_priority = cluster["Ivy_tests_priority"]

    # pull gpu image for gpu testing
    if device == "gpu":
        os.system("docker pull unifyai/multicuda:base_and_requirements")

    # read the tests to be run
    with open("tests_to_run", "r") as f:
        for line in f:
            print(f"\n{'*' * 100}")
            print(f"{line[:-1]}")
            print(f"{'*' * 100}\n")

            # get the test, submodule, backend and version
            test_path, backend = line.strip().split(",")
            is_frontend_test = "test_frontends" in test_path
            collection = db["frontend_tests"] if is_frontend_test else db["ivy_tests"]
            submodule, function_name = get_submodule_and_function_name(
                test_path, is_frontend_test
            )
            version = get_latest_package_version(backend).replace(".", "_")

            # old
            coll, submod, test_fn = old_helpers.get_submodule(test_path)
            backend_version = "latest-stable"

            # multi-version tests
            if version_flag == "true":
                backends = [backend.strip()]
                backend_name, backend_version = backend.split("/")
                other_backends = [
                    fw for fw in BACKENDS if (fw != backend_name and fw != "paddle")
                ]
                for other_backend in other_backends:
                    backends.append(
                        other_backend + "/" + get_latest_package_version(other_backend)
                    )
                print("Backends:", backends)
                command = (
                    f"docker run --rm --env REDIS_URL={redis_url} --env"
                    f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy/ivy'
                    ' unifyai/multiversion:latest /bin/bash -c "python'
                    f" multiversion_framework_directory.py {' '.join(backends)};cd"
                    f' ivy;pytest --tb=short {test_path} --backend={backend.strip()}"'
                )
                backend = backend.split("/")[0] + "\n"
                backend_version = backend_version.strip()
                print("Running", command)

            # gpu tests
            elif device == "gpu":
                command = (
                    f"docker run --rm --gpus all --env REDIS_URL={redis_url} --env"
                    f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                    ' "$(pwd)"/.hypothesis:/.hypothesis'
                    " unifyai/multicuda:base_and_requirements python3 -m pytest"
                    f" --tb=short {test_path} --device=gpu:0 -B={backend}"
                )

            # cpu tests
            else:
                command = (
                    f"docker run --rm --env REDIS_URL={redis_url} --env"
                    f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                    ' "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3'
                    f" -m pytest --tb=short {test_path} --backend {backend}"
                )

            # run the test
            sys.stdout.flush()
            status = not os.system(command)

            # old (populate the old database with results)
            if status:
                res = old_helpers.make_clickable(
                    run_id, old_helpers.result_config["success"]
                )
            else:
                res = old_helpers.make_clickable(
                    run_id, old_helpers.result_config["failure"]
                )
                failed = True
            frontend_version = None
            if coll[0] in ["numpy", "jax", "tensorflow", "torch", "paddle"]:
                frontend_version = "latest-stable"
            if priority_flag:
                print("Updating Priority DB")
                old_helpers.update_individual_test_results(
                    old_db_priority[coll[0]],
                    coll[1],
                    submod,
                    backend,
                    test_fn,
                    res,
                    "latest-stable",
                    frontend_version,
                    device,
                )
            else:
                print(backend_version)
                old_helpers.update_individual_test_results(
                    db[coll[0]],
                    coll[1],
                    submod,
                    backend,
                    test_fn,
                    res,
                    backend_version,
                    frontend_version,
                    device,
                )

            # run transpilation tests if the test passed
            if status:
                print(f"\n{'*' * 100}")
                print(f"{line[:-1]} --> transpilation tests")
                print(f"{'*' * 100}\n")
                os.system(f"{command} --num-examples 1 --with-transpile")

            # load data from report if generated
            report_path = os.path.join(
                __file__[: __file__.rfind(os.sep)], "report.json"
            )
            report_content = {}
            if os.path.exists(report_path):
                report_content = json.load(open(report_path))

            # create a prefix str for the update query for frontend tests
            # (with frontend version)
            test_info = dict()
            prefix_str = ""
            if is_frontend_test:
                frontend = test_path[test_path.find("test_frontends") :].split(os.sep)[
                    1
                ][5:]
                frontend_version = get_latest_package_version(frontend).replace(
                    ".", "_"
                )
                test_info["frontend"] = frontend
                if report_content:
                    test_info = {
                        **test_info,
                        "fw_time": report_content["fw_time"],
                        "ivy_nodes": report_content["ivy_nodes"],
                    }
                prefix_str = f"{frontend_version}."

            # initialize test information for ci_dashboard db
            # format of the last 2 keys
            #   <frontend_version>.<backend_name>.<backend_version>.<status>.<device>
            #   <backend_name>.<backend_version>.<status>.<device>
            # for frontend tests and ivy tests respectively
            test_info = {
                "_id": function_name,
                "test_path": test_path,
                "submodule": submodule,
                f"{prefix_str}{backend}.{version}.status.{device}": status,
                f"{prefix_str}{backend}.{version}.workflow.{device}": run_id,
            }

            # add transpilation metrics if report generated
            if status and report_content:
                transpilation_metrics = {
                    "nodes": report_content["nodes"][backend],
                    "time": report_content["time"][backend],
                    "args": report_content["args"][backend],
                    "kwargs": report_content["kwargs"][backend],
                }
                for metric, value in transpilation_metrics.items():
                    test_info[f"{prefix_str}{backend}.{version}.{metric}"] = value

            # populate the ci_dashboard db
            id = test_info.pop("_id")
            print(collection.update_one({"_id": id}, {"$set": test_info}, upsert=True))

    # if any tests fail, the workflow fails
    if not status:
        exit(1)
