# Run Tests
import os
import sys
from pymongo import MongoClient
from pymongo.errors import WriteError
import json
import old_run_test_helpers as old_helpers
from helpers import (
    get_latest_package_version,
    get_submodule_and_function_name,
)
from get_all_tests import BACKENDS


if __name__ == "__main__":
    redis_url = sys.argv[1]
    redis_pass = sys.argv[2]
    mongo_key = sys.argv[3]
    version_flag = sys.argv[4]
    gpu_flag = sys.argv[5]
    workflow_id = sys.argv[6]
    priority_flag = sys.argv[7]
    tracer_flag = sys.argv[8]
    tracer_flag_each = sys.argv[9]

    if len(sys.argv) > 10 and sys.argv[10] != "null":
        run_id = sys.argv[10]
    else:
        run_id = f"https://github.com/unifyai/ivy/actions/runs/{workflow_id}"

    device = "cpu"
    if gpu_flag == "true":
        device = "gpu"

    tracer_str = ""
    if tracer_flag == "true":
        tracer_flag = "tracer_"
        tracer_str = " --with-trace-testing"
    else:
        tracer_flag = ""

    tracer_str_each = ""
    if not tracer_flag and tracer_flag_each == "true":
        tracer_flag_each = "tracer_each_"
        tracer_str_each = " --with-trace-testing-each"
    else:
        tracer_flag_each = ""

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
        os.system("docker pull ivyllc/ivy:latest-gpu")

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
                    fw for fw in BACKENDS if (fw not in (backend_name, "paddle"))
                ]
                for other_backend in other_backends:
                    backends.append(
                        other_backend + "/" + get_latest_package_version(other_backend)
                    )
                print("Backends:", backends)
                os.system(
                    'docker run --name test-container -v "$(pwd)":/ivy/ivy '
                    f"-e REDIS_URL={redis_url} -e REDIS_PASSWD={redis_pass} "
                    "-itd unifyai/multiversion:latest /bin/bash -c"
                    f'python multiversion_framework_directory.py {" ".join(backends)};'
                )
                os.system(
                    "docker exec test-container cd ivy; python3 -m pytest --tb=short "
                    f"{test_path} --backend={backend.strip()}"
                )
                backend = backend.split("/")[0] + "\n"
                backend_version = backend_version.strip()

            else:
                device_str = ""
                device_access_str = ""
                image = "ivyllc/ivy:latest"

                # gpu tests
                if device == "gpu":
                    image = "ivyllc/ivy:latest-gpu"
                    device_str = " --device=gpu:0"
                    device_access_str = " --gpus all"
                    os.system("docker pull ivyllc/ivy:latest-gpu")

                os.system(
                    f"docker run{device_access_str} --name test-container -v "
                    '"$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis -e '
                    f"REDIS_URL={redis_url} -e REDIS_PASSWD={redis_pass} -itd {image}"
                )
                command = (
                    "docker exec test-container python3 -m pytest --tb=short"
                    f" {test_path}{device_str} --backend {backend}"
                    f"{tracer_str}{tracer_str_each}"
                )
                os.system(command)

            # run the test
            sys.stdout.flush()
            failed = bool(os.system(command))

            # old (populate the old database with results)
            if not failed:
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
            try:
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
                        old_db[coll[0]],
                        coll[1],
                        submod,
                        backend,
                        test_fn,
                        res,
                        backend_version,
                        frontend_version,
                        device,
                    )
            except WriteError:
                print("Old DB Write Error")

            # skip updating db for instance methods as of now
            # run transpilation tests if the test passed
            if not failed and function_name:
                print(f"\n{'*' * 100}")
                print(f"{line[:-1]} --> transpilation tests")
                print(f"{'*' * 100}\n")
                command = f"{command} --num-examples 5 --with-transpile"
                sys.stdout.flush()
                os.system(command)
                os.system(
                    "docker cp test-container:/ivy/report.json"
                    f" {__file__[: __file__.rfind(os.sep)]}/report.json"
                )

            # load data from report if generated
            report_path = os.path.join(
                __file__[: __file__.rfind(os.sep)], "report.json"
            )
            report_content = {}
            if os.path.exists(report_path):
                report_content = json.load(open(report_path))

            # create a prefix str for the update query for frontend tests
            # (with frontend version)
            test_info = {}
            prefix_str = ""
            if is_frontend_test:
                frontend = test_path[test_path.find("test_frontends") :].split(os.sep)[
                    1
                ][5:]
                frontend_version = get_latest_package_version(frontend).replace(
                    ".", "_"
                )
                test_info["frontend"] = frontend
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
                f"{prefix_str}{backend}.{version}.{tracer_flag}{tracer_flag_each}"
                f"status.{device}": (not failed),
                f"{prefix_str}{backend}.{version}.{tracer_flag}{tracer_flag_each}"
                f"workflow.{device}": (run_id),
            }

            # add transpilation metrics if report generated
            if not failed and report_content and not (tracer_flag or tracer_flag_each):
                if is_frontend_test:
                    test_info = {
                        **test_info,
                        "fw_time": report_content["fw_time"],
                        "ivy_nodes": report_content["ivy_nodes"],
                    }
                transpilation_metrics = {
                    "nodes": report_content["nodes"][backend],
                    "time": report_content["time"][backend],
                    "args": report_content["args"][backend],
                    "kwargs": report_content["kwargs"][backend],
                }
                for metric, value in transpilation_metrics.items():
                    test_info[f"{prefix_str}{backend}.{version}.{metric}"] = value

            # populate the ci_dashboard db, skip instance methods
            if function_name:
                id = test_info.pop("_id")
                print(
                    collection.update_one({"_id": id}, {"$set": test_info}, upsert=True)
                )

            # delete the container
            os.system("docker rm -f test-container")

    # if any tests fail, the workflow fails
    if failed:
        sys.exit(1)
