# Run Tests
import os
import sys
from pymongo import MongoClient
import requests
import json


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
            test_function_idx = test_file_content.find(test_function.split(",")[0])
            function_name = test_file_content[
                test_file_content[:test_function_idx].find('fn_tree="') + 9 :
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

    device = "cpu"
    if gpu_flag == "true":
        device = "gpu"

    status = dict()
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["ci_dashboard"]

    if device == "gpu":
        os.system("docker pull unifyai/multicuda:base_and_requirements")

    with open("tests_to_run", "r") as f:
        for line in f:
            print(f"\n{'*' * 100}")
            print(f"{line[:-1]}")
            print(f"{'*' * 100}\n")

            backends = ["all"]
            test_arg = line.split(",")
            if len(test_arg) > 1:
                backends = [test_arg[1].strip()]
            if backends[0] == "all":
                backends = ["numpy", "jax", "tensorflow", "torch", "paddle"]

            test_path = test_arg[0].strip()
            is_frontend_test = "test_frontends" in test_path
            collection = db["frontend_tests"] if is_frontend_test else db["ivy_tests"]
            submodule, function_name = get_submodule_and_function_name(
                test_path, is_frontend_test
            )
            versions = dict()

            for backend in backends:
                versions[backend] = get_latest_package_version(backend)
                if version_flag == "true":
                    # This would most probably break at the moment
                    [backend, backend_version] = backend.split("/")
                    versions[backend] = backend_version
                    command = (
                        f"docker run --rm --env REDIS_URL={redis_url} --env"
                        f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                        ' "$(pwd)"/.hypothesis:/.hypothesis'
                        ' unifyai/multiversion:latest /bin/bash -c "cd docker;python'
                        f" multiversion_framework_directory.py {' '.join(backends)};cd"
                        f' ..;pytest --tb=short {test_path} --backend={backend}"'
                    )
                elif device == "gpu":
                    command = (
                        f"docker run --rm --gpus all --env REDIS_URL={redis_url} --env"
                        f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                        ' "$(pwd)"/.hypothesis:/.hypothesis'
                        " unifyai/multicuda:base_and_requirements python3 -m pytest"
                        f" --tb=short {test_path} --device=gpu:0 -B={backend}"
                    )
                else:
                    command = (
                        f"docker run --rm --env REDIS_URL={redis_url} --env"
                        f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                        ' "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3'
                        f" -m pytest --tb=short {test_path} --backend {backend}"
                    )

                sys.stdout.flush()
                status[backend] = not os.system(command)
                if status[backend]:
                    command = (
                        f"docker run --rm --env REDIS_URL={redis_url} --env"
                        f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                        ' "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3'
                        f" -m pytest --tb=short {test_path} --backend"
                        f" {backend} --num-examples 1 --with-transpile"
                    )
                    ret = os.system(command)

            report_path = os.path.join(
                __file__[: __file__.rfind(os.sep)], "report.json"
            )
            report_content = {}
            if os.path.exists(report_path):
                report_content = json.load(open(report_path))

            backend_specific_info = dict()
            test_info = {
                "_id": function_name,
                "test_path": test_path,
                "submodule": submodule,
            }

            for backend in status:
                backend_specific_info[backend] = {
                    "status": {device: status[backend]},
                }
                if status[backend] and report_content:
                    backend_specific_info[backend] = {
                        versions[backend]: {
                            **backend_specific_info[backend],
                            "nodes": report_content["nodes"][backend],
                            "time": report_content["time"][backend],
                            "args": report_content["args"][backend],
                            "kwargs": report_content["kwargs"][backend],
                        }
                    }
            test_info["results"] = backend_specific_info

            if is_frontend_test:
                frontend = test_path[test_path.find("test_frontends") :].split(os.sep)[
                    1
                ][5:]
                frontend_version = get_latest_package_version(frontend)
                test_info["frontend"] = frontend
                if report_content:
                    test_info = {
                        **test_info,
                        "fw_time": report_content["fw_time"],
                        "ivy_nodes": report_content["ivy_nodes"],
                    }
                test_info["results"] = {frontend_version: test_info["results"]}

            id = test_info.pop("_id")
            print(collection.update_one({"_id": id}, {"$set": test_info}, upsert=True))

    if any(not result for _, result in status.items()):
        exit(1)
