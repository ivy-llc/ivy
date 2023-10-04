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


def get_submodule(test_path):
    test_path = test_path.split("/")
    submodule_test = test_path[-1]
    submodule, _ = submodule_test.split("::")
    submodule = submodule.replace("test_", "").replace(".py", "")
    return submodule


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
                backends = [test_arg[1]]
            if backends[0] == "all":
                backends = ["numpy", "jax", "tensorflow", "torch", "paddle"]

            test_path = test_arg[0]
            is_frontend = "test_frontends" in test_path
            collection = db["frontend_tests"] if is_frontend else db["ivy_tests"]
            submodule = get_submodule(test_path)
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

            contents = json.load(open("report.json"))

            backend_specific_info = dict()
            test_info = {
                "_id": (
                    contents["frontend_func"] if is_frontend else contents["fn_name"]
                ),
                "test_path": test_path,
                "submodule": submodule,
            }

            for backend in status:
                backend_specific_info[backend] = {
                    "status": {device: status[backend]},
                }
                if status[backend]:
                    backend_specific_info[backend] = {
                        versions[backend]: {
                            **backend_specific_info[backend],
                            "status": {device: status[backend]},
                            "nodes": contents["nodes"][backend],
                            "time": contents["time"][backend],
                            "args": contents["args"][backend],
                            "kwargs": contents["kwargs"][backend],
                        }
                    }
            test_info["results"] = backend_specific_info

            if is_frontend:
                frontend = test_path[test_path.find("test_frontends") :].split(os.sep)[
                    1
                ][5:]
                frontend_version = get_latest_package_version(frontend)
                test_info = {
                    **test_info,
                    "frontend": frontend,
                    "fw_time": contents["fw_time"],
                    "ivy_nodes": contents["ivy_nodes"],
                }
                test_info["results"] = {frontend_version: test_info["results"]}

            json.dump({"$set": test_info}, open("output.json", "w"))
            id = test_info.pop("_id")
            print(collection.update_one({"_id": id}, {"$set": test_info}, upsert=True))

    if any(not result for _, result in status.items()):
        exit(1)
