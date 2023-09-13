# Run Tests
import os
import sys
from pymongo import MongoClient
import requests
from run_tests_CLI.get_all_tests import BACKENDS


submodules = (
    "test_paddle",
    "test_tensorflow",
    "test_torch",
    "test_jax",
    "test_numpy",
    "test_functional",
    "test_experimental",
    "test_stateful",
    "test_misc",
    "test_scipy",
    "test_pandas",
    "test_mindspore",
    "test_onnx",
    "test_sklearn",
    "test_xgboost",
)
db_dict = {
    "test_functional/test_core": ["core", 10],
    "test_experimental/test_core": ["exp_core", 11],
    "test_functional/test_nn": ["nn", 12],
    "test_experimental/test_nn": ["exp_nn", 13],
    "test_stateful": ["stateful", 14],
    "test_torch": ["torch", 15],
    "test_jax": ["jax", 16],
    "test_tensorflow": ["tensorflow", 17],
    "test_numpy": ["numpy", 18],
    "test_misc": ["misc", 19],
    "test_paddle": ["paddle", 20],
    "test_scipy": ["scipy", 21],
    "test_pandas": ["pandas", 22],
    "test_mindspore": ["mindspore", 23],
    "test_onnx": ["onnx", 24],
    "test_sklearn": ["sklearn", 25],
    "test_xgboost": ["xgboost", 26],
}
result_config = {
    "success": "https://img.shields.io/badge/-success-success",
    "failure": "https://img.shields.io/badge/-failure-red",
}


def get_latest_package_version(package_name):
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        response.raise_for_status()
        package_info = response.json()
        return package_info["info"]["version"]
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch package information for {package_name}.")
        return None


def make_clickable(url, name):
    return (
        f'<a href="{url}" rel="noopener noreferrer" '
        + f'target="_blank"><img src={name}></a>'
    )


def get_submodule(test_path):
    test_path = test_path.split("/")
    for name in submodules:
        if name in test_path:
            if name == "test_functional":
                if len(test_path) > 3 and test_path[3] == "test_experimental":
                    coll = db_dict[f"test_experimental/{test_path[4]}"]
                else:
                    coll = db_dict[f"test_functional/{test_path[-2]}"]
            else:
                coll = db_dict[name]
            break
    submod_test = test_path[-1]
    submod, test_fn = submod_test.split("::")
    submod = submod.replace("test_", "").replace(".py", "")
    return coll, submod, test_fn


def update_individual_test_results(
    collection,
    id,
    submod,
    backend,
    test,
    result,
    backend_version=None,
    frontend_version=None,
    device=None,
):
    key = f"{submod}.{backend}"
    if backend_version is not None:
        backend_version = backend_version.replace(".", "_")
        key += f".{backend_version}"
    if frontend_version is not None:
        frontend_version = frontend_version.replace(".", "_")
        key += f".{frontend_version}"
    key += f".{test}"
    if device:
        key += f".{device}"
    collection.update_one(
        {"_id": id},
        {"$set": {key: result}},
        upsert=True,
    )


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
    failed = False
    # GPU Testing
    with_gpu = False
    if gpu_flag == "true":
        with_gpu = True
    if priority_flag == "true":
        priority_flag = True
    else:
        priority_flag = False
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["Ivy_tests_multi_gpu"]
    db_priority = cluster["Ivy_tests_priority"]
    if with_gpu:
        os.system("docker pull unifyai/multicuda:base_and_requirements")
    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            coll, submod, test_fn = get_submodule(test)
            print(f"\n{'*' * 100}")
            print(f"{line[:-1]}")
            print(f"{'*' * 100}\n")
            sys.stdout.flush()
            if version_flag == "true":
                backends = [backend]
                other_backends = [fw for fw in BACKENDS if fw != backend.split("/")[0]]
                for backend in other_backends:
                    backends.append(backend + "/" + get_latest_package_version(backend))
                print("Backends:", backends)
                ret = os.system(
                    f"docker run --rm --env REDIS_URL={redis_url} --env"
                    f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                    ' "$(pwd)"/.hypothesis:/.hypothesis unifyai/multiversion:latest'
                    f" python docker/multiversion_framework_directory.py {backends};"
                    f"python -m pytest --tb=short {test} --backend={backend}"
                )
            else:
                if with_gpu:
                    ret = os.system(
                        f"docker run --rm --gpus all --env REDIS_URL={redis_url} --env"
                        f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                        ' "$(pwd)"/.hypothesis:/.hypothesis'
                        " unifyai/multicuda:base_and_requirements python3 -m pytest"
                        f" --tb=short {test} --device=gpu:0 -B={backend}"
                        # noqa
                    )
                else:
                    ret = os.system(
                        f"docker run --rm --env REDIS_URL={redis_url} --env"
                        f' REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v'
                        ' "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m'
                        f" pytest --tb=short {test} --backend {backend}"
                        # noqa
                    )
            if ret != 0:
                res = make_clickable(run_id, result_config["failure"])
                failed = True
            else:
                res = make_clickable(run_id, result_config["success"])
            frontend_version = None
            if coll[0] in ["numpy", "jax", "tensorflow", "torch", "paddle"]:
                frontend_version = "latest-stable"
            if priority_flag:
                print("Updating Priority DB")
                update_individual_test_results(
                    db_priority[coll[0]],
                    coll[1],
                    submod,
                    backend,
                    test_fn,
                    res,
                    "latest-stable",
                    frontend_version,
                    "gpu" if with_gpu else "cpu",
                )
            else:
                backend_version = (
                    backend.split("/")[1] if version_flag == "true" else "latest-stable"
                )
                print(backend_version)
                update_individual_test_results(
                    db[coll[0]],
                    coll[1],
                    submod,
                    backend,
                    test_fn,
                    res,
                    backend_version,
                    frontend_version,
                    "gpu" if with_gpu else "cpu",
                )

    if failed:
        exit(1)
