# Run Tests
import os
import subprocess
import sys
from pymongo import MongoClient

result_config = {
    "success": "https://img.shields.io/badge/-success-success",
    "failure": "https://img.shields.io/badge/-failure-red",
}


def make_clickable(url, name):
    return '<a href="{}" rel="noopener noreferrer" '.format(
        url
    ) + 'target="_blank"><img src={}></a>'.format(name)


def get_submodule(test_path):
    test_path = test_path.split("/")
    submod_test = test_path[-1]
    submod, test_fn = submod_test.split("::")
    submod = submod.replace("test_", "").replace(".py", "")
    return ["array_api", 0], submod, test_fn


def update_individual_test_results(
    collection,
    id,
    submod,
    backend,
    test,
    result,
    backend_version=None,
    frontend_version=None,
):
    key = submod + "." + backend
    if backend_version is not None:
        backend_version = backend_version.replace(".", "_")
        key += "." + backend_version
    if frontend_version is not None:
        frontend_version = frontend_version.replace(".", "_")
        key += "." + frontend_version
    key += "." + test
    collection.update_one(
        {"_id": id},
        {"$set": {key: result}},
        upsert=True,
    )


BACKENDS = ["numpy", "jax", "tensorflow", "torch"]


def main():
    redis_url = sys.argv[1]
    redis_pass = sys.argv[2]
    mongo_key = sys.argv[3]
    workflow_id = sys.argv[4]
    if len(sys.argv) > 5:
        run_id = sys.argv[5]
    else:
        run_id = "https://github.com/unifyai/ivy/actions/runs/" + workflow_id
    failed = False
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{mongo_key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"  # noqa
    )
    db = cluster["Ivy_tests_multi"]
    k_flag = {}
    subprocess.run(
        ["python3", "ivy_tests/array_api_testing/write_array_api_tests_k_flag.py"],
        check=True,
    )
    for backend in BACKENDS:
        k_flag_file = f"ivy_tests/array_api_testing/.array_api_tests_k_flag_{backend}"
        with open(k_flag_file, "r") as f:
            array_api_tests_k_flag = f.read().strip()
        if backend == "torch":
            array_api_tests_k_flag += " and not (uint16 or uint32 or uint64)"
        k_flag[backend] = array_api_tests_k_flag

    with open("tests_to_run", "r") as f:
        for line in f:
            test, backend = line.split(",")
            backend = backend.strip("\n")
            coll, submod, test_fn = get_submodule(test)
            command = f'docker run --rm --env IVY_BACKEND={backend} --env ARRAY_API_TESTS_MODULE="ivy" --env REDIS_URL={redis_url} --env REDIS_PASSWD={redis_pass} -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest timeout 30m python3 -m pytest {test} -k "{k_flag[backend]}" --tb=short -vv'  # noqa
            print(f"\n{'*' * 100}")
            print(f"{line[:-1]}")
            print(f"{'*' * 100}\n")
            sys.stdout.flush()
            ret = os.system(command)
            if ret != 0:
                res = make_clickable(run_id, result_config["failure"])
                failed = True
            else:
                res = make_clickable(run_id, result_config["success"])
            update_individual_test_results(
                db[coll[0]],
                coll[1],
                submod,
                backend,
                test_fn,
                res,
                "latest-stable",
            )
    if failed:
        exit(1)


if __name__ == "__main__":
    main()
