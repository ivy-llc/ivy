from get_all_tests import BACKENDS
from packaging import version
from pymongo import MongoClient
import requests
import sys


def get_latest_package_version(package_name):
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        package_info = response.json()
        versions = list(package_info["releases"].keys())
        key = lambda x: version.parse(x)
        return sorted(versions, key=key, reverse=True)
    except requests.exceptions.RequestException:
        print(f"Error: Failed to fetch package information for {package_name}.")
        return None


def main():
    # connect to the database
    priority = sys.argv[1] == "true"
    run_iter = int(sys.argv[2]) - 1
    cluster = MongoClient(
        "mongodb+srv://readonly-user:hvpwV5yVeZdgyTTm@cluster0.qdvf8q3.mongodb.net"
    )
    ci_dashboard_db = cluster["ci_dashboard"]
    ivy_tests_collection = ci_dashboard_db["ivy_tests"]
    frontend_tests_collection = ci_dashboard_db["frontend_tests"]

    # iterate over demos and collect ivy and frontend functions used
    ivy_test_docs = ivy_tests_collection.find()
    frontend_test_docs = frontend_tests_collection.find()
    ivy_functions = [
        ivy_test_doc["_id"]
        for ivy_test_doc in ivy_test_docs
        if not priority or ivy_test_doc.get("demos", None)
    ]
    frontend_functions = [
        frontend_test_doc["_id"]
        for frontend_test_doc in frontend_test_docs
        if not priority or frontend_test_doc.get("demos", None)
    ]
    ivy_functions = sorted(list(set(ivy_functions)))
    frontend_functions = sorted(list(set(frontend_functions)))
    versions = {
        backend: [
            version_name.replace(".", "_")
            for version_name in get_latest_package_version(backend)
        ]
        for backend in BACKENDS
    }

    # find corresponding test paths for those functions
    ivy_test_paths = []
    frontend_test_paths = []
    for function in ivy_functions:
        print("function", function)
        result = ivy_tests_collection.find_one({"_id": function})
        if result:
            for backend in BACKENDS:
                if backend in result:
                    for version_name in versions[backend]:
                        if version_name in result[backend]:
                            if "status" in result[backend][version_name]:
                                status = result[backend][version_name]["status"].get(
                                    "cpu"
                                )
                                if not status and status is not None:
                                    ivy_test_paths.append(
                                        f"{result['test_path']},{backend}"
                                    )
                                break

    for function in frontend_functions:
        print("frontend function", function)
        frontend = function.split(".")[0]
        result = frontend_tests_collection.find_one({"_id": function})
        if result and frontend in versions:
            for frontend_version in versions[frontend]:
                if frontend_version in result:
                    backend_result = result[frontend_version]
                    for backend in BACKENDS:
                        if backend in backend_result:
                            for version_name in versions[backend]:
                                if version_name in backend_result[backend]:
                                    if (
                                        "status"
                                        in backend_result[backend][version_name]
                                    ):
                                        status = backend_result[backend][version_name][
                                            "status"
                                        ].get("cpu")
                                        if not status and status is not None:
                                            frontend_test_paths.append(
                                                f"{result['test_path']},{backend}"
                                            )
                                        break

    all_tests = ivy_test_paths + frontend_test_paths
    all_tests = [test_path.strip() for test_path in all_tests]
    tests_per_run = 50
    num_tests = len(all_tests)
    start = run_iter * tests_per_run
    end = (run_iter + 1) * tests_per_run
    end = min(end, num_tests)
    if start < end:
        tests = all_tests[start:end]
    else:
        tests = []

    # add those paths to the tests_to_run
    with open("tests_to_run", "w") as write_file:
        write_file.write("\n".join(tests))


if __name__ == "__main__":
    main()
