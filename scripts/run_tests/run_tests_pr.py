# Run Tests
import os
import sys
from pymongo import MongoClient

from scripts.run_tests.run_tests import (
    get_latest_package_version,
    get_submodule_and_function_name,
)


if __name__ == "__main__":
    failed = False
    cluster = MongoClient(
        "mongodb+srv://readonly-user:hvpwV5yVeZdgyTTm@cluster0.qdvf8q3.mongodb.net"
    )
    db = cluster["ci_dashboard"]
    with open(sys.argv[1], "w") as f_write:
        with open("tests_to_run", "r") as f:
            for line in f:
                test_path, backend = line.strip().split(",")
                is_frontend_test = "test_frontends" in test_path
                collection = (
                    db["frontend_tests"] if is_frontend_test else db["ivy_tests"]
                )
                submodule, function_name = get_submodule_and_function_name(
                    test_path, is_frontend_test
                )
                version = get_latest_package_version(backend).replace(".", "_")
                print(f"\n{'*' * 100}")
                print(f"{line[:-1]}")
                print(f"{'*' * 100}\n")
                sys.stdout.flush()
                ret = os.system(
                    f'docker run --rm -v "$(pwd)":/ivy -v "$(pwd)"/.hypothesis:/.hypothesis unifyai/ivy:latest python3 -m pytest --tb=short {test_path} --skip-trace-testing --backend {backend}'  # noqa
                )
                if ret != 0:
                    failed = True
                    document = collection.find_one({"_id": function_name})
                    if document:
                        try:
                            if document[backend][version]["status"]["cpu"]:
                                line = line.strip("\n") + " (main: pass)\n"
                        except KeyError:
                            print(
                                f"Could not find {backend}.{version}.status.cpu for"
                                f" document : {document}"
                            )
                    f_write.write(line)

    if failed:
        sys.exit(1)
