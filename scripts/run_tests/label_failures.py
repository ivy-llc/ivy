# Run Tests
import sys
from pymongo import MongoClient

from helpers import (
    get_latest_package_version,
    get_submodule_and_function_name,
)


if __name__ == "__main__":
    failed = False
    cluster = MongoClient(
        "mongodb+srv://readonly-user:hvpwV5yVeZdgyTTm@cluster0.qdvf8q3.mongodb.net"
    )
    db = cluster["ci_dashboard"]
    with open(sys.argv[2], "w") as f_write:
        with open(sys.argv[1], "r") as f:
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
                document = collection.find_one({"_id": function_name})
                if document:
                    try:
                        if document[backend][version]["status"]["cpu"]:
                            line = line.strip("\n") + " (main: pass)\n"
                    except KeyError:
                        print(
                            f"Could not find {backend}.{version}.status.cpu for"
                            " document"
                        )
                f_write.write(line)
