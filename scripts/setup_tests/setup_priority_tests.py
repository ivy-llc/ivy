import sys
from pymongo import MongoClient
from get_all_tests import BACKENDS


def main(cron=False):
    # connect to the database
    run_id = int(sys.argv[1]) - 1
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
        if ivy_test_doc.get("demos", None)
    ]
    frontend_functions = [
        frontend_test_doc["_id"]
        for frontend_test_doc in frontend_test_docs
        if frontend_test_doc.get("demos", None)
    ]
    ivy_functions = sorted(list(set(ivy_functions)))
    frontend_functions = sorted(list(set(frontend_functions)))

    # find corresponding test paths for those functions
    ivy_test_paths = []
    frontend_test_paths = []
    for function in ivy_functions:
        result = ivy_tests_collection.find_one({"_id": function})
        if result:
            ivy_test_paths.append(result["test_path"])
    for function in frontend_functions:
        result = frontend_tests_collection.find_one({"_id": function})
        if result:
            frontend_test_paths.append(result["test_path"])

    # split the tests based on the runner
    test_paths = ivy_test_paths + frontend_test_paths
    test_names = []
    for i, test_path in enumerate(test_paths):
        for backend in BACKENDS:
            test_names.append(f"{test_path},{backend}")

    if cron:
        with open("tests_to_run", "w") as write_file:
            for test_name in test_names:
                test_name = test_name.strip()
                write_file.write(test_name + "\n")
    else:
        # add those paths to the tests_to_run based on the runner
        tests_per_run = 10
        num_tests = len(test_names)
        start = run_id * tests_per_run
        end = (run_id + 1) * tests_per_run
        print("Running Tests:")
        with open("tests_to_run", "w") as write_file:
            for i in range(start, end):
                i = i % num_tests
                test = test_names[i].strip()
                print(test)
                write_file.write(test + "\n")


if __name__ == "__main__":
    main()
