import argparse
import json
from pymongo import MongoClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add all the tests within test_report.json to the remote MongoDB."
    )
    parser.add_argument(
        "--workflow-link",
        type=str,
        help="Link to the GitHub actions workflow corresponding to this test.",
    )
    parser.add_argument("--db-key", type=str, help="Key for the MongoDB database")

    args = parser.parse_args()
    json_report_file = "test_report.json"

    with open(json_report_file, "r") as file:
        data = json.load(file)

    tests_data = data.get("tests", None)

    uri = f"mongodb+srv://{args.db_key}@ivytestdashboard.mnzyom5.mongodb.net/?retryWrites=true&w=majority&appName=IvyTestDashboard"
    client = MongoClient(uri)
    db = client.ivytestdashboard
    collection = db["test_results"]

    for test in tests_data:
        test_path, test_function_args = test["nodeid"].split("::")
        test_path = test_path.replace("ivy_tests/test_ivy/", "")
        test_function, test_args = test_function_args.split("[")
        test_function = test_function.replace("test_", "")
        test_args = test_args.replace("]", "")
        test_args = test_args.split("-")
        backend = test_args[1]

        if "frontends" in test_path:
            test_function = test_function.split("_", 1)[1]

        document = {
            "backend": backend,
            "function": test_function,
            "path": test_path,
            "workflow_link": args.workflow_link,
            "outcome": test["outcome"],
        }
        filter_criteria = {
            "backend": backend,
            "function": test_function,
            "path": test_path,
        }

        result = collection.replace_one(filter_criteria, document, upsert=True)
