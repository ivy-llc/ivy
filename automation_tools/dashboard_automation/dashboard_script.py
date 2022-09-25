from github import Github
from pymongo import MongoClient
import sys
import pandas as pd

dashboard_config = {
    "test-array-api": ["array_api", "array_api_dashboard.md"],
    "test-core-ivy": ["ivy_core", "functional_core_dashboard.md"],
    "test-nn-ivy": ["ivy_nn", "functional_nn_dashboard.md"],
    "test-stateful-ivy": ["ivy_stateful", "stateful_dashboard.md"],
}


def update_dashboard():
    key, token, workflow = sys.argv[1], sys.argv[2], sys.argv[3]
    g = Github(token)
    repo = g.get_repo("unifyai/ivy")
    cluster = MongoClient(
        f"mongodb+srv://deep-ivy:{key}@cluster0.qdvf8q3.mongodb.net/?retryWrites=true&w=majority"
    )
    db = cluster["Ivy_tests"]
    collection = db[dashboard_config[workflow][0]]
    cursor = list(collection.find())
    workflow_dict = cursor[0]
    workflow_dict.pop("_id")
    module_df = pd.DataFrame.from_dict(workflow_dict)
    file = repo.get_contents(
        f"test_dashboards/{dashboard_config[workflow][1]}", ref="dashboard"
    )
    repo.update_file(
        file.path,
        f"update {dashboard_config[workflow][1]}",
        module_df.to_markdown(),
        file.sha,
        branch="dashboard",
    )


if __name__ == "__main__":
    update_dashboard()
