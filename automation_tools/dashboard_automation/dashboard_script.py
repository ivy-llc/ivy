import sys
import requests
import json
import emoji
import pandas as pd
from github import Github
from typing import Dict, Union

url = "https://api.github.com/repos/unifyai/ivy/actions/runs?branch=master&per_page=100"

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": "",
}
functional_nn_dict = dict()
functional_core_dict = dict()
stateful_dict = dict()
array_api_dict = dict()

config: Dict[Union[str, int], Union[str, type(emoji)]] = {
    0: "functional_core_dashboard",
    1: "functional_nn_dashboard",
    2: "stateful_dashboard",
    3: "array_api_dashboard",
    "success": emoji.emojize(":white_check_mark:", language="alias"),
    "failure": emoji.emojize(":x:", language="alias"),
    "in_progress": emoji.emojize(":hourglass:", language="alias"),
    "cancelled": emoji.emojize(":red_circle:", language="alias"),
}
results = []


def get_api_results(url, token, headers):
    headers["Authorization"] = "Bearer " + token
    response = requests.request("GET", url, headers=headers)
    return json.loads(response.text)


def get_DataFrame(result_dict: dict) -> pd.DataFrame:
    data = pd.DataFrame.from_dict(
        result_dict, orient="index", columns=["Numpy", "Torch", "Jax", "Tensorflow"]
    )
    data.index.names = ["Submodules"]
    for (index_label, row_series) in data.iterrows():
        data.at[index_label] = [
            row_series.values[i][1]
            if len(row_series[i]) < 3
            else ("   ").join(row_series.values[i][1:])
            for i in range(4)
        ]
    return data


def make_clickable(url, name):
    return '<a href="{}" rel="noopener noreferrer" target="_blank">{}</a>'.format(
        url, name
    )


def get_previous_workflow(workflow_df):
    grouped = workflow_df.groupby("created_at")
    if len(grouped) <= 1:
        return None
    for i, (name, group) in enumerate(grouped):
        if i == 1:
            return group


def workflow_results(token):
    output = get_api_results(url, token, headers)
    for info in output["workflow_runs"]:
        if info["name"] not in (
            "test-core-ivy",
            "test-stateful-ivy",
            "test-nn-ivy",
            "test-array-api",
        ):
            continue
        results.append(
            (
                info["id"],
                info["name"],
                info["jobs_url"],
                info["status"],
                info["created_at"],
            )
        )
    workflow_df = pd.DataFrame(
        results, columns=["id", "name", "jobs_url", "status", "created_at"]
    )
    workflow_df_prev = get_previous_workflow(workflow_df)
    workflow_df_latest = workflow_df.drop_duplicates(subset=["name"], keep="first")
    return workflow_df_prev, workflow_df_latest


def get_previous_job_data(
    workflow_df_prev, workflow_name, job_name, prev_action_link, prev_conclusion, token
):
    jobs_url_object = workflow_df_prev.loc[workflow_df_prev["name"] == workflow_name]
    jobs_url = [url for url in jobs_url_object["jobs_url"]]
    for info in get_api_results(jobs_url[0] + "?per_page=100", token, headers)["jobs"]:
        if info["name"] == job_name:
            # return when previous job is queued as well
            if info["status"] in ("in_progress", "queued"):
                return (prev_action_link, prev_conclusion)
            return (info["html_url"], config[info["conclusion"]])


def get_matrix_job_data(token):
    # list all workflows running for the branch
    workflow_df_prev, workflow_df_latest = workflow_results(token)
    # extract jobs from the workflows above
    for name, jobs_url in zip(
        workflow_df_latest["name"], workflow_df_latest["jobs_url"]
    ):
        for info in get_api_results(jobs_url + "?per_page=100", token, headers)["jobs"]:
            prev_action_link, prev_conclusion = (None, None)
            # extract backend and submodule name from json
            if name == "test-array-api":
                backend_submod = (
                    info["name"].strip("run-array-api-tests")[2:-1].split(",")
                )
                backend, submodule = backend_submod[0], backend_submod[1]
            else:
                backend_submod = (
                    info["name"].strip("run-nightly-tests")[2:-1].split(",")
                )
                backend, submodule = backend_submod[0], backend_submod[1]

            if info["status"] in ("in_progress, queued"):
                conclusion = config["in_progress"]
                if workflow_df_prev is not None:
                    prev_action_link, prev_conclusion = get_previous_job_data(
                        workflow_df_prev,
                        name,
                        info["name"],
                        prev_action_link,
                        prev_conclusion,
                        token,
                    )
            else:
                conclusion = config[info["conclusion"]]

            if name == "test-array-api":
                if submodule not in array_api_dict:
                    array_api_dict[submodule] = []
                if prev_action_link and prev_conclusion is not None:
                    array_api_dict[submodule].append(
                        (
                            backend,
                            make_clickable(prev_action_link, prev_conclusion),
                            make_clickable(info["html_url"], conclusion),
                        )
                    )
                else:
                    array_api_dict[submodule].append(
                        (backend, make_clickable(info["html_url"], conclusion))
                    )

            if name == "test-core-ivy":
                if submodule not in functional_core_dict:
                    functional_core_dict[submodule] = []
                if prev_action_link and prev_conclusion is not None:
                    functional_core_dict[submodule].append(
                        (
                            backend,
                            make_clickable(prev_action_link, prev_conclusion),
                            make_clickable(info["html_url"], conclusion),
                        )
                    )
                else:
                    functional_core_dict[submodule].append(
                        (backend, make_clickable(info["html_url"], conclusion))
                    )
            elif name == "test-nn-ivy":
                if submodule not in functional_nn_dict:
                    functional_nn_dict[submodule] = []
                if prev_action_link or prev_conclusion is not None:
                    functional_nn_dict[submodule].append(
                        (
                            backend,
                            make_clickable(prev_action_link, prev_conclusion),
                            make_clickable(info["html_url"], conclusion),
                        )
                    )
                else:
                    functional_nn_dict[submodule].append(
                        (backend, make_clickable(info["html_url"], conclusion))
                    )
            elif name == "test-stateful-ivy":
                if submodule not in stateful_dict:
                    stateful_dict[submodule] = []
                if prev_action_link or prev_conclusion is not None:
                    stateful_dict[submodule].append(
                        (
                            backend,
                            make_clickable(prev_action_link, prev_conclusion),
                            make_clickable(info["html_url"], conclusion),
                        )
                    )
                else:
                    stateful_dict[submodule].append(
                        (backend, make_clickable(info["html_url"], conclusion))
                    )

    return (functional_core_dict, functional_nn_dict, stateful_dict, array_api_dict)


def main():
    token = str(sys.argv[1])
    g = Github(token)
    repo = g.get_repo("unifyai/ivy")
    ivy_modules = get_matrix_job_data(token)
    for i, module in enumerate(ivy_modules):
        module_df = get_DataFrame(module)
        file = repo.get_contents(f"test_dashboards/{config[i]}.md", ref="dashboard")
        repo.update_file(
            file.path,
            f"update {config[i]}",
            module_df.to_markdown(),
            file.sha,
            branch="dashboard",
        )


if __name__ == "__main__":
    main()
