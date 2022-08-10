import sys
import requests
import json
import emoji
import pandas as pd


url = "https://api.github.com/repos/unifyai/ivy/actions/runs?branch=master"

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": "",
}
functional_nn_dict = dict()
functional_core_dict = dict()
stateful_dict = dict()

output_files: dict = {
    0: "functional_core_dashboard",
    1: "functional_nn_dashboard",
    2: "stateful_dashboard",
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
        data.at[index_label] = [row_series.values[i][1] for i in range(4)]
    return data


def workflow_results(token):
    output = get_api_results(url, token, headers)
    for info in output["workflow_runs"]:
        if info["name"] not in (
            "test-core-ivy",
            "test-stateful-ivy",
            "test-nn-ivy",
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
    workflow_df = workflow_df.drop_duplicates(subset=["name"], keep="first")
    return workflow_df


def get_matrix_job_data(token):
    # list all workflows running for the branch
    workflow_df = workflow_results(token)
    # extract jobs from the workflows above
    for name, jobs_url in zip(workflow_df["name"], workflow_df["jobs_url"]):
        for info in get_api_results(jobs_url + "?per_page=100", token, headers)["jobs"]:
            # extract backend and submodule name from json
            backend = info["name"].strip("run-nightly-tests")[2:-1].split(",")[0]
            submodule = info["name"].strip("run-nightly-tests")[2:-1].split(",")[1]

            if name == "test-core-ivy":
                if submodule not in functional_core_dict:
                    functional_core_dict[submodule] = []
                if info["conclusion"] == "failure":
                    functional_core_dict[submodule].append(
                        (backend, emoji.emojize(":x:", language="alias"))
                    )
                elif info["conclusion"] == "success":
                    functional_core_dict[submodule].append(
                        (backend, emoji.emojize(":white_check_mark:", language="alias"))
                    )
                else:
                    functional_core_dict[submodule].append(
                        (backend, emoji.emojize(":clock9:", language="alias"))
                    )
            elif name == "test-nn-ivy":
                if submodule not in functional_nn_dict:
                    functional_nn_dict[submodule] = []
                if info["conclusion"] == "failure":
                    functional_nn_dict[submodule].append(
                        (backend, emoji.emojize(":x:", language="alias"))
                    )
                elif info["conclusion"] == "success":
                    functional_nn_dict[submodule].append(
                        (backend, emoji.emojize(":white_check_mark:", language="alias"))
                    )
                else:
                    functional_nn_dict[submodule].append(
                        (backend, emoji.emojize(":clock9:", language="alias"))
                    )

            elif name == "test-stateful-ivy":
                if submodule not in stateful_dict:
                    stateful_dict[submodule] = []
                if info["conclusion"] == "failure":
                    stateful_dict[submodule].append(
                        (backend, emoji.emojize(":x:", language="alias"))
                    )
                elif info["conclusion"] == "success":
                    stateful_dict[submodule].append(
                        (backend, emoji.emojize(":white_check_mark:", language="alias"))
                    )
                else:
                    stateful_dict[submodule].append(
                        (backend, emoji.emojize(":clock9:", language="alias"))
                    )

    return (functional_core_dict, functional_nn_dict, stateful_dict)


def main():
    path, token = (str(sys.argv[1]), str(sys.argv[2]))
    ivy_modules = get_matrix_job_data(token)
    for i, module in enumerate(ivy_modules):
        module_df = get_DataFrame(module)
        module_df.to_html(f"{path}/{output_files[i]}.html")


if __name__ == "__main__":
    main()
