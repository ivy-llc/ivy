from github import Github
import sys
import urllib
import json

test_name_mappings = {
    "test-ivy-core": ["core.json", "functional_core_dashboard.md"],
    "test-ivy-core-cron": ["core.json", "functional_core_dashboard.md"],
    "test-ivy-nn": ["nn.json", "functional_nn_dashboard.md"],
    "test-ivy-nn-cron": ["nn.json", "functional_nn_dashboard.md"],
    "test-ivy-stateful": ["stateful.json", "stateful_dashboard.md"],
    "test-ivy-stateful-cron": ["stateful.json", "stateful_dashboard.md"],
    "test-array-api": ["array_api.json", "array_api_dashboard.md"],
}


def update_job_status(file, fw, submod, result):
    with urllib.request.urlopen(file.download_url) as f:
        data = json.load(f)
        data[fw][submod] = result
        print(data)
        return data


def main():
    token, workflow_name, backend, submodule, result = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
    )
    print(workflow_name, backend, submodule, result)
    g = Github(token)
    repo = g.get_repo("unifyai/ivy")
    file = repo.get_contents(
        f"results/{test_name_mappings[workflow_name][0]}", ref="dashboard"
    )
    updated_json = update_job_status(file, backend, submodule, result)
    repo.update_file(
        file.path,
        f"update {test_name_mappings[workflow_name][0]}",
        json.dumps(updated_json, indent=6),
        file.sha,
        branch="dashboard",
    )


if __name__ == "__main__":
    main()
