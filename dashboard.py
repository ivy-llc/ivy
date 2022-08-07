import requests
import json
import pandas as pd

url = "https://api.github.com/repos/unifyai/ivy/actions/runs?actor=Aarsh2001&branch=postman3"


def get_api_results(url, headers, payload):
    response = requests.request("GET", url, headers=headers, data=payload)
    return json.loads(response.text)


payload = {}
headers = {
    "Accept": "application/vnd.github+json",
    "branch": "postman3",
    "Authorization": "Bearer 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",
}

output = get_api_results(url, headers, payload)
some = []

for info in output["workflow_runs"]:
    if info["name"] == "lint":
        continue
    some.append(
        (info["id"], info["name"], info["jobs_url"], info["status"], info["created_at"])
    )

df = pd.DataFrame(some, columns=["id", "name", "jobs_url", "status", "created_at"])
df["created_at"] = pd.to_datetime(df["created_at"])
v = (pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.total_seconds() / 3600
df = df[v < 4]
print(df)

functional_core_list = []
functional_nn_list = []
stateful_list = []
"""
for name, jobs_url in zip(df["name"], df["jobs_url"]):
    for info in get_api_results(jobs_url, headers, payload)["jobs"]:
        backend = ""
        submodule = ""
        if info['name'].startswith("run-nightly-tests"):
            job_name = info['name']
            backend = job_name.strip("run-nightly-tests")[2:-1].split(",")[0]
            submodule = job_name.strip("run-nightly-tests")[2:-1].split(",")[1]
        else:
            backend = info["name"]
            submodule = info["name"]

        if name == "test-core-ivy":
            functional_core_list.append((backend, submodule, info["conclusion"], info['completed_at']))
        elif name == "test-functional-ivy":
            functional_nn_list.append((backend, submodule, info["conclusion"], info['completed_at']))
        else:
            stateful_list.append((backend, submodule, info["conclusion"], info["completed_at"]))

data = pd.DataFrame(functional_core_list, columns=["backend", "submodule", "conclusion", "completed_at"])
data2 = pd.DataFrame(functional_nn_list, columns=["backend", "submodule", "conclusion", "completed_at"])
data3 = pd.DataFrame(stateful_list, columns=["backend", "submodule", "conclusion", "completed_at"])

print(data)
"""
