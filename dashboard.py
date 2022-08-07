import requests
import json
import pandas as pd


def get_results(url):
    response = requests.get(url)
    return json.loads(response.text)


output = get_results(
    "https://api.github.com/repos/unifyai/ivy/actions/runs?branch=postman3"
)
some = []

for info in output["workflow_runs"]:
    if info["name"] == "lint":
        continue
    some.append((info["id"], info["name"], info["jobs_url"], info["created_at"]))

df = pd.DataFrame(some, columns=["id", "name", "jobs_url", "created_at"])
df["created_at"] = pd.to_datetime(df["created_at"])
v = (pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.total_seconds() / 3600
df = df[v < 2.5]
functional_core_list = []
functional_nn_list = []
stateful_list = []
for name, jobs_url in zip(df["name"], df["jobs_url"]):
    for info in get_results(jobs_url)["jobs"]:
        if name == "test-core-ivy":
            functional_core_list.append((info["name"], info["conclusion"]))
        elif name == "test-nn-ivy":
            functional_nn_list.append((info["name"], info["conclusion"]))
        elif name == "test-stateful-ivy":
            stateful_list.append((info["name"], info["conclusion"]))

data = pd.DataFrame(functional_core_list, columns=["job_name", "conclusion"])
data2 = pd.DataFrame(functional_nn_list, columns=["job_name", "conclusion"])
data3 = pd.DataFrame(stateful_list, columns=["job_name", "conclusion"])

print(data)
print(data2)
print(data3)
