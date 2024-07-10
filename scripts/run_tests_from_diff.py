"""Script that attempts to find the test file(s) corresponding the all the
changes made in a commit (git diff stored in commit-diff.txt), and runs all the
tests it finds."""

import os
import re
import subprocess
import sys


with open("commit-diff.txt", "r") as f:
    diff_lines = f.readlines()

modified_files = set()
for line in diff_lines:
    if line.startswith("diff --git a/"):
        file_path = line.split(" ")[2].strip().lstrip("a/")
        modified_files.add(file_path)
    elif line.startswith("diff --git b/"):
        file_path = line.split(" ")[2].strip().lstrip("b/")
        modified_files.add(file_path)
    elif line.startswith("diff --git "):
        file_path = line.split(" ")[2].strip().lstrip("--git ")
        modified_files.add(file_path)

nn = [
    "activations",
    "layers",
    "losses",
    "norms",
]
exclude = [
    "conftest",
]
test_paths = []

for file_path in set(modified_files):
    try:
        suffix = file_path.rsplit("/", 1)[1].replace(".py", "")
    except IndexError:
        continue
    if suffix in exclude:
        continue

    # some regex logic to get the correct test file for the given modified file
    if "/frontends/" in file_path:
        file_path = file_path.replace("/functional", "")

    if "/backends/" in file_path:
        file_path = re.sub(r"/backends/.*?/", "/", file_path)
        file_path = file_path.rsplit("/", 1)[0]
        if suffix in nn:
            file_path += f"/nn/{suffix}.py"
        else:
            file_path += f"/core/{suffix}.py"

    if not file_path.startswith("ivy_tests/"):
        path_list = file_path.split("/")
        path_list = ["test_" + p for p in path_list]
        file_path = "ivy_tests/" + "/".join(path_list)

    # if the test file doesn't exist, step up a directory
    # and run those tests instead (if that exists)
    if not os.path.exists(file_path):
        file_path = file_path.rsplit("/", 1)[0]

    if os.path.exists(file_path) and "ivy_tests/test_ivy/" in file_path:
        test_paths.append(file_path)

test_paths = set(test_paths)
print("Running tests:", test_paths)

for test_path in test_paths:
    pytest_command = (
        f"pytest {test_path} -p no:warnings --tb=short --backend jax,tensorflow,torch"
    )
    print(f"Running test command: {pytest_command}")
    result = subprocess.run(pytest_command, shell=True)

    if result.returncode != 0:
        sys.exit(result.returncode)
