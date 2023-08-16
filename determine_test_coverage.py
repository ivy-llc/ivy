import os
import sys
from pydriller import Repository
import pickle  # noqa
from tqdm import tqdm
import bz2
import _pickle as cPickle
from run_tests_CLI.get_all_tests import get_all_tests


# Shared Map
tests = {}

N = 128
run_iter = int(sys.argv[1]) - 1

test_names = get_all_tests()

# Create a Dictionary of Test Names to Index
tests["index_mapping"] = test_names
tests["tests_mapping"] = {}
for i in range(len(test_names)):
    tests["tests_mapping"][test_names[i]] = i

if __name__ == "__main__":
    directories = (
        [x[0] for x in os.walk("ivy")]
        + [x[0] for x in os.walk("ivy_tests/test_ivy")]
        + ["ivy_tests"]
    )
    directories_filtered = [
        x for x in directories if not (x.endswith("__pycache__") or "hypothesis" in x)
    ]
    directories = set(directories_filtered)
    num_tests = len(test_names)
    tests_per_run = num_tests // N
    start = run_iter * tests_per_run
    end = num_tests if run_iter == N - 1 else (run_iter + 1) * tests_per_run
    for test_backend in tqdm(test_names[start:end]):
        test_name, backend = test_backend.split(",")
        command = (
            f'docker run -v "$(pwd)":/ivy unifyai/ivy:latest timeout 30m /bin/bash -c "coverage run --source=ivy,'  # noqa
            f"ivy_tests -m pytest {test_name} --backend {backend} --disable-warnings > coverage_output;coverage "  # noqa
            f'annotate > coverage_output" '
        )
        os.system(command)
        for directory in directories:
            for file_name in os.listdir(directory):
                if file_name.endswith("cover"):
                    file_name = directory + "/" + file_name
                    if file_name not in tests:
                        tests[file_name] = []
                        with open(file_name) as f:
                            for line in f:
                                tests[file_name].append(set())
                    with open(file_name) as f:
                        i = 0
                        for line in f:
                            if line[0] == ">":
                                tests[file_name][i].add(
                                    tests["tests_mapping"][test_backend]
                                )
                            i += 1
        os.system("find . -name \\*cover -type f -delete")

commit_hash = ""
for commit in Repository(".", order="reverse").traverse_commits():
    commit_hash = commit.hash
    break
tests["commit"] = commit_hash
with bz2.BZ2File("tests.pbz2", "w") as f:
    cPickle.dump(tests, f)
