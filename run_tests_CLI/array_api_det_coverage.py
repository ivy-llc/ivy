import os
import subprocess
from pydriller import Repository
from tqdm import tqdm
import bz2
import _pickle as cPickle


def main():
    BACKENDS = ["numpy", "jax", "tensorflow", "torch"]

    test_names = []
    func_folder = "ivy_tests/array_api_testing/array_api_methods_to_test"
    func_fnames = os.listdir(func_folder)
    func_fnames.sort()
    framework_tests_to_run = {
        "jax": list(),
        "numpy": list(),
        "torch": list(),
        "tensorflow": list(),
    }

    # add from each filepath
    for fname in func_fnames:
        fpath = os.path.join(func_folder, fname)
        with open(fpath, "r") as file:
            contents = file.read()
            contents = [line.replace("__", "") for line in contents.split("\n")]
            for framework in framework_tests_to_run:
                tests_to_run = list()
                for s in contents:
                    if s == "":
                        continue
                    if ("#" not in s) or (
                        "#" in s
                        and not (framework in s.lower())
                        and any(f in s.lower() for f in framework_tests_to_run)
                    ):
                        submod = f"ivy_tests/array_api_testing/test_array_api/array_api_tests/test_{fname.replace('.txt', '.py')}"  # noqa

                        test_name = (
                            submod
                            + "::test_"
                            + (s if ("#" not in s) else s.split("#")[1].split(" ")[0])
                        )
                        tests_to_run += [test_name]
                framework_tests_to_run[framework] += tests_to_run

    for backend, tests in framework_tests_to_run.items():
        test_names += [test + "," + backend for test in set(tests)]

    # Create a Dictionary of Test Names to Index
    tests = {"index_mapping": test_names, "tests_mapping": {}}
    for i in range(len(test_names)):
        tests["tests_mapping"][test_names[i]] = i

    # Create k flag files for each backend:
    k_flag = {}
    subprocess.run(
        ["python3", "ivy_tests/array_api_testing/write_array_api_tests_k_flag.py"],
        check=True,
    )
    for backend in BACKENDS:
        k_flag_file = f"ivy_tests/array_api_testing/.array_api_tests_k_flag_{backend}"
        with open(k_flag_file, "r") as f:
            array_api_tests_k_flag = f.read().strip()

        if backend == "torch":
            array_api_tests_k_flag += " and not (uint16 or uint32 or uint64)"
        k_flag[backend] = array_api_tests_k_flag
    directories = (
        [x[0] for x in os.walk("ivy")]
        + [x[0] for x in os.walk("ivy_tests/array_api_testing")]
        + ["ivy_tests"]
    )
    directories_filtered = [
        x for x in directories if not (x.endswith("__pycache__") or "hypothesis" in x)
    ]
    directories = set(directories_filtered)
    for test_backend in tqdm(test_names):
        test_name, backend = test_backend.split(",")
        command = f'docker run --rm --env IVY_BACKEND={backend} --env ARRAY_API_TESTS_MODULE="ivy" -v "$(pwd)":/ivy unifyai/ivy:latest timeout 30m /bin/bash -c "coverage run --source=ivy,ivy_tests -m pytest {test_name} -k \\"{k_flag[backend]}\\" --disable-warnings --tb=short -vv > coverage_output;coverage annotate > coverage_output" '  # noqa
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


if __name__ == "__main__":
    main()
