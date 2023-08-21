import pickle  # noqa
import subprocess

from pydriller import Repository
import os  # noqa
import bz2
import _pickle as cPickle
import sys
from run_tests_CLI.get_all_tests import get_all_tests

MAX_TESTS = 10


def get_tests(_tests_file, _line):
    tests_file_line = set()
    if 0 <= _line < len(_tests_file):
        tests_file_line = _tests_file[_line]
    return set() if len(tests_file_line) >= MAX_TESTS else tests_file_line


def determine_tests_line(_tests_file, _line, _tests_to_run):
    tests_file_line = get_tests(_tests_file, _line)
    tests_file_prev = get_tests(_tests_file, _line - 1)
    tests_file_next = get_tests(_tests_file, _line + 1)
    _tests_to_run.update(tests_file_line)
    _tests_to_run.update(tests_file_prev)
    _tests_to_run.update(tests_file_next)
    return _tests_to_run


def main():
    tests = bz2.BZ2File("tests.pbz2", "rb")
    tests = cPickle.load(tests)
    ref_commit_hash = tests["commit"]
    print("Reference Commit: ", ref_commit_hash)
    tests_to_run = set()
    for commit in Repository(".", single=ref_commit_hash).traverse_commits():
        ref_commit = commit._c_object
        break

    for commit in Repository(".", order="reverse").traverse_commits():
        tests["commit"] = commit.hash
        diff_index = ref_commit.diff(commit._c_object, create_patch=True)
        modified_files = commit._parse_diff(diff_index)
        for file in modified_files:
            try:
                file_name = file.new_path + ",cover"
            except:  # noqa
                continue
            if file_name not in tests.keys():
                continue
            tests_file = tests[file_name]
            change = file.diff_parsed
            added = set([x - 1 for (x, _) in change["added"]])
            deleted = set([x - 1 for (x, _) in change["deleted"]])
            updated = added.intersection(deleted)
            added = added.difference(updated)
            deleted = deleted.difference(updated)
            # Now Update the Mapping and compute the tests to run
            for line in deleted:
                tests_to_run = determine_tests_line(tests_file, line, tests_to_run)
            for line in sorted(deleted, reverse=True):
                if line < len(tests_file):
                    del tests_file[line]
            for line in added:
                top = -1
                bottom = -1
                if 0 <= line - 1 < len(tests_file):
                    top = tests_file[line - 1]
                if 0 <= line + 1 < len(tests_file):
                    bottom = tests_file[line + 1]
                tests_line = set()
                if top != -1 and bottom != -1:
                    tests_line = top.intersection(bottom)
                elif top != -1:
                    tests_line = top
                elif bottom != -1:
                    tests_line = bottom
                tests_file.insert(line, tests_line)
            tests[file_name] = tests_file
            # Now Compute the Tests to Run
            for line in updated:
                tests_to_run = determine_tests_line(tests_file, line, tests_to_run)
            for line in added:
                tests_to_run = determine_tests_line(tests_file, line, tests_to_run)
        break

    if len(sys.argv) >= 2 and sys.argv[1] == "1":
        print("Checking for any new tests added!")
        new_tests = get_all_tests()
        print("Done!")
        # Check for any new tests present
        old_tests = tests["index_mapping"]
        added_tests = set(new_tests) - set(old_tests)
        removed_tests = set(old_tests) - set(new_tests)
        with open("tests_to_remove", "w") as f:
            for test in removed_tests:
                f.write(test + "\n")
        added_tests = list(added_tests)
        # if it is a PR, we must check that the tests added were in the files_changes
        if len(sys.argv) >= 3 and sys.argv[2] == "pr":
            relevant_added_tests = []
            subprocess.run(
                ["git", "remote", "add", "upstream", "https://github.com/unifyai/ivy"]
            )
            subprocess.run(["git", "fetch", "upstream"])
            lca_sha = subprocess.check_output(
                ["git", "merge-base", "HEAD", "upstream/master"]
            )
            lca_hash = lca_sha.decode().strip()
            for commit in Repository(".", single=lca_hash).traverse_commits():
                lca_commit = commit._c_object
                break
            for commit in Repository(".", order="reverse").traverse_commits():
                diff_index = lca_commit.diff(commit._c_object, create_patch=True)
                modified_files = commit._parse_diff(diff_index)
                break
            for test in added_tests:
                for file in modified_files:
                    if file.new_path.strip() in test:
                        relevant_added_tests.append(test)
                        break
            added_tests = relevant_added_tests
        else:
            if len(added_tests) > 50:
                added_tests = added_tests[:50]
        # Add these new_tests in the Mapping
        old_num_tests = len(old_tests)
        tests["index_mapping"] += added_tests
        new_tests = tests["index_mapping"]
        num_tests = len(new_tests)
        for i in range(old_num_tests, num_tests):
            tests["tests_mapping"][new_tests[i]] = i
        directories = (
            [x[0] for x in os.walk("ivy")]
            + [x[0] for x in os.walk("ivy_tests/test_ivy")]
            + ["ivy_tests"]
        )
        directories_filtered = [
            x
            for x in directories
            if not (x.endswith("__pycache__") or "hypothesis" in x)
        ]
        directories = set(directories_filtered)
        for test_backend in new_tests[old_num_tests:num_tests]:
            tests_to_run.add(tests["tests_mapping"][test_backend])
            if len(sys.argv) < 3:
                print("Computing Coverage:", test_backend)
                test_name, backend = test_backend.split(",")
                command = (
                    f'docker run -v "$(pwd)":/ivy unifyai/ivy:latest /bin/bash -c "coverage run --source=ivy,'  # noqa
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
                                    if i >= len(tests[file_name]):
                                        tests[file_name].append(set())
                                    if line[0] == ">":
                                        tests[file_name][i].add(
                                            tests["tests_mapping"][test_backend]
                                        )
                                    i += 1
                os.system("find . -name \\*cover -type f -delete")

    with bz2.BZ2File("tests.pbz2", "w") as f:
        cPickle.dump(tests, f)

    print("----- Determined Tests -----")
    print(len(tests_to_run))
    for test_index in tests_to_run:
        print(tests["index_mapping"][test_index])
    print("----------------------------")

    with open("tests_to_run", "w") as f:
        for test_index in tests_to_run:
            test = tests["index_mapping"][test_index]
            f.write(test + "\n")


if __name__ == "__main__":
    main()
