import pickle  # noqa
from pydriller import Repository
import os  # noqa
import bz2
import _pickle as cPickle

BACKENDS = ["numpy", "jax", "tensorflow", "torch"]


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


MAX_TESTS = 10
if __name__ == "__main__":
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
                file_name = f"{file.new_path},cover"
            except Exception:
                continue
            if file_name not in tests.keys():
                continue
            tests_file = tests[file_name]
            change = file.diff_parsed
            added = {x - 1 for (x, _) in change["added"]}
            deleted = {x - 1 for (x, _) in change["deleted"]}
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
