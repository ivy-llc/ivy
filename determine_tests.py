import pickle  # noqa
from pydriller import Repository
import os  # noqa
import bz2
import _pickle as cPickle

MAX_TESTS = 2
if __name__ == "__main__":
    tests = bz2.BZ2File("tests.pbz2", "rb")
    tests = cPickle.load(tests)
    tests_to_run = set()
    ref_commit_hash = tests["commit"]
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
            # Now Update the Tests and compute the tests to run
            for line in deleted:
                tests_file_line = tests_file[line]
                if len(tests_file_line) >= MAX_TESTS:
                    continue
                tests_to_run.update(tests_file_line)
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
                tests_file_line = tests_file[line]
                if len(tests_file_line) >= MAX_TESTS:
                    continue
                tests_to_run.update(tests_file_line)
            for line in added:
                tests_file_line = tests_file[line]
                if len(tests_file_line) >= MAX_TESTS:
                    continue
                tests_to_run.update(tests_file_line)
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
