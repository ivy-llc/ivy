import pickle
from pydriller import Repository
import os


if __name__ == "__main__":
    with open("tests.pkl", "rb") as f:
        tests = pickle.load(f)
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
            file_name = file.new_path + ",cover"
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
                tests_to_run.update(tests_file[line])
            for line in sorted(deleted, reverse=True):
                if line < len(tests_file):
                    del tests_file[line]
            for line in added:
                top = -1
                bottom = -1
                if line - 1 < len(tests_file):
                    top = tests_file[line - 1]
                if line + 1 < len(tests_file):
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
                tests_to_run.update(tests_file[line])
            for line in added:
                tests_to_run.update(tests_file[line])
        break

    # with open("tests_to_run", "w") as f:
    #     for test in tests_to_run:
    #         f.write(test + "\n")

    # Run Tests
    failed = False
    for test in tests_to_run:
        ret = os.system(
            f'docker run --rm -it -v "$(pwd)":/ivy unifyai/ivy:latest python3 -m pytest {test}'  # noqa
        )
        if ret != 0:
            failed = True

    if failed:
        exit(1)
    # print(tests_to_run)
    # Output Tests to a File
