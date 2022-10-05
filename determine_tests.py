import pickle

from pydriller import Repository


if __name__ == "__main__":
    with open("tests.pkl", "rb") as f:
        tests = pickle.load(f)

    # pprint(tests)
    for commit in Repository(".", order="reverse").traverse_commits():
        for file in commit.modified_files:
            file_name = file.new_path + ",cover"
            if file_name not in tests.keys():
                continue
            tests_file = tests[file_name]
            # pprint(tests_file)
            change = file.diff_parsed
            added = set([x - 1 for (x, _) in change["added"]])
            deleted = set([x - 1 for (x, _) in change["deleted"]])
            updated = added.intersection(deleted)
            added = added.difference(updated)
            deleted = deleted.difference(updated)
            # Now Update the Tests and compute the tests to run
            tests_to_run = set()
            for line in deleted:
                tests_to_run.update(tests_file[line])
            for line in deleted:
                del tests_file[line]
            for line in added:
                top = tests_file.get(line - 1, -1)
                bottom = tests_file.get(line + 1, -1)
                if top != -1 and bottom != -1:
                    tests_line = top.intersection(bottom)
                elif top != -1:
                    tests_line = top
                else:
                    tests_line = bottom
                tests_file.insert(line, tests_file)
            tests[file_name] = tests_file
            # Now Compute the Tests to Run
            for line in updated:
                tests_to_run.update(tests_file[line])
            for line in added:
                tests_to_run.update(tests_file[line])
            print(tests_to_run)
        break
