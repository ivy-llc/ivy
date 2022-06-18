import sys
import json
from process_pr import Process_pr


def import_file(file_path):
    with open(file_path, "r") as file:
        data = json.loads(file.read())
    return data


def main():
    pr = Process_pr(int(sys.argv[1]), sys.argv[2])
    interns_assigned_volunteers = import_file("volunteer_go_to_intern.json")
    interns = list(interns_assigned_volunteers.keys())
    for ivy_intern, assigned_volunteers in interns_assigned_volunteers.items():
        if pr.author() in assigned_volunteers:
            pr.assign_intern(ivy_intern)
            print(f"[+] {ivy_intern} was assigned to PR {pr.pr_number()}")
            sys.exit(0)
    pr.assign_random_intern(interns)


if __name__ == "__main__":
    main()
