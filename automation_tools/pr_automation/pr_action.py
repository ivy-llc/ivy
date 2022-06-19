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
    
    # If a volunteer has opened a PR then assign it's coresponding Ivy Team Member
    for ivy_intern, assigned_volunteers in interns_assigned_volunteers.items():
        if pr.author() in assigned_volunteers:
            pr.assign_intern(ivy_intern)
            print(f"[+] {ivy_intern} was assigned to PR {pr.pr_number()}")
            sys.exit(0)
    
    # Get all PRs, count how many times each Ivy Team Member has been assigned
    all_prs = pr.command(f'gh pr list --json assignees')
    all_names = [i['assignees'][0]['login'] for i in all_prs]
    unique_names = set(all_names)
    count = {}
    for i in unique_names:
        count[i] = int(all_names.count(i) * 100 / len(all_names))
    
    # Assigning an intern evenly based on it's percentage. The lower the percentage, the bigger the chance to get assigned.
    # This way the workload is spread evenly.
    max_percentage = max(count.values())
    interns = [name for name, percentage in count.items() if percentage < max_percentage]
    pr.assign_random_intern(interns)


if __name__ == "__main__":
    main()
