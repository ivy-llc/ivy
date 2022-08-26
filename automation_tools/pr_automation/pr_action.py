import sys
import json
from github import Github
from process_pr import Process_pr


def import_file(file_path):
    with open(file_path, "r") as file:
        data = json.loads(file.read())
    return data


def main():
    pr = Process_pr(int(sys.argv[1]), sys.argv[2])
    token = str(sys.argv[3])
    g = Github(token)
    repo = g.get_repo("unifyai/ivy")
    
    interns_assigned_volunteers = repo.get_contents("/automation_tools/pr_automation/assets/volunteer_go_to_intern.json", ref="automations")
    interns_assigned_volunteers = json.loads(interns_assigned_volunteers.decoded_content.decode('utf-8'))
    intern_points_of_contact = repo.get_contents("/automation_tools/pr_automation/assets/intern_poc.json", ref="automations")
    intern_points_of_contact = json.loads(intern_points_of_contact.decoded_content.decode('utf-8'))
    intern_assign_rates = repo.get_contents("/automation_tools/pr_automation/assets/intern_assign_rate.json", ref="automations")
    intern_assign_rates = json.loads(intern_assign_rates.decoded_content.decode('utf-8'))
    
    interns_pocs = intern_points_of_contact.keys()
    interns = intern_points_of_contact.values()

    volunteer_pocs = interns_assigned_volunteers.keys()
    volunteers = interns_assigned_volunteers.values()
    
    for volunteer, volunteer_poc in zip(volunteers, volunteer_pocs):
        if pr.author() in volunteer:
            pr.assign_intern(volunteer_poc)
            sys.exit(0)
    for intern, interns_poc in zip(interns, interns_pocs):
        if pr.author() in intern:
            pr.assign_intern(interns_poc)
            sys.exit(0)

    pr.assign_random_intern(intern_assign_rates)


if __name__ == "__main__":
    main()
