import sys
import json
from process_pr import Process_pr


def import_file(file_path):
    with open(file_path, "r") as file:
        data = json.loads(file.read())
    return data


def main():
    pr = Process_pr(int(sys.argv[1]), sys.argv[2])
    interns_assigned_volunteers = import_file("assets/volunteer_go_to_intern.json")
    intern_points_of_contact = import_file("assets/intern_poc.json")
    intern_assign_rates = import_file("assets/intern_assign_rate.json")
    
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
