# flake8: noqa
import re
import sys
import json
from process_issue import Process_issue


def load_labels():
    with open("issue_labels.json", "r") as file:
        return json.load(file)


def targeted_labels_in_main_issue(accepted_labels, main_issue_labels):
    for key, labels in accepted_labels["Determine_Labels"].items():
        if labels == main_issue_labels:
            return key
    return False


def set_child_issue_labels(key, accepted_labels):
    return accepted_labels["Set_Labels"][key]


def main():

    main_issue = Process_issue(
        int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[-1]
    )
    main_issue_labels = main_issue.get_labels()
    issue_in_cmt, issue_id = main_issue.child_issue_id_in_comment()
    alocate_functions = main_issue.main_issue_body_functions(allocated=True)
    non_alocate_functions = main_issue.main_issue_body_functions(allocated=False)

    accepted_labels = load_labels()
    targeted_labels = targeted_labels_in_main_issue(accepted_labels, main_issue_labels)

    if targeted_labels:
        if issue_in_cmt:
            comment_issue_id = int(issue_id[1:])
            comment_issue_title = main_issue.child_issue_title(issue_id)
            if comment_issue_title in non_alocate_functions:
                print("Function Free")
                # ToDo: Add Labels "Array API" "Single Function"
                main_issue_body = re.sub(
                    r"\b%s\b" % comment_issue_title,
                    issue_id.replace("/", "#"),
                    main_issue.get_issue_body().replace("_", "\_"),
                )
                child_issue_labels = set_child_issue_labels(
                    targeted_labels, accepted_labels
                )
                main_issue.command(
                    f'gh issue edit {comment_issue_id} --add-label "{child_issue_labels[0]}","{child_issue_labels[1]}"',
                    save_output=False,
                )
                main_issue.command(
                    f'gh issue edit {main_issue.get_issue_number()} --body "{main_issue_body}"',
                    save_output=False,
                )
                main_issue.delete_comment()
            elif (comment_issue_title not in non_alocate_functions) and (
                comment_issue_id not in alocate_functions
            ):
                print("Function already allocated, closing issue.")
                main_issue.command(
                    f'gh issue comment {comment_issue_id} --body "This issue is being closed because the function has already been taken, please choose another function and recommend on the main issue."',
                    save_output=False,
                )
                main_issue.command(
                    f"gh issue close {comment_issue_id}", save_output=False
                )
                main_issue.delete_comment()
            elif comment_issue_id in alocate_functions:
                print("Issue ID already in use...")
                main_issue.delete_comment()
        else:
            # ToDo: Delete comment
            print("Deleting comment! No issue found.")
            main_issue.delete_comment()
    else:
        print("Skipping step")


if __name__ == "__main__":
    main()

# main_issue_ids = main_issue_numbers(command('gh issue list --label "ToDo" --json number'))

# if issue_number in main_issue_ids:
