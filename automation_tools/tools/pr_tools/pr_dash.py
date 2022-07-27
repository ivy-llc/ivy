import os
import sys
import json
import datetime
import subprocess
from sys import argv, platform
from prettytable import PrettyTable

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(
    'macro-precinct-357609-220191a202ca.json', scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
sheet = client.open("Github").sheet1
# flake8: noqa


def set_platform():
    if "linux" in platform:
        return True
    elif "win" in platform:
        return False
    return False


def set_path():
    cwd = os.getcwd()
    if cwd.endswith("pr_tools"):
        os.chdir("../../../")
        sys.path.insert(0, os.getcwd())
    elif not cwd.endswith("ivy"):
        print("[-] You have to run the script from within ivy home directory.")


def diff_between_2_dates(d1, d2=datetime.datetime.now().date(), dformat="%Y-%m-%d"):
    if not isinstance(d1, datetime.date):
        d1 = datetime.datetime.strptime(d1, dformat).date()
    if type(d2) == str:
        d2 = datetime.datetime.strptime(d2, dformat).date()
    return abs((d2 - d1).days)


def get_file_contents(path, mode="r", read_type=None):
    with open(path, mode) as file:
        if read_type == "readlines":
            return file.readlines()
        elif read_type == "read":
            return file.read()
        elif not read_type:
            print("[-] 'read_type' can not be None.")
        else:
            print(f"[-] Invalid read_type: '{read_type}'")


def create_rows(prs):
    tmp = []
    ignore_authors = [
        auth.replace("\n", "")
        for auth in get_file_contents(
            "automation_tools/tools/pr_tools/assets/exclude.txt", read_type="readlines"
        )
    ]
    nr = 1
    for pr in prs:
        # Check for last comment or reviews.
        diff = 0
        last_review = pr["latestReviews"]
        last_comment = pr["comments"]
        last_update = pr["updatedAt"][:-10]
        pr_author = pr["author"]["login"]

        if pr_author not in ignore_authors:
            # Check if the PR has both a review and comments and check which was the last one.
            if last_review and last_comment:
                review_submit_date = datetime.datetime.strptime(
                    last_review[-1]["submittedAt"][:-10], "%Y-%m-%d"
                ).date()
                comment_submit_date = datetime.datetime.strptime(
                    last_comment[-1]["createdAt"][:-10], "%Y-%m-%d"
                ).date()
                comment_author = last_comment[-1]["author"]["login"]

                # If the comment was the last update on the PR and the comment author is not one of Ivy
                # team members then calculate the inactivity days
                if (
                    comment_submit_date > review_submit_date
                    and comment_author not in ignore_authors
                ):
                    diff = diff_between_2_dates(comment_submit_date)
                elif (
                    comment_submit_date < review_submit_date
                    and comment_author not in ignore_authors
                ):
                    diff = diff_between_2_dates(review_submit_date)
            # If the last update on the PR is a comment and the comment author is not a intern calculate the inactivity days
            elif not last_review and last_comment:
                comment_author = last_comment[-1]["author"]["login"]
                if comment_author not in ignore_authors:
                    comment_submit_date = datetime.datetime.strptime(
                        last_comment[-1]["createdAt"][:-10], "%Y-%m-%d"
                    ).date()
                    diff = diff_between_2_dates(comment_submit_date)
            # If there are no comments or reviews on the PR, calculate the inactivity days based on the last PR update.
            elif not last_review and not last_comment:
                diff = diff_between_2_dates(last_update)

        if diff >= 3:
            row = [pr["title"].strip(), diff, pr["url"], pr["author"]["login"], "-"]
            if pr["assignees"]:
                row[-1] = pr["assignees"][0]["login"]
            tmp.append(truncate_pr_title(row))
    tmp.sort(reverse=True, key=lambda diff: diff[1])

    nr = 1
    for row in tmp:
        row.insert(0, nr)
        nr += 1
    return tmp


def truncate_pr_title(row):
    if len(row[0]) >= 30:
        row[0] = row[0][:30] + "..."
    return row


def sort_prs(rows, table, criteria=None, r_index=None):
    nr = 1
    for row in rows:
        if criteria == row[r_index]:
            row[0] = nr
            table.add_row(row)
            nr += 1


def command(cmd, save_output=True):
    set_path()
    try:
        if save_output:
            return json.loads(
                subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, shell=set_platform()
                ).communicate()[0]
            )
        else:
            subprocess.run(cmd, shell=set_platform())
    except json.decoder.JSONDecodeError as e:
        print(e)
        exit()


def help_menu():
    help_table = PrettyTable()
    help_table.field_names = ["Command", "Description"]
    cmds_description = [
        ["-h", "This help menu"],
        ["-au", "Sorts PRs by author by providing a name"],
        ["-as", "Sorts PRs by assigners by providing a name"],
        [" -all", "Provides a table of all PRs that are inactive for more than 3 days"],
    ]

    for cmd_description in cmds_description:
        help_table.add_row(cmd_description)
    print(help_table)
    exit()


def main():
    try:
        table = PrettyTable()
        table.field_names = [
            "Nr",
            "Title",
            "Inactivity Days",
            "URL",
            "Author",
            "Assignee",
        ]

        r_index = None
        if argv[1] == "-au":
            r_index = 4
        elif argv[1] == "-as":
            r_index = 5
        elif argv[1] == "-all":
            prs = command(
                "gh pr list -L 200 --json title,url,updatedAt,assignees,latestReviews,comments,author"  # noqa
            )
            rows = create_rows(prs)
            for row in rows:
                sheet.append_rows(row)
                table.add_row(row)
        elif argv[1] == "-h":
            help_menu()

        if r_index:
            prs = command(
                "gh pr list -L 200 --json title,url,updatedAt,assignees,latestReviews,comments,author"  # noqa
            )
            rows = create_rows(prs)

            sort_prs(rows, table, argv[2], r_index)
        print(table)
    except IndexError as e:
        help_menu()


if __name__ == "__main__":
    main()
