import os
import sys
import json
import datetime
import subprocess
from sys import argv, platform
from prettytable import PrettyTable


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
        print("[-] You have to run the script form within ivy home directory.")


def diff_between_2_dates(d1, d2=datetime.datetime.now().date(), dformat="%Y-%m-%d"):
    d1 = datetime.datetime.strptime(d1, dformat).date()
    if type(d2) == str:
        d2 = datetime.datetime.strptime(d2, dformat).date()
    return abs((d2 - d1).days)


def create_rows(prs):
    tmp = []
    for pr in prs:
        try:
            latestReviews = pr["latestReviews"][0]["submittedAt"][:-10]
            updatedAt = pr["updatedAt"][:-10]
            if diff_between_2_dates(latestReviews) > diff_between_2_dates(updatedAt):
                diff = diff_between_2_dates(updatedAt)
            else:
                diff = diff_between_2_dates(latestReviews)
        except IndexError:
            diff = diff_between_2_dates(pr["updatedAt"][:-10])

        if diff >= 3:
            row = [pr["title"].strip(), diff, pr["url"]]
            if pr["latestReviews"] and pr["assignees"]:
                row += [
                    pr["assignees"][0]["login"],
                    pr["latestReviews"][0]["author"]["login"],
                ]
            elif not pr["latestReviews"] and pr["assignees"]:
                row += [pr["assignees"][0]["login"], "-"]
            elif pr["latestReviews"] and not pr["assignees"]:
                row += ["-", pr["latestReviews"][0]["author"]["login"]]
            elif not pr["latestReviews"] and not pr["assignees"]:
                row += ["-", "-"]
            tmp.append(truncate_pr_title(row))
    tmp.sort(reverse=True, key=lambda diff: diff[1])
    return tmp


def truncate_pr_title(row):
    if len(row[0]) >= 30:
        row[0] = row[0][:30] + "..."
    return row


def sort_prs(rows, table, criteria=None, r_index=None):
    for row in rows:
        if criteria == row[r_index]:
            table.add_row(row)


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
        ["-a", "Sorts PRs by assigners by providing a name"],
        ["-lr", "Sorts PRs by last reviewers by providing a name"],
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
            "Title",
            "Inactivity Days",
            "URL",
            "Assignee",
            "Last Reviewer",
        ]

        r_index = None
        if argv[1] == "-a":
            r_index = 3
        elif argv[1] == "-lr":
            r_index = -1
        elif argv[1] == "-all":
            prs = command(
                "gh pr list -L 100 --json title,url,updatedAt,assignees,latestReviews"
            )
            rows = create_rows(prs)
            for row in rows:
                table.add_row(row)
        elif argv[1] == "-h":
            help_menu()

        if r_index:
            prs = command(
                "gh pr list -L 100 --json title,url,updatedAt,assignees,latestReviews"
            )
            rows = create_rows(prs)

            sort_prs(rows, table, argv[2], r_index)
        print(table)
    except IndexError:
        help_menu()


if __name__ == "__main__":
    main()
