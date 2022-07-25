import os
import json
import random as rn


class Process_pr:
    def __init__(self, pr_number, pr_author):
        self.__pr_number = pr_number
        self.__pr_author = pr_author

    def author(self):
        return self.__pr_author

    def pr_number(self):
        return self.__pr_number

    def command(self, cmd, save_output=True):
        try:
            if save_output:
                return json.loads(os.popen(cmd).read())
            else:
                os.system(cmd)
        except json.decoder.JSONDecodeError:
            print("PR doesn't exist. Exiting process!")
            exit()

    def assign_intern(self, ivy_intern):
        # --add-reviewer "{ivy_intern}"
        # Need to find a way how to overcome the permissions for GH Actions
        self.command(
            f'gh pr edit {self.pr_number()} --add-assignee "{ivy_intern}"',
            save_output=False,
        )
        print(f"[+] {ivy_intern} was assigned to PR {self.pr_number()}")

    def assign_random_intern(self, intern_list):
        gh_ids = list(intern_list.keys())
        weights = tuple(intern_list.values())
        random_intern = rn.choices(gh_ids, weights=weights, k=1)[0]
        # --add-reviewer "{random_intern}"
        # Need to find a way how to overcome the permissions for GH Actions
        self.command(
            f'gh pr edit {self.pr_number()} --add-assignee "{random_intern}"',
            save_output=False,
        )
        print(f"[+] {random_intern} was assigned to PR {self.pr_number()}")
