import os
import re
import json
from github import Github


class Process_issue:
    def __init__(
        self, issue_number, comment_number, comment_body, comment_author, token
    ):
        self._token = token
        self._issue_number = issue_number
        self._comment_body = comment_body
        self._comment_author = comment_author
        self._comment_number = comment_number
        self._issue = self.command(
            f"gh issue view {self._issue_number} --json title,body,labels"
        )

    def get_issue_number(self):
        return self._issue_number

    def get_issue_body(self):
        return self._issue["body"]

    def command(self, cmd, save_output=True):
        try:
            if save_output:
                return json.loads(os.popen(cmd).read())
            else:
                os.system(cmd)
        except json.decoder.JSONDecodeError:
            print("Issue doesn't exist. Exiting process!")
            self.delete_comment(self._token, self._issue_number, self._comment_number)
            exit()

    def delete_comment(self):
        g = Github(self._token)
        repo = g.get_repo("unifyai/contributor-automations")
        issue_comment = repo.get_issue(self._issue_number).get_comment(
            self._comment_number
        )
        issue_comment.delete()
        print("Comment deleted!")

    def child_issue_id_in_comment(self):
        try:
            issue_id = re.search(r"[#/]\d+", self._comment_body)
            return True, issue_id.group(0)
        except AttributeError:
            return False, None

    def child_issue_title(self, child_issue_id):
        x = self.command(f"gh issue view {int(child_issue_id[1:])} --json title")
        return x["title"].strip()

    def get_labels(self):
        return [label["name"] for label in self._issue["labels"]]

    def main_issue_body_functions(self, allocated=True):
        if allocated:
            return [int(i[1:]) for i in re.findall(r"[#/]\d+", self._issue["body"])]
        return [i.strip() for i in re.findall(r"\b\w.+\b", self._issue["body"])]
