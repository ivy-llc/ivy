from github import Github
import ast
import base64
import requests
import sys

def check_function_names(file_content, prefix):
    tree = ast.parse(file_content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if not node.name.startswith(prefix):
                return False
    return True

def enforce_test_naming_convention(token):
    acc = Github(token)
    repo = acc.get_repo("unifyai/ivy")

    pulls = repo.get_pulls(state="open", sort="created", direction="desc")

    for pull in pulls:
        files = pull.get_files()
        for file in files:
            if file.filename.startswith("ivy_tests/test_ivy/test_frontends"):
                frontend_name = file.filename.split('/')[-2]  
                expected_prefix = "test_" + frontend_name
                raw_url = file.raw_url
                response = requests.get(raw_url)
                if response.status_code == 200:
                    file_content = response.text
                    is_convention_respected = check_function_names(file_content, expected_prefix)
                    if not is_convention_respected:
                        print(f"Naming convention error in PR: {pull.title}. In file {file.filename} there are functions that do not start with {expected_prefix}")
                        pull.create_review(
                            commit=repo.get_commit(sha=pull.head.sha),
                            body=f":x: Naming convention error: In file {file.filename} there are functions that do not start with {expected_prefix}. Please correct the naming.",
                            event="REQUEST_CHANGES"
                        )

if __name__ == "__main__":
    TOKEN = sys.argv[1]
    enforce_test_naming_convention(TOKEN)
