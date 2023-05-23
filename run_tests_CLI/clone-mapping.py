import os
import git
import bz2
import _pickle as cPickle

# The path to your Mapping directory
mapping_dir = "Mapping/"

# Check if the directory exists
if not os.path.exists(mapping_dir):
    print(f"Directory does not exist: {mapping_dir}")
    exit(1)

# Create a Repo object to interact with the Git repositories
current_repo = git.Repo("ivy/")
mapping_repo = git.Repo(mapping_dir)

# Get the commit history of the current repository (limit to top 100 commits)
current_repo_commits = [
    commit.hexsha for commit in current_repo.iter_commits(max_count=100)
]

# The path to the tests.pbz2 file
test_file_path = os.path.join(mapping_dir, "tests.pbz2")

# Go back in the history of the Mapping repository
for commit in mapping_repo.iter_commits():
    print(commit.hexsha)
    try:
        mapping_repo.git.checkout(commit)
    except git.GitCommandError as e:
        print(f"Error checking out commit: {commit.hexsha}\n{str(e)}")
        continue

    # Check if the tests.pbz2 file exists
    if not os.path.isfile(test_file_path):
        print(f"File does not exist in this commit: {commit.hexsha}")
        continue

    # Unpickle the file
    tests = bz2.BZ2File(test_file_path, "rb")
    tests = cPickle.load(tests)

    # Get the commit hash
    commit_hash = tests.get("commit")
    print("Commit:", commit_hash)

    if commit_hash is None:
        print("Commit hash not found in the test dictionary.")
        continue

    # Check if the commit hash exists in the current repository's history
    if commit_hash in current_repo_commits:
        print(f"Found matching commit hash in current repository: {commit_hash}")
        break
