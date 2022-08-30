import json
import subprocess
import sys

before_sha, after_sha = sys.argv[-2:]

output = {}

with open('name-changed') as f:
    for diff_filename in f.readlines():
        diff_filename = diff_filename.strip()
        # TODO: works for PR, but not for multiple commits of a single push (latter not allowed BTW)
        diff_command = f"git --no-pager diff '{before_sha}..{after_sha}' --no-color --unified=0 -- {diff_filename} | grep -Po '^\+\+\+ ./\K.*|^@@ -[0-9]+(,[0-9]+)? \+\K[0-9]+(,[0-9]+)?(?= @@)' "
        try:
            diff_ret = subprocess.check_output(
                diff_command, shell=True, stderr=subprocess.STDOUT).decode('utf-8')
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output))

        output[diff_filename] = diff_ret

print(output)
json.dump(output, open('name-changed.json', 'w'))
