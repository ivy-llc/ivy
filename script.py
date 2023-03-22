import os

i = 0
def handle(path):
    # open file in write mode
    with open(path, "w") as file:
        file.writelines(['<html><head><meta http-equiv="refresh" content="0; url=https://lets-unify.ai/ivy/"></head></html>'])

for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".html"):
             handle(os.path.join(root, file))