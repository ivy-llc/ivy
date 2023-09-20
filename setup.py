# lint as: python3
# Copyright 2021 The Ivy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License..
# ==============================================================================
__version__ = None

import setuptools
from setuptools import setup
from pathlib import Path
from pip._vendor.packaging import tags
from urllib import request
import os
import json
import re


def _get_paths_from_binaries(binaries, root_dir=""):
    """Get all the paths from the binaries.json into a list."""
    paths = []
    if isinstance(binaries, str):
        return [os.path.join(root_dir, binaries)]
    elif isinstance(binaries, dict):
        for k, v in binaries.items():
            paths += _get_paths_from_binaries(v, os.path.join(root_dir, k))
    else:
        for i in binaries:
            paths += _get_paths_from_binaries(i, root_dir)
    return paths


def _strip(line):
    return line.split(" ")[0].split("#")[0].split(",")[0]


# Download all relevant binaries in binaries.json
all_tags = list(tags.sys_tags())
binaries_dict = json.load(open("binaries.json"))
binaries_paths = _get_paths_from_binaries(binaries_dict)
terminate = False
version = os.environ["VERSION"] if "VERSION" in os.environ else "main"
configs_response = request.urlopen(
    "https://github.com/unifyai/binaries/raw/main/configs.txt",
    timeout=40,
)
available_configs = repr(f"{configs_response.read()}").strip(r"\"b\'").split(r"\\n")

# download binaries for the tag with highest precedence
for tag in all_tags:
    if terminate:
        break
    if str(tag) not in available_configs:
        continue
    for path in binaries_paths:
        if os.path.exists(path):
            continue
        folders = path.split(os.sep)
        folder_path, file_path = os.sep.join(folders[:-1]), folders[-1]
        file_name = f"{file_path[:-3]}_{tag}.so"
        search_path = f"compiler/{file_name}"
        try:
            response = request.urlopen(
                f"https://github.com/unifyai/binaries/raw/{version}/{search_path}",
                timeout=40,
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(response.read())
            terminate = path == binaries_paths[-1]
        except request.HTTPError:
            break

# verify if all binaries are available
for idx, path in enumerate(binaries_paths):
    if not os.path.exists(path):
        if idx == 0:
            config_str = "\n".join(available_configs)
            print(f"\nFollowing are the supported configurations :\n{config_str}\n")
        print(
            f"Could not download {path}.",
            end="\n\n" if idx == len(binaries_paths) - 1 else "\n",
        )


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Remove img tags that have class "only-dark"
long_description = re.sub(
    r"<img [^>]*class=\"only-dark\"[^>]*>",
    "",
    long_description,
    flags=re.MULTILINE,
)

# Remove a tags that have class "only-dark"
long_description = re.sub(
    r"<a [^>]*class=\"only-dark\"[^>]*>((?:(?!<\/a>).)|\s)*<\/a>\n",
    "",
    long_description,
    flags=re.MULTILINE,
)

# Apply version
with open("ivy/_version.py") as f:
    exec(f.read(), __version__)

setup(
    name="ivy",
    version=__version__,
    author="Unify",
    author_email="hello@unify.ai",
    description=(
        "The unified machine learning framework, enabling framework-agnostic "
        "functions, layers and libraries."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://unify.ai/ivy",
    project_urls={
        "Docs": "https://unify.ai/docs/ivy/",
        "Source": "https://github.com/unifyai/ivy",
    },
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        _strip(line)
        for line in open("requirements/requirements.txt", "r", encoding="utf-8")
    ],
    classifiers=["License :: OSI Approved :: Apache Software License"],
    license="Apache 2.0",
)
