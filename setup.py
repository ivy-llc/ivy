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
from urllib import request
import os
import json
import re


def _get_paths_from_binaries(binaries, root_dir=""):
    """Get all the paths from the binaries.json into a list."""
    paths = []
    ext = "pyd" if os.name == "nt" else "so"
    if isinstance(binaries, str):
        return [os.path.join(root_dir, binaries + "." + ext)]
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
binaries_dict = json.load(open("binaries.json"))
available_configs = json.load(open("available_configs.json"))
binaries_paths = _get_paths_from_binaries(binaries_dict)
version = os.environ.get("VERSION", "main")
fixed_tag = os.environ.get("TAG", None)
clean = os.environ.get("CLEAN", None)
terminate = False
all_tags, python_tag, plat_name, options = None, None, None, None
if fixed_tag:
    python_tag, _, plat_name = str(fixed_tag).split("-")
    options = {"bdist_wheel": {"python_tag": python_tag, "plat_name": plat_name}}
    all_tags = [fixed_tag]
else:
    from pip._vendor.packaging import tags

    all_tags = list(tags.sys_tags())

# download binaries for the tag with highest precedence
for tag in all_tags:
    if terminate:
        break
    for path in binaries_paths:
        module = path.split(os.sep)[1]
        if (os.path.exists(path) and not clean) or str(tag) not in available_configs[
            module
        ]:
            continue
        folders = path.split(os.sep)
        folder_path, file_path = os.sep.join(folders[:-1]), folders[-1]
        ext = "pyd" if os.name == "nt" else "so"
        file_name = f"{file_path[:-(len(ext)+1)]}_{tag}.{ext}"
        search_path = f"{module}/{file_name}"
        try:
            response = request.urlopen(
                f"https://github.com/ivy-llc/binaries/raw/{version}/{search_path}",
                timeout=40,
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(response.read())
            terminate = path == binaries_paths[-1]
        except request.HTTPError:
            break


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
    author="Transpile AI",
    author_email="hello@transpile-ai.com",
    description=(
        "The unified machine learning framework, enabling framework-agnostic "
        "functions, layers and libraries."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://ivy.dev",
    project_urls={
        "Docs": "https://ivy.dev/docs/",
        "Source": "https://github.com/ivy-llc/ivy",
    },
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        _strip(line)
        for line in open("requirements/requirements.txt", "r", encoding="utf-8")
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
    license="Apache 2.0",
    options=options,
)
