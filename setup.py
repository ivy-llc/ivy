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

from pathlib import Path
import re
import setuptools
from setuptools import setup


def _strip(line):
    return line.split(" ")[0].split("#")[0].split(",")[0]


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
    author="Ivy",
    author_email="contact@ivy.dev",
    description="Convert Machine Learning Code Between Frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://ivy.dev",
    project_urls={
        "Docs": "https://docs.ivy.dev/",
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
)
