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
import setuptools
from pathlib import Path
from distutils.core import setup


def _strip(line):
    return line.split(' ')[0].split('#')[0].split(',')[0]

def is_html(line):
    line_squashed = line.replace(' ', '')
    if not line_squashed:
        return False
    if line_squashed[0] == '<' and line_squashed[-1] == '>':
        return True
    return False

def is_raw_block(line):
    line_squashed = line.replace(' ', '')
    if len(line_squashed) < 11:
        return False
    if line_squashed[-11:] == '..raw::html':
        return True
    return False


this_directory = Path(__file__).parent
lines = (this_directory / "README.rst").read_text().split('\n')
lines = [line for line in lines if not (is_html(line) or is_raw_block(line))]
long_description = '\n'.join(lines)


setup(name='ivy-core',
      version='1.1.6',
      author='Ivy Team',
      author_email='ivydl.team@gmail.com',
      description='The templated deep learning framework, enabling framework-agnostic functions, layers and libraries.',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      url='https://ivy-dl.org/ivy',
      project_urls={
            'Docs': 'https://ivy-dl.org/ivy/',
            'Source': 'https://github.com/ivy-dl/ivy',
      },
      packages=setuptools.find_packages(),
      install_requires=[_strip(line) for line in open('requirements.txt', 'r')],
      classifiers=['License :: OSI Approved :: Apache Software License'],
      license='Apache 2.0'
      )
