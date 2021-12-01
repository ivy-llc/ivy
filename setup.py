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

def _replace_logos_html(txt):

    # html-containing chunks
    chunks = txt.split('.. raw:: html')

    # backend logos
    backends_chunk = chunks[2]
    bc = backends_chunk.split('\n\n')
    img_str = '.. image:: https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/logos/supported/frameworks.png?raw=true\n' \
              '   :width: 100%'
    backends_chunk = '\n\n'.join(bc[0:1] + [img_str] + bc[2:])

    # library logos
    libraries_chunk = chunks[3]
    lc = libraries_chunk.split('\n\n')
    img_str = '.. image:: https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_libraries.png?raw=true\n' \
              '   :width: 100%'
    libraries_chunk = '\n\n'.join(lc[0:1] + [img_str] + lc[2:])

    # re-join
    chunks[3] = libraries_chunk
    return ''.join(
        ['.. raw:: html'.join(chunks[0:2]), backends_chunk, '.. raw:: html'.join(chunks[3:])])

def _replace_gif(gif_chunk):
    png_url = 'https://{}.png'.format(gif_chunk.split(".gif?raw=true'>")[0].split('https://')[-1])
    gc = gif_chunk.split('\n\n')
    img_str = '.. image:: {}?raw=true\n' \
              '   :width: 100%'.format(png_url)
    return '\n\n'.join(gc[0:1] + [img_str] + gc[2:])

def _replace_gifs_html(txt):

    # html-containing chunks
    chunks = txt.split('.. raw:: html')

    # go through each chunk, replacing all html gifs with rst images
    return_str = ''
    for i, chunk in enumerate(chunks):
        new_chunk = chunk
        delimiter = '.. raw:: html'
        if ".gif?raw=true'>" in chunk:
            new_chunk = _replace_gif(chunk)
            delimiter = ''
        if i == 0:
            return_str = chunk
        else:
            return_str = delimiter.join([return_str, new_chunk])
    return return_str

def _is_html(line):
    line_squashed = line.replace(' ', '')
    if not line_squashed:
        return False
    if line_squashed[0] == '<' and line_squashed[-1] == '>':
        return True
    return False

def _is_raw_block(line):
    line_squashed = line.replace(' ', '')
    if len(line_squashed) < 11:
        return False
    if line_squashed[-11:] == '..raw::html':
        return True
    return False


this_directory = Path(__file__).parent
text = (this_directory / "README.rst").read_text()
text = _replace_logos_html(text).replace('. Click on the icons below to learn more!', '!')
text = _replace_gifs_html(text)
lines = text.split('\n')
lines = [line for line in lines if not (_is_html(line) or _is_raw_block(line))]
long_description = '\n'.join(lines)

setup(name='ivy-models',
      version='1.1.9',
      author='Ivy Team',
      author_email='ivydl.team@gmail.com',
      description='Collection of pre-trained models, compatible with any backend framework',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      url='https://ivy-dl.org/models',
      project_urls={
            'Docs': 'https://ivy-dl.org/models/',
            'Source': 'https://github.com/ivy-dl/models',
      },
      packages=setuptools.find_packages(),
      install_requires=[_strip(line) for line in open('requirements.txt', 'r')],
      classifiers=['License :: OSI Approved :: Apache Software License'],
      license='Apache 2.0'
      )
