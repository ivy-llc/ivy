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
from distutils.core import setup
import setuptools
setup(name='ivy-core',
      version='1.1.4',
      author='Ivy Team',
      author_email='ivydl.team@gmail.com',
      description='The templated deep learning framework, enabling framework-agnostic functions, layers and libraries.',
      long_description="""# What is Ivy?\n\nIvy is a templated deep learning framework which maximizes the portability
      of deep learning codebases. Ivy wraps the functional APIs of existing frameworks. Framework-agnostic functions,
      libraries and layers can then be written using Ivy, with simultaneous support for all frameworks.
      Ivy currently supports Jax, TensorFlow, PyTorch, MXNet and Numpy. Check out the [docs](https://ivy-dl.org/ivy) for more info!""",
      long_description_content_type='text/markdown',
      url='https://ivy-dl.org/ivy',
      project_urls={
            'Docs': 'https://ivy-dl.org/ivy/',
            'Source': 'https://github.com/ivy-dl/ivy',
      },
      packages=setuptools.find_packages(),
      install_requires=['h5py', 'numpy', 'termcolor'],
      classifiers=['License :: OSI Approved :: Apache Software License'],
      license='Apache 2.0'
      )
