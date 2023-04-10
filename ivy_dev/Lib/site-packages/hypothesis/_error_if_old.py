# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import sys

from hypothesis.version import __version__

message = """
Hypothesis {} requires Python 3.7 or later.

This can only happen if your packaging toolchain is older than python_requires.
See https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

if sys.version_info[:2] < (3, 7):
    raise Exception(message.format(__version__))
