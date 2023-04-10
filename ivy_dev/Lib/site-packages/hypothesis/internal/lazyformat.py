# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.


class lazyformat:
    """A format string that isn't evaluated until it's needed."""

    def __init__(self, format_string, *args):
        self.__format_string = format_string
        self.__args = args

    def __str__(self):
        return self.__format_string % self.__args

    def __eq__(self, other):
        return (
            isinstance(other, lazyformat)
            and self.__format_string == other.__format_string
            and self.__args == other.__args
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__format_string)
