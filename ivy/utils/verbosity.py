# TODO should this still be here?
import termcolor
import os
import sys
from contextlib import contextmanager

level = 0


def cprint(message, color="green"):
    print(termcolor.colored(message, color))


@contextmanager
def hide_prints():
    _original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = _original_stdout