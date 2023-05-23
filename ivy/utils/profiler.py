import cProfile
import pstats
import subprocess
import logging
from tempfile import NamedTemporaryFile
from importlib.util import find_spec

is_snakeviz = find_spec("snakeviz")


class Profiler(cProfile.Profile):
    """
    A Profiler class that allows code profiling.

    Attributes
    ----------
        print_stats (bool, optional): prints profiling statistics.
        viz (bool, optional): visualizes the results using `snakeviz`.

        Bonus args and kwargs are passed to cProfile.Profile __init__

    Example
    -------
        with Profiler(print_stats=False, viz=True):
            fn(x, y)
    """

    def __init__(self, *args, **kwargs):
        self.print_stats = kwargs.pop("print_stats", True)
        self.viz = kwargs.pop("viz", False)

        super().__init__(*args, **kwargs)

    def __enter__(self, *args, **kwargs):
        self.pr = super().__enter__(*args, **kwargs)

        return self.pr

    def __exit__(self, *exc):
        super().__exit__(*exc)
        if exc == (None, None, None):
            stats = pstats.Stats(self.pr)
            stats.sort_stats(pstats.SortKey.TIME)

            if self.viz:
                if is_snakeviz:
                    # creates a temp file that gets automatically
                    # deleted when everything is done
                    with NamedTemporaryFile(suffix=".prof") as f:
                        stats.dump_stats(filename=f.name)
                        subprocess.run(["snakeviz", f"{f.name}"])
                else:
                    logging.warn("snakeviz must be installed for visualization")

            if self.print_stats:
                stats.print_stats()
