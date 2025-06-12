import warnings


def cleanup_and_fetch_binaries(clean=True):
    warnings.warn(
        "The Ivy binaries are no longer used - there is no need to fetch them",
        DeprecationWarning,
    )
