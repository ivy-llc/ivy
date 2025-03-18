from ivy.utils.exceptions import IvyException
import collections


class SourceToSourceTranslatorException(IvyException):
    def __init__(self, *messages, include_backend=False, propagate=False):
        super().__init__(*messages, include_backend=include_backend)
        self.propagate = propagate


class InvalidSourceException(SourceToSourceTranslatorException):
    def __init__(self, *messages, include_backend=False, propagate=False):
        super().__init__(*messages, include_backend=include_backend, propagate=propagate)


class InvalidTargetException(SourceToSourceTranslatorException):
    def __init__(self, *messages, include_backend=False, propagate=False):
        super().__init__(*messages, include_backend=include_backend, propagate=propagate)


class InvalidObjectException(SourceToSourceTranslatorException):
    def __init__(self, *messages, include_backend=False, propagate=False):
        super().__init__(*messages, include_backend=include_backend, propagate=propagate)


class ProhibitedObjectAccessError(SourceToSourceTranslatorException):
    def __init__(self, *messages, include_backend=False, propagate=False):
        default_message = (
            "Direct access to the '.obj' attribute is prohibited. "
            "Use the available class methods or attributes to retrieve object information instead."
        )

        # If no custom messages are passed, use the default message
        if not messages:
            messages = (default_message,)
        super().__init__(*messages, include_backend=include_backend, propagate=propagate)


class ExpiredBinariesException(SourceToSourceTranslatorException):
    def __init__(self, *messages, include_backend=False, propagate=False):
        super().__init__(*messages, include_backend=include_backend, propagate=propagate)


def format_missing_frontends_msg(frequency):
    """Format the missing frontends warning message."""
    missing_functions = "\n-> ".join(
        [f" ({freq[1]}) \t{freq[0]}" for freq in frequency]
    )

    msg = (
        "\n\n(MissingFrontendsWarning): Some functions are not yet implemented in the Ivy frontend API."
        + " Visit Ivy's open task page to learn more about contributing: "
        + "https://www.docs.ivy.dev/overview/contributing/open_tasks.html \n\n"
        + "The missing functions are listed below as <(number of calls) function_path>:\n-> {}\n\n"
        + "Proceeding with transpilation, but be aware that the computation may fail if it reaches a point "
        + "where these missing frontends are required.\n"
    ).format(missing_functions)

    return msg
