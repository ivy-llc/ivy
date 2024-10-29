import logging

logging_modes = ["DEBUG", "INFO", "WARNING", "ERROR"]
# Set up the initial logging mode
logging.basicConfig(level=logging.WARNING)
logging_mode_stack = [logging.WARNING]


def set_logging_mode(mode):
    """Set the current logging mode for Ivy.

    Possible modes are 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
    """
    assert mode in logging_modes, "Invalid logging mode. Choose from: " + ", ".join(
        logging_modes
    )

    # Update the logging level
    logging.getLogger().setLevel(mode)
    logging_mode_stack.append(mode)


def unset_logging_mode():
    """Remove the most recently set logging mode, returning to the previous
    one."""
    if len(logging_mode_stack) > 1:
        # Remove the current mode
        logging_mode_stack.pop()

        # Set the previous mode
        logging.getLogger().setLevel(logging_mode_stack[-1])


# Expose the functions to the main Ivy package
__all__ = ["set_logging_mode", "unset_logging_mode"]
