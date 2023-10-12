import pytest
import logging
import ivy


def test_set_logging_mode():
    ivy.set_logging_mode("DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    ivy.set_logging_mode("INFO")
    assert logging.getLogger().level == logging.INFO

    ivy.set_logging_mode("WARNING")
    assert logging.getLogger().level == logging.WARNING

    ivy.set_logging_mode("ERROR")
    assert logging.getLogger().level == logging.ERROR


def test_unset_logging_mode():
    ivy.set_logging_mode("DEBUG")
    ivy.set_logging_mode("INFO")
    ivy.unset_logging_mode()
    assert logging.getLogger().level == logging.DEBUG


def test_invalid_logging_mode():
    with pytest.raises(AssertionError):
        ivy.set_logging_mode("INVALID")
