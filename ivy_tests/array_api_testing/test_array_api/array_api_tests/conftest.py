import re

import pytest

r_int_pow_promotion = re.compile(r"test.+promotion\[.*pow.*\(u?int.+\]")


def pytest_collection_modifyitems(config, items):
    """Skips the faulty integer type promotion tests for pow-related functions"""
    for item in items:
        if r_int_pow_promotion.match(item.name):
            item.add_marker(
                pytest.mark.skip(
                    reason="faulty test logic - negative exponents generated"
                )
            )
