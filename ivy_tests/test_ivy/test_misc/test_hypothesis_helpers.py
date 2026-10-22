import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.fixture(scope="module")
def helper_samples_by_hash_seed():
    script = textwrap.dedent(
        """
        import json
        from hypothesis import given, settings, strategies as st

        from ivy_tests.test_ivy import helpers
        from ivy_tests.test_ivy.helpers import globals as test_globals

        samples = {"get_dtypes": [], "einsum_helper": []}
        test_globals.setup_api_test(
            backend="numpy", ground_truth_backend=None, device="cpu"
        )

        @settings(
            derandomize=True, database=None, deadline=None, max_examples=30
        )
        @given(st.data())
        def collect(data):
            if helper == "get_dtypes":
                all_dtypes = data.draw(
                    helpers.get_dtypes(prune_function=False)
                )
                selected = data.draw(
                    helpers.get_dtypes(
                        prune_function=False, full=False, index=1
                    )
                )
                with_none = data.draw(
                    helpers.get_dtypes(prune_function=False, none=True)
                )
                samples[helper].append([all_dtypes, selected, with_none])
            else:
                equation, arrays, dtypes = data.draw(helpers.einsum_helper())
                samples[helper].append(
                    [equation, [array.shape for array in arrays], dtypes]
                )

        try:
            for helper in samples:
                collect()
        finally:
            test_globals.teardown_api_test()
        print(json.dumps(samples))
        """
    )
    outputs = []
    for hash_seed in ("1", "2"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        outputs.append(json.loads(result.stdout))

    return outputs


@pytest.mark.parametrize("helper", ["get_dtypes", "einsum_helper"])
def test_hypothesis_helpers_are_independent_of_hash_seed(
    helper, helper_samples_by_hash_seed
):
    first, second = helper_samples_by_hash_seed
    assert first[helper] == second[helper]
