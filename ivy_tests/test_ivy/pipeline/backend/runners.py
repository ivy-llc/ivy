import numpy as np
from ivy_tests.test_ivy.pipeline.base.runners import (
    TestCaseRunner,
    TestCaseSubRunnerResult,
)


class BackendTestCaseRunner(TestCaseRunner):
    def __init__(
        self, backend_handler, backend_to_test, ground_truth_backend, rtol, atol
    ):
        self.backend_handler = backend_handler
        self.backend_to_test = backend_to_test
        self.grond_truth_backend = ground_truth_backend
        self.rtol = rtol
        self.atol = atol

    def _assert_type(self, target_type, ground_truth_type):
        assert target_type == ground_truth_type

    def _assert_dtype(self, target_dtype, ground_truth_dtype):
        assert target_dtype == ground_truth_dtype

    def _assert_device(self, target_device, ground_truth_device):
        assert target_device == ground_truth_device, (
            f"ground truth backend ({self.ground_truth_backend}) returned array on"
            f" device {ground_truth_device} but target backend ({self.backend_to_test})"
            f" returned array on device {target_device}"
        )

    def _assert_equal_elements(self, target_elements, ground_truth_elements):
        assert np.allclose(
            np.nan_to_num(target_elements),
            np.nan_to_num(ground_truth_elements),
            rtol=self.rtol,
            atol=self.atol,
        ), (
            f" the results from backend {self.backend_to_test} "
            f"and ground truth framework {self.ground_truth_backend} "
            f"do not match\n {target_elements}!={ground_truth_elements} \n\n"
        )

    def _call_target(self, test_arguments, test_flags):
        pass

    def _call_ground_truth(self, test_arguments, test_flags):
        pass

    def run(self, test_arguments, test_flags):
        target_results: TestCaseSubRunnerResult = self._call_target(
            test_arguments, test_flags
        )
        ground_truth_results: TestCaseSubRunnerResult = self._call_ground_truth(
            test_arguments, test_flags
        )

        self._assert_dtype(target_results.dtype, ground_truth_results.dtype)
        self._assert_type(target_results.type, ground_truth_results.type)
        self._assert_device(target_results.device, ground_truth_results.device)
        self._assert_equal_elements(
            target_results.flatten_elements_np, ground_truth_results.flatten_elements_np
        )
