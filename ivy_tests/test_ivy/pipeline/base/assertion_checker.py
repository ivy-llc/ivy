import numpy as np
from .runners import TestCaseSubRunnerResult


class AssertionChecker:
    def __init__(
        self,
        target_results: TestCaseSubRunnerResult,
        ground_truth_results: TestCaseSubRunnerResult,
        rtol: float,
        atol: float,
    ):
        self.target_results = target_results
        self.ground_truth_results = ground_truth_results
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

    def check_assertions(self):
        self._assert_dtype(self.target_results.dtype, self.ground_truth_results.dtype)
        self._assert_type(self.target_results.type, self.ground_truth_results.type)
        self._assert_device(
            self.target_results.device, self.ground_truth_results.device
        )
        self._assert_equal_elements(
            self.target_results.flatten_elements_np,
            self.ground_truth_results.flatten_elements_np,
        )
