from ivy_tests.test_ivy.pipeline.base.assertion_checker import AssertionChecker


class FrontendAssertionChecker(AssertionChecker):
    def check_assertions(self):
        self._assert_dtype(self.target_results.dtype, self.ground_truth_results.dtype)
        self._assert_device(
            self.target_results.device, self.ground_truth_results.device
        )
        self._assert_same_length(
            self.target_results.flatten_elements_np,
            self.ground_truth_results.flatten_elements_np,
        )
        self._assert_equal_elements(
            self.target_results.flatten_elements_np,
            self.ground_truth_results.flatten_elements_np,
        )
