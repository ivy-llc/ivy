import numpy as np
from ivy_tests.test_ivy.pipeline.base.runners import (
    TestCaseRunner,
    TestCaseSubRunner,
    TestCaseSubRunnerResult,
)


class FunctionTestCaseSubRunner(TestCaseSubRunner):
    def __init__(self, backend_handler, backend, input_dtypes, test_flags):
        self._backend_handler = backend_handler
        self.__ivy = self._backend_handler.set_backend(backend)
        self.test_flags = test_flags
        self.input_dtypes = input_dtypes

    @property
    def backend_handler(self):
        return self._backend_handler

    @property
    def _ivy(self):
        return self.__ivy

    def _search_args(self):
        pass

    def _preprocess_flags(self, total_num_arrays):
        # Make all array-specific test flags and dtypes equal in length
        if len(self.input_dtypes) < total_num_arrays:
            self.input_dtypes = [self.input_dtypes[0] for _ in range(total_num_arrays)]
        if len(self.test_flags.as_variable) < total_num_arrays:
            self.test_flags.as_variable = [
                self.test_flags.as_variable[0] for _ in range(total_num_arrays)
            ]
        if len(self.test_flags.native_arrays) < total_num_arrays:
            self.test_flags.native_arrays = [
                self.test_flags.native_arrays[0] for _ in range(total_num_arrays)
            ]
        if len(self.test_flags.container) < total_num_arrays:
            self.test_flags.container = [
                self.test_flags.container[0] for _ in range(total_num_arrays)
            ]
        # Update variable flags to be compatible with float dtype and with_out args
        self.test_flags.as_variable = [
            v if self._ivy.is_float_dtype(d) and not self.test_flags.with_out else False
            for v, d in zip(self.test_flags.as_variable, self.input_dtypes)
        ]

        # TODO this is not ideal, modifying Hypothesis generated value
        # May result in weird bugs. Should instead update strategies to
        # Not generate this in first place.
        # update instance_method flag to only be considered if the
        self.test_flags.instance_method = self.test_flags.instance_method and (
            not self.test_flags.native_arrays[0] or self.test_flags.container[0]
        )
        return self.test_flags

    def _preprocess_args(self):
        pass

    def _call_function(self):
        pass

    def get_results(self, test_arguments):
        # split the arguments into their positional and keyword components
        args_np, kwargs_np = self._split_args_to_args_and_kwargs(
            num_positional_args=self.test_flags.num_positional_args,
            test_arguments=test_arguments,
        )

        # Extract all arrays from the arguments and keyword arguments
        arg_np_arrays, arrays_args_indices, n_args_arrays = self._get_nested_np_arrays(
            args_np
        )
        kwarg_np_arrays, arrays_kwargs_indices, n_kwargs_arrays = (
            self._get_nested_np_arrays(kwargs_np)
        )

        total_num_arrays = n_args_arrays + n_kwargs_arrays
        self._preprocess_flags(total_num_arrays)

        args, kwargs, idx_args, idx_kwargs = self._search_args(test_arguments)
        args, kwargs = self._preprocess_args(args, kwargs, idx_args, idx_kwargs)
        return self._call_function(args, kwargs)


class BackendTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        backend_handler,
        fn_name,
        backend_to_test,
        ground_truth_backend,
        on_device,
        rtol,
        atol,
    ):
        self.fn_name = fn_name
        self.backend_handler = backend_handler
        self.backend_to_test = backend_to_test
        self.grond_truth_backend = ground_truth_backend
        self.on_device = on_device
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

    def _run_target(self, input_dtypes, test_arguments, test_flags):
        sub_runner_target = FunctionTestCaseSubRunner(
            self.backend_handler, self.backend_to_test, input_dtypes, test_flags
        )
        sub_runner_target.get_results(test_arguments)
        sub_runner_target.exit()

    def _run_ground_truth(self, input_dtypes, test_arguments, test_flags):
        pass

    def run(self, input_dtypes, test_arguments, test_flags):
        target_results: TestCaseSubRunnerResult = self._run_target(
            input_dtypes, test_arguments, test_flags
        )
        ground_truth_results: TestCaseSubRunnerResult = self._run_ground_truth(
            input_dtypes, test_arguments, test_flags
        )

        self._assert_dtype(target_results.dtype, ground_truth_results.dtype)
        self._assert_type(target_results.type, ground_truth_results.type)
        self._assert_device(target_results.device, ground_truth_results.device)
        self._assert_equal_elements(
            target_results.flatten_elements_np, ground_truth_results.flatten_elements_np
        )
