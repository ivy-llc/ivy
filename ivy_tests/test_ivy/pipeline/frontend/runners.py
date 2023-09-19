from ivy_tests.test_ivy.pipeline.base.argument_searcher import TestArgumentsSearchResult
import numpy as np
from ivy_tests.test_ivy.pipeline.base.runners import (
    TestCaseRunner,
    TestCaseSubRunner,
    TestCaseSubRunnerResult,
)


class FunctionTestCaseSubRunner(TestCaseSubRunner):
    def __init__(
        self, fn_tree, input_dtypes, backend, device, backend_handler, test_flags
    ):
        self.fn_tree = fn_tree
        self.test_flags = test_flags
        self.input_dtypes = input_dtypes
        self.on_device = device
        self.backend = backend
        self._backend_handler = backend_handler
        self.__ivy = self._backend_handler.set_backend(backend)
        self.local_importer = self._ivy.utils.dynamic_import

    @staticmethod
    def _is_frontend_array(x):
        return hasattr(x, "ivy_array")

    @property
    def _ivy(self):
        return self.__ivy

    def _args_to_frontend(
        self, *args, frontend_array_fn=None, include_derived=None, **kwargs
    ):
        def _arrays_to_frontend(self, frontend_array_fn=None):
            def _new_fn(x, *args, **kwargs):
                if self._is_frontend_array(x):
                    return x
                elif self._ivy.is_array(x):
                    if tuple(x.shape) == ():
                        try:
                            ret = frontend_array_fn(
                                x, dtype=self._ivy.Dtype(str(x.dtype))
                            )
                        except self._ivy.utils.exceptions.IvyException:
                            ret = self._ivy(x, dtype=self._ivy.array(x).dtype)
                    else:
                        ret = frontend_array_fn(x)
                    return ret
                return x

            return _new_fn

        frontend_args = self._ivy.nested_map(
            args,
            _arrays_to_frontend(frontend_array_fn=frontend_array_fn),
            include_derived,
            shallow=False,
        )
        frontend_kwargs = self._ivy.nested_map(
            kwargs,
            _arrays_to_frontend(frontend_array_fn=frontend_array_fn),
            include_derived,
            shallow=False,
        )
        return frontend_args, frontend_kwargs

    def _search_args(self, test_arguments):
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
        args_result = TestArgumentsSearchResult(
            args_np, arg_np_arrays, arrays_args_indices
        )
        kwargs_result = TestArgumentsSearchResult(
            kwargs_np, kwarg_np_arrays, arrays_kwargs_indices
        )
        return args_result, kwargs_result, total_num_arrays

    def _get_frontend_submodule(self, fn_tree: str):
        split_index = fn_tree.rfind(".")
        frontend_submods, fn_name = fn_tree[:split_index], fn_tree[split_index + 1 :]
        return frontend_submods, fn_name

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

        self.test_flags.as_variable = [
            v if self._ivy.is_float_dtype(d) and not self.test_flags.with_out else False
            for v, d in zip(self.test_flags.as_variable, self.input_dtypes)
        ]
        return self.test_flags

    def _preprocess_args(self, args_result, kwargs_result):
        assert (
            not self.test_flags.with_out or not self.test_flags.inplace
        ), "only one of with_out or with_inplace can be set as True"
        ret = []
        for result, start_index_of_arguments in zip(
            [args_result, kwargs_result], [0, len(args_result.values)]
        ):
            temp = self._ivy.copy_nest(result.original, to_mutable=False)
            self._ivy.set_nest_at_indices(
                temp,
                result.indices,
                self.test_flags.apply_flags(
                    result.values,
                    self.input_dtypes,
                    start_index_of_arguments,
                    backend=self.backend,
                    on_device=self.on_device,
                ),
            )
            ret.append(temp)

        return ret[0], ret[1]

    def _get_frontend_function(self, args, kwargs):
        f_submod, fn_name = self._get_frontend_submodule(self.fn_tree)
        function_module = self.local_importer.import_module(f_submod)
        frontend_fn = getattr(function_module, fn_name)
        frontend_fn = self.compiled_if_required(
            frontend_fn,
            test_compile=self.test_flags.test_compile,
            args=args,
            kwargs=kwargs,
        )
        return frontend_fn

    def _call_function(self, args, kwargs):
        # determine the target frontend_fn
        # frontend_fn = self._get_frontend_function()
        #
        # as_ivy_arrays = not self.test_flags.generate_frontend_arrays
        # if not as_ivy_arrays and self.test_flags.test_compile:
        #     args, kwargs = self._ivy.nested_map(
        #         (args, kwargs),
        #         _frontend_array_to_ivy,
        #         include_derived={"tuple": True}
        #     )
        # with self._ivy.PreciseMode(self.test_flags.precision_mode):
        #     ret = frontend_fn(*args, **kwargs)
        # if self.test_flags.test_compile and frontend_array_function is not None:
        #     if as_ivy_arrays:
        #         ret = self._ivy.nested_map(
        #             ret, self._ivy.asarray, include_derived={"tuple": True}
        #         )
        #     else:
        #         ret = self._ivy.nested_map(
        #             ret,
        #             arrays_to_frontend(self.backend, frontend_array_function),
        #             include_derived={"tuple": True},
        #         )
        # elif as_ivy_arrays:
        #     ret = self._ivy.nested_map(
        #         ret, _frontend_array_to_ivy, include_derived={"tuple": True}
        #     )
        # return ret
        pass

    def get_results(self, test_arguments):
        # split the arguments into their positional and keyword components
        args_result, kwargs_result, total_num_arrays = self._search_args(test_arguments)

        self._preprocess_flags(total_num_arrays)
        args, kwargs = self._preprocess_args(args_result, kwargs_result)

        return self._call_function(args, kwargs)


class FrontendTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        backend_handler,
        fn_tree,
        backend_to_test,
        gt_fn_tree,
        on_device,
        rtol,
        atol,
    ):
        self.fn_tree = fn_tree
        self.backend_handler = backend_handler
        self.backend_to_test = backend_to_test
        self.gt_fn_tree = gt_fn_tree
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

    def _call_target(self, input_dtypes, test_arguments, test_flags):
        sub_runner_target = FunctionTestCaseSubRunner(
            self.fn_tree,
            input_dtypes,
            self.backend_to_test,
            self.on_device,
            self.backend_handler,
            test_flags,
        )
        sub_runner_target.get_results(test_arguments)

    def _call_ground_truth(self, input_dtypes, test_arguments, test_flags):
        pass

    def run(self, input_dtypes, test_arguments, test_flags):
        target_results: TestCaseSubRunnerResult = self._call_target(
            input_dtypes, test_arguments, test_flags
        )
        ground_truth_results: TestCaseSubRunnerResult = self._call_ground_truth(
            input_dtypes, test_arguments, test_flags
        )

        self._assert_dtype(target_results.dtype, ground_truth_results.dtype)
        self._assert_type(target_results.type, ground_truth_results.type)
        self._assert_device(target_results.device, ground_truth_results.device)
        self._assert_equal_elements(
            target_results.flatten_elements_np, ground_truth_results.flatten_elements_np
        )
