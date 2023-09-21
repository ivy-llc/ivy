# global
import importlib
import numpy as np

# local

from ivy_tests.test_ivy.pipeline.base.argument_searcher import ArgumentsSearcher
from ivy_tests.test_ivy.pipeline.base.runners import (
    TestCaseRunner,
    TestCaseSubRunner,
    TestCaseSubRunnerResult,
)
import ivy


class FunctionTestCaseSubRunner(TestCaseSubRunner):
    def __init__(
        self,
        fn_tree,
        frontend,
        backend_handler,
        backend,
        device,
        input_dtypes,
        test_flags,
    ):
        self.fn_tree = fn_tree
        self.frontend = frontend
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

    @staticmethod
    def _frontend_array_to_ivy(x):
        if FunctionTestCaseSubRunner._is_frontend_array(x):
            return x.ivy_array
        else:
            return x

    @property
    def _ivy(self):
        return self.__ivy

    def _arrays_to_frontend(self, frontend_array_fn=None):
        def _new_fn(x, *args, **kwargs):
            if FunctionTestCaseSubRunner._is_frontend_array(x):
                return x
            elif self._ivy.is_array(x):
                if tuple(x.shape) == ():
                    try:
                        ret = frontend_array_fn(x, dtype=self._ivy.Dtype(str(x.dtype)))
                    except self._ivy.utils.exceptions.IvyException:
                        ret = self._ivy(x, dtype=self._ivy.array(x).dtype)
                else:
                    ret = frontend_array_fn(x)
                return ret
            return x

        return _new_fn

    def _args_to_frontend(
        self, *args, frontend_array_fn=None, include_derived=None, **kwargs
    ):
        frontend_args = self._ivy.nested_map(
            args,
            self._arrays_to_frontend(frontend_array_fn=frontend_array_fn),
            include_derived,
            shallow=False,
        )
        frontend_kwargs = self._ivy.nested_map(
            kwargs,
            self._arrays_to_frontend(frontend_array_fn=frontend_array_fn),
            include_derived,
            shallow=False,
        )
        return frontend_args, frontend_kwargs

    def _search_args(self, test_arguments):
        # args_np, kwargs_np = self._split_args_to_args_and_kwargs(
        #     num_positional_args=self.test_flags.num_positional_args,
        #     test_arguments=test_arguments,
        # )
        # # Extract all arrays from the arguments and keyword arguments
        # arg_np_arrays, arrays_args_indices, n_args_arrays = (
        #   self._get_nested_np_arrays(
        #       args_np
        #   )
        # )
        # kwarg_np_arrays, arrays_kwargs_indices, n_kwargs_arrays = (
        #     self._get_nested_np_arrays(kwargs_np)
        # )
        #
        # total_num_arrays = n_args_arrays + n_kwargs_arrays
        # args_result = TestArgumentsSearchResult(
        #     args_np, arg_np_arrays, arrays_args_indices
        # )
        # kwargs_result = TestArgumentsSearchResult(
        #     kwargs_np, kwarg_np_arrays, arrays_kwargs_indices
        # )
        # return args_result, kwargs_result, total_num_arrays
        arg_searcher = ArgumentsSearcher(test_arguments)
        return arg_searcher.search_args(self.test_flags.num_positional_args)

    def _get_frontend_submodule(self):
        split_index = self.fn_tree.rfind(".")
        frontend_submods, fn_name = (
            self.fn_tree[:split_index],
            self.fn_tree[split_index + 1 :],
        )
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
        # ToDO: conversion of args and kwargs to frontend arrays
        assert (
            not self.test_flags.with_out or not self.test_flags.inplace
        ), "only one of with_out or with_inplace can be set as True"
        self._get_frontend_creation_fn()
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

    def _get_frontend_creation_fn(self):
        # ToDo: do this through config file
        return self.local_importer.import_module(
            f"ivy.functional.frontends.{self.frontend}"
        )._frontend_array

    def _get_frontend_function(self, args, kwargs):
        f_submod, fn_name = self._get_frontend_submodule()
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
        frontend_fn = self._get_frontend_function(args, kwargs)

        as_ivy_arrays = not self.test_flags.generate_frontend_arrays
        if not as_ivy_arrays and self.test_flags.test_compile:
            args, kwargs = self._ivy.nested_map(
                (args, kwargs),
                FunctionTestCaseSubRunner._frontend_array_to_ivy,
                include_derived={"tuple": True},
            )
        with self._ivy.PreciseMode(self.test_flags.precision_mode):
            ret = frontend_fn(*args, **kwargs)
        if self.test_flags.test_compile:
            ret = self._ivy.nested_map(
                ret, self._ivy.asarray, include_derived={"tuple": True}
            )
        else:
            ret = self._ivy.nested_map(
                ret,
                FunctionTestCaseSubRunner._frontend_array_to_ivy,
                include_derived={"tuple": True},
            )
        # TodO: check cls_type after you've created the asertion
        cls_type = None
        if (
            self.test_flags.generate_frontend_arrays
            and not self.test_flags.test_compile
        ):
            cls_type = ret.__class__.__name__

        ret_device = self._ivy.dev(ret) if self._ivy.is_array(ret) else None
        # got the ret as ivy array
        # flattening it to numpy array
        ret_np_flat = self._flatten_and_to_np(ret=ret)
        return TestCaseSubRunnerResult(
            flatten_elements_np=ret_np_flat,
            shape=self._ivy.shape(ret),
            device=ret_device,
            dtype=self._ivy.dtype(ret),
            type=cls_type,
        )

    def get_results(self, test_arguments):
        # split the arguments into their positional and keyword components
        args_result, kwargs_result, total_num_arrays = self._search_args(test_arguments)

        self._preprocess_flags(total_num_arrays)
        args, kwargs = self._preprocess_args(args_result, kwargs_result)

        return self._call_function(args, kwargs)


class GTFunctionTestCaseSubRunner(TestCaseSubRunner):
    def __init__(
        self,
        gt_fn_tree,
        fn_tree,
        frontend,
        backend_handler,
        backend,
        device,
        input_dtypes,
        test_flags,
    ):
        self.gt_fn_tree = gt_fn_tree
        self.fn_tree = fn_tree
        self.frontend = frontend
        self.test_flags = test_flags
        self.input_dtypes = input_dtypes
        self.on_device = device
        self.backend = backend
        self._backend_handler = backend_handler
        self.__ivy = self._backend_handler.set_backend(backend)
        self.frontend_config = self._get_frontend_config()

    def _get_frontend_config(self):
        config_module = importlib.import_module(
            f"ivy_tests.test_ivy.test_frontends.config.{self.frontend}"
        )
        return config_module.get_config()

    def _get_frontend_submodule(self):
        fn_name = self.fn_tree[self.fn_tree.rfind(".") + 1 :]
        # if gt_fn_tree and gt_fn_name are different from our frontend structure
        if self.gt_fn_tree is not None:
            split_index = self.gt_fn_tree.rfind(".")
            gt_frontend_submods, gt_fn_name = (
                self.gt_fn_tree[:split_index],
                self.gt_fn_tree[split_index + 1 :],
            )
        else:
            gt_frontend_submods, gt_fn_name = (
                self.fn_tree[25 : self.fn_tree.rfind(".")],
                fn_name,
            )
        return gt_frontend_submods, gt_fn_name

    def _get_frontend_fn(self):
        gt_frontend_submods, gt_fn_name = self._get_frontend_submodule(self.gt_fn_tree)
        frontend_fw = importlib.import_module(gt_frontend_submods)
        return frontend_fw.__dict__[gt_fn_name]

    def _search_args(self, test_arguments):
        arg_searcher = ArgumentsSearcher(test_arguments)
        return arg_searcher.search_args(self.test_flags.num_positional_args)

    def _preprocess_args(self, args, kwargs):
        # getting values from TestArgumentsSearchResult
        args, kwargs = args.original, kwargs.original
        args_frontend = ivy.nested_map(
            lambda x: (
                self.frontend_config.native_array(x)
                if isinstance(x, np.ndarray)
                else (
                    self.frontend_config.as_native_dtype(x)
                    if isinstance(x, self.frontend_config.Dtype)
                    else x
                )
            ),
            args,
            shallow=False,
        )
        kwargs_frontend = ivy.nested_map(
            lambda x: (
                self.frontend_config.native_array(x) if isinstance(x, np.ndarray) else x
            ),
            kwargs,
            shallow=False,
        )

        # change ivy dtypes to native dtypes
        if "dtype" in kwargs_frontend and kwargs_frontend["dtype"] is not None:
            kwargs_frontend["dtype"] = self.frontend_config.as_native_dtype(
                kwargs_frontend["dtype"]
            )

        # change ivy device to native devices
        if "device" in kwargs_frontend:
            kwargs_frontend["device"] = self.frontend_config.as_native_device(
                kwargs_frontend["device"]
            )
        return args_frontend, kwargs_frontend

    def _flatten_and_to_np(self, *, ret):
        if self.frontend_config.isscalar(ret):
            frontend_ret_np_flat = [self.frontend_config.to_numpy(ret)]
        else:
            # tuplify the frontend return
            if not isinstance(ret, tuple):
                frontend_ret = (ret,)
            frontend_ret_idxs = ivy.nested_argwhere(
                frontend_ret, self.frontend_config.is_native_array
            )
            frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
            frontend_ret_np_flat = [
                self.frontend_config.to_numpy(x) for x in frontend_ret_flat
            ]
        return frontend_ret_np_flat

    def _call_function(self, args, kwargs):
        frontend_fn = self._get_frontend_fn()
        ret = frontend_fn(*args, **kwargs)
        ret_np_flat = self._flatten_and_to_np(ret=ret)
        # ToDo: modify frontend configs to get shape as list and dtype/device as str

        return TestCaseSubRunnerResult(
            flatten_elements_np=ret_np_flat,
            # shape=ivy.shape(ret),
            # device=ret_device,
            # dtype=ivy.dtype(ret)
        )

    def get_results(self, test_arguments):
        args_result, kwargs_result, _ = self._search_args(test_arguments)
        args, kwargs = self._preprocess_args(args_result, kwargs_result)
        return self._call_function(args, kwargs)


class FrontendTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        backend_handler,
        fn_tree,
        backend_to_test,
        gt_fn_tree,
        frontend,
        on_device,
        rtol,
        atol,
    ):
        self.fn_tree = fn_tree
        self.backend_handler = backend_handler
        self.backend_to_test = backend_to_test
        self.gt_fn_tree = gt_fn_tree
        self.frontend = frontend
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
            self.fn_tree,
            self.frontend,
            self.backend_handler,
            self.backend_to_test,
            self.on_device,
            input_dtypes,
            test_flags,
        )
        ret = sub_runner_target.get_results(test_arguments)
        sub_runner_target.exit()
        return ret

    def _run_ground_truth(self, input_dtypes, test_arguments, test_flags):
        # no need to exit since we're not setting any backend for this
        sub_runner_gt = GTFunctionTestCaseSubRunner(
            self.gt_fn_tree,
            self.fn_tree,
            self.frontend,
            self.on_device,
            input_dtypes,
            test_flags,
        )
        return sub_runner_gt.get_results(test_arguments)

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
