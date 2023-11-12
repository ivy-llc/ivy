# global
import importlib
import inspect
import types
import numpy as np

# local

from ivy_tests.test_ivy.pipeline.base.argument_searcher import ArgumentsSearcher
from .assertion_checker import FrontendAssertionChecker
import ivy_tests.test_ivy.helpers.globals as t_globals
from ivy_tests.test_ivy.pipeline.base.runners import (
    TestCaseRunner,
    TestCaseSubRunner,
    TestCaseSubRunnerResult,
)
import ivy

try:
    import tensorflow as tf
except ImportError:
    tf = types.SimpleNamespace()
    tf.TensorShape = None


class FrontendTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        backend_handler,
        backend_to_test,
        frontend,
        on_device,
        rtol,
        atol,
        tolerance_dict,
        traced_fn,
        test_values,
    ):
        self.backend_handler = backend_handler
        self.backend_to_test = backend_to_test
        self.frontend = frontend
        self.on_device = on_device
        self.rtol = rtol
        self.atol = atol
        self.tolerance_dict = tolerance_dict
        self.traced_fn = traced_fn
        self.test_values = test_values

    def _check_assertions(self, target_results, ground_truth_results):
        if not self.test_values:
            return
        assertion_checker = FrontendAssertionChecker(
            target_results,
            ground_truth_results,
            self.backend_to_test,
            self.frontend,
            self.tolerance_dict,
            self.rtol,
            self.atol,
        )
        assertion_checker.check_assertions()


class FrontendTestCaseSubRunner(TestCaseSubRunner):
    def __init__(
        self,
        frontend,
        backend_handler,
        backend_to_test,
        on_device,
        traced_fn,
    ):
        self.frontend = frontend
        self.on_device = on_device
        self.backend_to_test = backend_to_test
        self.traced_fn = traced_fn
        self._backend_handler = backend_handler
        self.__ivy = self._backend_handler.set_backend(backend_to_test)
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

    @property
    def backend_handler(self):
        return self._backend_handler

    def _get_dev_str(self, x):
        config_module = importlib.import_module(
            f"ivy_tests.test_ivy.test_frontends.config.{self.frontend}"
        )
        return config_module.get_config().as_native_device(self._ivy.dev(x))

    def _arrays_to_frontend(self):
        def _new_fn(x):
            if FunctionTestCaseSubRunner._is_frontend_array(x):
                return x
            elif self._ivy.is_array(x):
                frontend_array_fn = self._get_frontend_creation_fn()
                if tuple(x.shape) == ():
                    try:
                        ret = frontend_array_fn(x, dtype=self._ivy.Dtype(str(x.dtype)))
                    except self._ivy.utils.exceptions.IvyException:
                        ret = frontend_array_fn(x, dtype=self._ivy.array(x).dtype)
                else:
                    ret = frontend_array_fn(x)
                return ret
            return x

        return _new_fn

    def _args_to_frontend(self, *args, include_derived=None, **kwargs):
        frontend_args = self._ivy.nested_map(
            self._arrays_to_frontend(),
            args,
            include_derived,
            shallow=False,
        )
        frontend_kwargs = self._ivy.nested_map(
            self._arrays_to_frontend(),
            kwargs,
            include_derived,
            shallow=False,
        )
        return frontend_args, frontend_kwargs

    def _get_frontend_creation_fn(self):
        # ToDo: do this through config file
        return self.local_importer.import_module(
            f"ivy.functional.frontends.{self.frontend}"
        )._frontend_array

    def _assert_ret_is_frontend_array(self, ret):
        assert self._ivy.nested_map(
            lambda x: (
                FunctionTestCaseSubRunner._is_frontend_array(x)
                if self._ivy.is_array(x)
                else True
            ),
            ret,
            shallow=False,
        ), "Frontend function returned non-frontend arrays: {}".format(ret)


class GTFrontendTestCaseSubRunner(TestCaseSubRunner):
    def __init__(
        self,
        frontend,
        on_device,
    ):
        self.frontend = frontend
        self.on_device = on_device
        self.frontend_config = self._get_frontend_config()

    def _get_frontend_config(self):
        config_module = importlib.import_module(
            f"ivy_tests.test_ivy.test_frontends.config.{self.frontend}"
        )
        return config_module.get_config()

    def _get_dev_str(self, x):
        return self.frontend_config.get_native_device(x)

    def _is_array(self, x):
        return self.frontend_config.is_native_array(x)

    def _flatten(self, *, ret):
        if self.frontend_config.isscalar(ret):
            frontend_ret_flat = [ret]
        else:
            # tuplify the frontend return
            frontend_ret = (ret,) if not isinstance(ret, tuple) else ret
            frontend_ret_idxs = ivy.nested_argwhere(
                frontend_ret, self.frontend_config.is_native_array
            )
            frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
        return frontend_ret_flat

    def _flatten_ret_to_np(self, ret_flat):
        return [self.frontend_config.to_numpy(x) for x in ret_flat]


class FrontendFunctionTestCaseRunner(FrontendTestCaseRunner):
    def __init__(
        self,
        backend_handler,
        fn_tree,
        backend_to_test,
        gt_fn_tree,
        frontend,
        on_device,
        traced_fn,
        test_values,
        tolerance_dict,
        rtol,
        atol,
    ):
        self.fn_tree = fn_tree
        self.gt_fn_tree = gt_fn_tree
        super().__init__(
            backend_handler,
            backend_to_test,
            frontend,
            on_device,
            rtol,
            atol,
            tolerance_dict,
            traced_fn,
            test_values,
        )

    def _run_target(self, input_dtypes, test_arguments, test_flags):
        sub_runner_target = FunctionTestCaseSubRunner(
            self.fn_tree,
            self.frontend,
            self.backend_handler,
            self.backend_to_test,
            self.on_device,
            self.traced_fn,
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
            test_flags,
            self.frontend,
            self.on_device,
        )
        ret = sub_runner_gt.get_results(test_arguments)
        return ret

    def run(self, input_dtypes, test_arguments, test_flags):
        # getting results from target and ground truth
        target_results: TestCaseSubRunnerResult = self._run_target(
            input_dtypes, test_arguments, test_flags
        )
        ground_truth_results: TestCaseSubRunnerResult = self._run_ground_truth(
            input_dtypes, test_arguments, test_flags
        )
        # checking assertions
        return self._check_assertions(target_results, ground_truth_results)


class FrontendMethodTestCaseRunner(FrontendTestCaseRunner):
    def __init__(
        self,
        backend_handler,
        frontend,
        frontend_method_data,
        backend_to_test,
        on_device,
        traced_fn,
        rtol_,
        atol_,
        tolerance_dict,
        test_values,
    ):
        self.frontend_method_data = frontend_method_data
        super().__init__(
            backend_handler,
            backend_to_test,
            frontend,
            on_device,
            rtol_,
            atol_,
            tolerance_dict,
            traced_fn,
            test_values,
        )

    def _run_target(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        sub_runner_target = MethodTestCaseSubRunner(
            self.frontend_method_data,
            self.frontend,
            self.backend_handler,
            self.backend_to_test,
            self.on_device,
            self.traced_fn,
            init_input_dtypes,
            method_input_dtypes,
            init_flags,
            method_flags,
        )
        ret = sub_runner_target.get_results(
            init_all_as_kwargs_np, method_all_as_kwargs_np
        )
        sub_runner_target.exit()
        return ret

    def _run_ground_truth(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        # no need to exit since we're not setting any backend for this
        sub_runner_target = GTMethodTestCaseSubRunner(
            self.frontend_method_data,
            self.frontend,
            self.on_device,
            init_flags,
            method_flags,
        )
        ret = sub_runner_target.get_results(
            init_all_as_kwargs_np, method_all_as_kwargs_np
        )
        return ret

    def run(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        # getting results from target and ground truth
        target_results: TestCaseSubRunnerResult = self._run_target(
            init_input_dtypes,
            method_input_dtypes,
            init_flags,
            method_flags,
            init_all_as_kwargs_np,
            method_all_as_kwargs_np,
        )
        ground_truth_results: TestCaseSubRunnerResult = self._run_ground_truth(
            init_input_dtypes,
            method_input_dtypes,
            init_flags,
            method_flags,
            init_all_as_kwargs_np,
            method_all_as_kwargs_np,
        )
        self._check_assertions(target_results, ground_truth_results)


class FunctionTestCaseSubRunner(FrontendTestCaseSubRunner):
    def __init__(
        self,
        fn_tree,
        frontend,
        backend_handler,
        backend_to_test,
        device,
        traced_fn,
        input_dtypes,
        test_flags,
    ):
        self.fn_tree = fn_tree
        self.test_flags = test_flags
        self.input_dtypes = input_dtypes
        super().__init__(
            frontend=frontend,
            backend_handler=backend_handler,
            backend_to_test=backend_to_test,
            on_device=device,
            traced_fn=traced_fn,
        )

    def _get_frontend_submodule(self):
        split_index = self.fn_tree.rfind(".")
        frontend_submods, fn_name = (
            self.fn_tree[:split_index],
            self.fn_tree[split_index + 1 :],
        )
        return frontend_submods, fn_name

    def _get_frontend_function(self, *args, **kwargs):
        if self.traced_fn is not None and self.test_flags.test_trace:
            return self.traced_fn
        f_submod, fn_name = self._get_frontend_submodule()
        function_module = self.local_importer.import_module(f_submod)
        frontend_fn = getattr(function_module, fn_name)
        if self.traced_fn is None:
            frontend_fn = self.trace_if_required(
                frontend_fn,
                test_trace=self.test_flags.test_trace,
                args=args,
                kwargs=kwargs,
            )
            t_globals.CURRENT_PIPELINE.set_traced_fn(frontend_fn)
        return frontend_fn

    def _test_inplace(self, frontend_fn, args, kwargs):
        # ToDO: Move this to the decorators and pass in inplace kwarg from there
        if "inplace" in list(inspect.signature(frontend_fn).parameters.keys()):
            kwargs["inplace"] = True
        check_array_fn = self._ivy.is_array
        conversion_fn = self._ivy.asarray
        if self.test_flags.generate_frontend_arrays and not self.test_flags.test_trace:
            check_array_fn = FunctionTestCaseSubRunner._is_frontend_array
            conversion_fn = FunctionTestCaseSubRunner._frontend_array_to_ivy
        first_array = self._ivy.func_wrapper._get_first_array(
            *args, array_fn=check_array_fn, **kwargs
        )
        with self._ivy.PreciseMode(self.test_flags.precision_mode):
            ret = frontend_fn(*args, **kwargs)
        if self.test_flags.test_trace:
            # output will always be native array
            if (
                self.test_flags.generate_frontend_arrays
                or not self.test_flags.native_arrays[0]
            ):
                # input is ivy array
                assert first_array.data is ret
            else:
                # input is native array
                assert first_array is ret
        else:
            # output will always be frontend array
            # asserting the returned arrays are frontend arrays
            self._assert_ret_is_frontend_array(ret)
            if self.test_flags.generate_frontend_arrays:
                # input is frontend array
                assert first_array is ret
            elif self.test_flags.native_arrays[0]:
                # input is native array
                assert first_array is ret.ivy_array.data
            else:
                # input is ivy array
                assert first_array is ret.ivy_array

        ret = self._ivy.nested_map(conversion_fn, ret, include_derived={"tuple": True})
        return ret

    def _test_out(self, ret, args, kwargs, frontend_fn):
        if not self.test_flags.test_trace:
            kwargs["out"] = ret
            ret = frontend_fn(*args, **kwargs)
            # Todo: confirm only ivy arrays are supposed to be inpalce updated
            assert self._ivy.nested_multi_map(
                lambda x, _: (
                    x[0].ivy_array is x[1]
                    if FunctionTestCaseSubRunner._is_frontend_array(x[0])
                    else True
                ),
                (ret, kwargs["out"]),
            ), "out argument is different from the returned array."
            del kwargs["out"]
        else:
            # Todo: confirm out behavior with tracer team
            pass

    def _search_args(self, test_arguments):
        arg_searcher = ArgumentsSearcher(test_arguments)
        return arg_searcher.search_args(self.test_flags.num_positional_args)

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
                    backend=self.backend_to_test,
                    on_device=self.on_device,
                ),
            )
            ret.append(temp)

        if self.test_flags.generate_frontend_arrays:
            ret[0], ret[1] = self._args_to_frontend(
                *ret[0],
                include_derived={"tuple": True},
                **ret[1],
            )
        return ret[0], ret[1]

    def _call_function(self, args, kwargs):
        # determine the target frontend_fn
        frontend_fn = self._get_frontend_function(*args, **kwargs)

        if self.test_flags.generate_frontend_arrays and self.test_flags.test_trace:
            # convert the args and kwargs to ivy arrays
            args, kwargs = self._ivy.nested_map(
                FunctionTestCaseSubRunner._frontend_array_to_ivy,
                (args, kwargs),
                include_derived={"tuple": True},
            )
        # testing inplace
        if self.test_flags.inplace:
            ret = self._test_inplace(frontend_fn, args, kwargs)
        else:
            with self._ivy.PreciseMode(self.test_flags.precision_mode):
                ret = frontend_fn(*args, **kwargs)

            # converting ret to ivy array
            if self.test_flags.test_trace:
                ret = self._ivy.nested_map(
                    self._ivy.asarray, ret, include_derived={"tuple": True}
                )
            else:
                # asserting the returned arrays are frontend arrays
                self._assert_ret_is_frontend_array(ret)

                ret = self._ivy.nested_map(
                    FunctionTestCaseSubRunner._frontend_array_to_ivy,
                    ret,
                    include_derived={"tuple": True},
                )

        # testing out argument
        if self.test_flags.with_out:
            self._test_out(ret, args, kwargs, frontend_fn)

        return self._get_results_from_ret(ret)

    def get_results(self, test_arguments):
        # split the arguments into their positional and keyword components
        args_result, kwargs_result, total_num_arrays = self._search_args(test_arguments)

        self._preprocess_flags(total_num_arrays)
        args, kwargs = self._preprocess_args(args_result, kwargs_result)

        return self._call_function(args, kwargs)


class MethodTestCaseSubRunner(FrontendTestCaseSubRunner):
    def __init__(
        self,
        frontend_method_data,
        frontend,
        backend_handler,
        backend_to_test,
        on_device,
        traced_fn,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
    ):
        self.frontend_method_data = frontend_method_data
        self.init_input_dtypes = init_input_dtypes
        self.method_input_dtypes = method_input_dtypes
        self.init_flags = init_flags
        self.method_flags = method_flags
        super().__init__(
            frontend=frontend,
            backend_handler=backend_handler,
            backend_to_test=backend_to_test,
            on_device=on_device,
            traced_fn=traced_fn,
        )

    def _get_frontend_function(
        self, init_args, init_kwargs, method_args, method_kwargs
    ):
        if self.traced_fn is not None and self.method_flags.test_trace:
            return self.traced_fn
        frontend_fw_module = self.local_importer.import_module(
            self.frontend_method_data.ivy_init_module
        )
        ivy_frontend_creation_fn = getattr(
            frontend_fw_module, self.frontend_method_data.init_name
        )
        ins = ivy_frontend_creation_fn(*init_args, **init_kwargs)
        frontend_fn = ins.__getattribute__(self.frontend_method_data.method_name)
        if self.traced_fn is None:
            frontend_fn = self.trace_if_required(
                frontend_fn,
                test_trace=self.method_flags.test_trace,
                args=method_args,
                kwargs=method_kwargs,
            )
            t_globals.CURRENT_PIPELINE.set_traced_fn(frontend_fn)
        return frontend_fn, ins

    def _test_inplace(self, frontend_fn, ins, args, kwargs):
        conversion_fn = self._ivy.asarray
        if (
            self.method_flags.generate_frontend_arrays
            and not self.method_flags.test_trace
        ):
            conversion_fn = FunctionTestCaseSubRunner._frontend_array_to_ivy
        with self._ivy.PreciseMode(self.method_flags.precision_mode):
            ret = frontend_fn(*args, **kwargs)
        # ins will always be a frontend array
        if self.method_flags.test_trace:
            # output will always be native array
            if (
                self.method_flags.generate_frontend_arrays
                or not self.method_flags.native_arrays[0]
            ):
                assert ins.ivy_array.data is ret
        else:
            # output will always be frontend array
            self._assert_ret_is_frontend_array(ret)
            if self.method_flags.generate_frontend_arrays:
                assert ins is ret

        ret = self._ivy.nested_map(conversion_fn, ret, include_derived={"tuple": True})
        return ret

    def _search_args(self, init_all_as_kwargs_np, method_all_as_kwargs_np):
        # init args searching
        arg_searcher = ArgumentsSearcher(init_all_as_kwargs_np)
        init_args = arg_searcher.search_args(self.init_flags.num_positional_args)

        # method args searching
        arg_searcher = ArgumentsSearcher(method_all_as_kwargs_np)
        method_args = arg_searcher.search_args(self.method_flags.num_positional_args)

        return init_args, method_args

    def _preprocess_init_flags(self, init_total_num_arrays):
        # Make all array-specific test flags and dtypes equal in length
        if len(self.init_input_dtypes) < init_total_num_arrays:
            self.init_input_dtypes = [
                self.init_input_dtypes[0] for _ in range(init_total_num_arrays)
            ]
        if len(self.init_flags.as_variable) < init_total_num_arrays:
            self.init_flags.as_variable = [
                self.init_flags.as_variable[0] for _ in range(init_total_num_arrays)
            ]
        if len(self.init_flags.native_arrays) < init_total_num_arrays:
            self.init_flags.native_arrays = [
                self.init_flags.native_arrays[0] for _ in range(init_total_num_arrays)
            ]

        self.init_flags.as_variable = [
            v if self._ivy.is_float_dtype(d) else False
            for v, d in zip(self.init_flags.as_variable, self.init_input_dtypes)
        ]

    def _preprocess_method_flags(self, method_total_num_arrays):
        # Make all array-specific test flags and dtypes equal in length
        if len(self.method_input_dtypes) < method_total_num_arrays:
            self.method_input_dtypes = [
                self.method_input_dtypes[0] for _ in range(method_total_num_arrays)
            ]
        if len(self.method_flags.as_variable) < method_total_num_arrays:
            self.method_flags.as_variable = [
                self.method_flags.as_variable[0] for _ in range(method_total_num_arrays)
            ]
        if len(self.method_flags.native_arrays) < method_total_num_arrays:
            self.method_flags.native_arrays = [
                self.method_flags.native_arrays[0]
                for _ in range(method_total_num_arrays)
            ]

        self.method_flags.as_variable = [
            v if self._ivy.is_float_dtype(d) else False
            for v, d in zip(self.method_flags.as_variable, self.method_input_dtypes)
        ]

    def _preprocess_flags(self, init_total_num_arrays, method_total_num_arrays):
        self._preprocess_init_flags(init_total_num_arrays)
        self._preprocess_method_flags(method_total_num_arrays)

    def _preprocess_init_args(self, init_args_result, init_kwargs_result):
        ret = []
        for result, start_index_of_arguments in zip(
            [init_args_result, init_kwargs_result], [0, len(init_args_result.values)]
        ):
            temp = self._ivy.copy_nest(result.original, to_mutable=False)
            self._ivy.set_nest_at_indices(
                temp,
                result.indices,
                self.init_flags.apply_flags(
                    result.values,
                    self.init_input_dtypes,
                    start_index_of_arguments,
                    backend=self.backend_to_test,
                    on_device=self.on_device,
                ),
            )
            ret.append(temp)

        return ret[0], ret[1]

    def _preprocess_method_args(self, method_args_result, method_kwargs_result):
        ret = []
        for result, start_index_of_arguments in zip(
            [method_args_result, method_kwargs_result],
            [0, len(method_args_result.values)],
        ):
            temp = self._ivy.copy_nest(result.original, to_mutable=False)
            self._ivy.set_nest_at_indices(
                temp,
                result.indices,
                self.method_flags.apply_flags(
                    result.values,
                    self.method_input_dtypes,
                    start_index_of_arguments,
                    backend=self.backend_to_test,
                    on_device=self.on_device,
                ),
            )
            ret.append(temp)
        if self.method_flags.generate_frontend_arrays:
            ret[0], ret[1] = self._args_to_frontend(
                *ret[0],
                include_derived={"tuple": True},
                **ret[1],
            )
        return ret[0], ret[1]

    def _preprocess_args(
        self,
        init_args_result,
        init_kwargs_result,
        method_args_result,
        method_kwargs_result,
    ):
        init_args, init_kwargs = self._preprocess_init_args(
            init_args_result, init_kwargs_result
        )
        method_args, method_kwargs = self._preprocess_method_args(
            method_args_result, method_kwargs_result
        )
        return init_args, init_kwargs, method_args, method_kwargs

    def _call_function(self, init_args, init_kwargs, method_args, method_kwargs):
        # determine the target frontend_fn
        frontend_fn, ins = self._get_frontend_function(
            init_args, init_kwargs, method_args, method_kwargs
        )

        if self.method_flags.generate_frontend_arrays and self.method_flags.test_trace:
            # convert the args and kwargs to ivy arrays
            method_args, method_kwargs = self._ivy.nested_map(
                (method_args, method_kwargs),
                FunctionTestCaseSubRunner._frontend_array_to_ivy,
                include_derived={"tuple": True},
            )

        # testing inplace
        if self.method_flags.inplace:
            ret = self._test_inplace(frontend_fn, ins, method_args, method_kwargs)
        else:
            with self._ivy.PreciseMode(self.method_flags.precision_mode):
                ret = frontend_fn(*method_args, **method_kwargs)

            # converting ret to ivy array
            if self.method_flags.test_trace:
                ret = self._ivy.nested_map(
                    ret, self._ivy.asarray, include_derived={"tuple": True}
                )
            else:
                # asserting the returned arrays are frontend arrays
                assert self._ivy.nested_map(
                    lambda x: (
                        FunctionTestCaseSubRunner._is_frontend_array(x)
                        if self._ivy.is_array(x)
                        else True
                    ),
                    ret,
                    shallow=False,
                ), "Frontend function returned non-frontend arrays: {}".format(ret)

                ret = self._ivy.nested_map(
                    FunctionTestCaseSubRunner._frontend_array_to_ivy,
                    ret,
                    include_derived={"tuple": True},
                )

        return self._get_results_from_ret(ret)

    def get_results(self, init_all_as_kwargs_np, method_all_as_kwargs_np):
        # preprocess init and method args and kwargs
        (
            (init_args_result, init_kwargs_result, init_total_num_arrays),
            (method_args_result, method_kwargs_result, method_total_num_arrays),
        ) = self._search_args(init_all_as_kwargs_np, method_all_as_kwargs_np)

        self._preprocess_flags(init_total_num_arrays, method_total_num_arrays)
        init_args, init_kwargs, method_args, method_kwargs = self._preprocess_args(
            init_args_result,
            init_kwargs_result,
            method_args_result,
            method_kwargs_result,
        )

        return self._call_function(init_args, init_kwargs, method_args, method_kwargs)


class GTFunctionTestCaseSubRunner(GTFrontendTestCaseSubRunner):
    def __init__(
        self,
        gt_fn_tree,
        fn_tree,
        test_flags,
        frontend,
        on_device,
    ):
        self.gt_fn_tree = gt_fn_tree
        self.fn_tree = fn_tree
        self.test_flags = test_flags
        super().__init__(
            frontend=frontend,
            on_device=on_device,
        )

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
        gt_frontend_submods, gt_fn_name = self._get_frontend_submodule()
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
                else (  # Todo: Fix this
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

    def _call_function(self, args, kwargs):
        frontend_fn = self._get_frontend_fn()
        ret = frontend_fn(*args, **kwargs)
        return self._get_results_from_ret(ret)

    def get_results(self, test_arguments):
        args_result, kwargs_result, _ = self._search_args(test_arguments)
        args, kwargs = self._preprocess_args(args_result, kwargs_result)
        return self._call_function(args, kwargs)


class GTMethodTestCaseSubRunner(GTFrontendTestCaseSubRunner):
    def __init__(
        self,
        frontend_method_data,
        frontend,
        on_device,
        init_flags,
        method_flags,
    ):
        self.frontend_method_data = frontend_method_data
        self.init_flags = init_flags
        self.method_flags = method_flags
        super().__init__(
            frontend=frontend,
            on_device=on_device,
        )

    def _get_frontend_fn(self, args_constructor, kwargs_constructor):
        frontend_creation_fn = getattr(
            importlib.import_module(self.frontend_method_data.framework_init_module),
            self.frontend_method_data.init_name,
        )
        ins_gt = frontend_creation_fn(*args_constructor, **kwargs_constructor)
        return ins_gt.__getattribute__(self.frontend_method_data.method_name)

    def _search_args(self, init_all_as_kwargs_np, method_all_as_kwargs_np):
        arg_searcher = ArgumentsSearcher(init_all_as_kwargs_np)
        init_args = arg_searcher.search_args(self.init_flags.num_positional_args)

        arg_searcher = ArgumentsSearcher(method_all_as_kwargs_np)
        method_args = arg_searcher.search_args(self.method_flags.num_positional_args)
        return init_args, method_args

    def _preprocess_args(self, init_args, init_kwargs, method_args, method_kwargs):
        # getting values from TestArgumentsSearchResult
        init_args, init_kwargs = init_args.original, init_kwargs.original
        method_args, method_kwargs = method_args.original, method_kwargs.original

        args_constructor = ivy.nested_map(
            lambda x: (
                self.frontend_config.native_array(x) if isinstance(x, np.ndarray) else x
            ),
            init_args,
            shallow=False,
        )
        kwargs_constructor = ivy.nested_map(
            lambda x: (
                self.frontend_config.native_array(x) if isinstance(x, np.ndarray) else x
            ),
            init_kwargs,
            shallow=False,
        )

        args_method = ivy.nested_map(
            lambda x: (
                self.frontend_config.native_array(x) if isinstance(x, np.ndarray) else x
            ),
            method_args,
            shallow=False,
        )
        kwargs_method = ivy.nested_map(
            lambda x: (
                self.frontend_config.native_array(x) if isinstance(x, np.ndarray) else x
            ),
            method_kwargs,
            shallow=False,
        )
        # change ivy dtypes to native dtypes
        if "dtype" in kwargs_method:
            kwargs_method["dtype"] = self.frontend_config.as_native_dtype(
                kwargs_method["dtype"]
            )

        # change ivy device to native devices
        if "device" in kwargs_method:
            kwargs_method["device"] = self.frontend_config.as_native_device(
                kwargs_method["device"]
            )
        return args_constructor, kwargs_constructor, args_method, kwargs_method

    def _call_function(
        self, args_constructor, kwargs_constructor, args_method, kwargs_method
    ):
        frontend_fn = self._get_frontend_fn(args_constructor, kwargs_constructor)
        ret = frontend_fn(*args_method, **kwargs_method)
        return self._get_results_from_ret(ret)

    def get_results(self, init_all_as_kwargs_np, method_all_as_kwargs_np):
        (
            (init_args_result, init_kwargs_result, _),
            (method_args_result, method_kwargs_result, _),
        ) = self._search_args(init_all_as_kwargs_np, method_all_as_kwargs_np)
        (
            args_constructor,
            kwargs_constructor,
            args_method,
            kwargs_method,
        ) = self._preprocess_args(
            init_args_result,
            init_kwargs_result,
            method_args_result,
            method_kwargs_result,
        )
        return self._call_function(
            args_constructor, kwargs_constructor, args_method, kwargs_method
        )
