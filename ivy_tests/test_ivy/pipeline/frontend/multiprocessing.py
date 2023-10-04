from .runners import (
    FrontendFunctionTestCaseRunner,
    GTFunctionTestCaseSubRunner,
    FunctionTestCaseSubRunner,
)
import ivy_tests.test_ivy.helpers.globals as t_globals
from ..base.runners import TestCaseSubRunnerResult


class FunctionTestCaseSubRunnerMP(FunctionTestCaseSubRunner):
    def _get_frontend_function(self, args, kwargs):
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
        return frontend_fn

    def _call_function(self, args, kwargs):
        # determine the target frontend_fn
        frontend_fn = self._get_frontend_function(args, kwargs)
        if self.test_flags.generate_frontend_arrays and self.test_flags.test_trace:
            # convert the args and kwargs to ivy arrays
            args, kwargs = self._ivy.nested_map(
                (args, kwargs),
                FunctionTestCaseSubRunnerMP._frontend_array_to_ivy,
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

        # testing out argument
        if self.test_flags.with_out:
            self._test_out(ret, args, kwargs, frontend_fn)

        if self.test_flags.test_trace:
            return self._get_results_from_ret(ret, raw=True), frontend_fn

        return self._get_results_from_ret(ret, raw=True), None


class GTFunctionTestCaseSubRunnerMP(GTFunctionTestCaseSubRunner):
    def _call_function(self, args, kwargs):
        frontend_fn = self._get_frontend_fn()
        ret = frontend_fn(*args, **kwargs)
        return self._get_results_from_ret(ret, raw=True)


class FrontendFunctionTestCaseRunnerMP(FrontendFunctionTestCaseRunner):
    def __init__(
        self,
        backend_handler,
        fn_tree,
        backend_to_test,
        gt_fn_tree,
        frontend,
        on_device,
        traced_fn,
        tolerance_dict,
        rtol,
        atol,
        mod_backend,
        mod_frontend,
    ):
        super().__init__(
            backend_handler,
            fn_tree,
            backend_to_test,
            gt_fn_tree,
            frontend,
            on_device,
            traced_fn,
            tolerance_dict,
            rtol,
            atol,
        )
        self.mod_backend = mod_backend
        self.mod_frontend = mod_frontend

    def _run_target(self, input_dtypes, test_arguments, test_flags):
        proc, input_queue, output_queue = self.mod_backend[self.backend_to_test]
        input_queue.put(
            (
                "_run_target_frontend",
                self.fn_tree,
                test_flags,
                self.frontend,
                self.backend_handler,
                self.on_device,
                input_dtypes,
                test_arguments,
                self.backend_to_test,
                self.traced_fn,
            )
        )
        (ret_np_flat, ret_shapes, ret_devices, ret_dtypes), traced_fn = (
            output_queue.get()
        )
        if test_flags.test_trace and self.traced_fn is None:
            t_globals.CURRENT_PIPELINE.set_traced_fn(traced_fn)
        return TestCaseSubRunnerResult(
            flatten_elements_np=ret_np_flat,
            shape=ret_shapes,
            device=ret_devices,
            dtype=ret_dtypes,
        )

    @staticmethod
    def _run_target_helper(
        fn_tree,
        test_flags,
        frontend,
        backend_handler,
        on_device,
        input_dtypes,
        test_arguments,
        backend_to_test,
        traced_fn,
    ):
        sub_runner_target = FunctionTestCaseSubRunnerMP(
            fn_tree,
            frontend,
            backend_handler,
            backend_to_test,
            on_device,
            traced_fn,
            input_dtypes,
            test_flags,
        )
        ret = sub_runner_target.get_results(test_arguments)
        sub_runner_target.exit()
        return ret

    @staticmethod
    def _run_ground_truth_helper(
        gt_fn_tree,
        fn_tree,
        test_flags,
        frontend,
        backend_handler,
        on_device,
        input_dtypes,
        test_arguments,
    ):
        sub_runner_gt = GTFunctionTestCaseSubRunnerMP(
            gt_fn_tree,
            fn_tree,
            test_flags,
            frontend,
            backend_handler,
            on_device,
        )
        ret = sub_runner_gt.get_results(test_arguments)
        sub_runner_gt.exit()
        return ret

    def _run_ground_truth(self, input_dtypes, test_arguments, test_flags):
        proc, input_queue, output_queue = self.mod_frontend[self.frontend]
        input_queue.put(
            (
                "_run_gt_frontend",
                self.gt_fn_tree,
                self.fn_tree,
                test_flags,
                self.frontend,
                self.backend_handler,
                self.on_device,
                input_dtypes,
                test_arguments,
            )
        )
        ret_np_flat, ret_shapes, ret_devices, ret_dtypes = output_queue.get()
        return TestCaseSubRunnerResult(
            flatten_elements_np=ret_np_flat,
            shape=ret_shapes,
            device=ret_devices,
            dtype=ret_dtypes,
        )
