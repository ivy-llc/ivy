from .runners import (
    FrontendTestCaseRunner,
    GTFunctionTestCaseSubRunner,
    FunctionTestCaseSubRunner,
)


class FrontendTestCaseRunnerMP(FrontendTestCaseRunner):
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
            ("_run_target_frontend", self, input_dtypes, test_arguments, test_flags)
        )
        ret = output_queue.get()
        return ret

    def _run_ground_truth_helper(self, input_dtypes, test_arguments, test_flags):
        sub_runner_gt = GTFunctionTestCaseSubRunner(
            self.gt_fn_tree,
            self.fn_tree,
            test_flags,
            self.frontend,
            self.backend_handler,
            self.on_device,
        )
        ret = sub_runner_gt.get_results(test_arguments)
        sub_runner_gt.exit()
        return ret

    def _run_target_helper(self, input_dtypes, test_arguments, test_flags):
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
        proc, input_queue, output_queue = self.mod_frontend[self.frontend]
        input_queue.put(
            ("_run_gt_frontend", self, input_dtypes, test_arguments, test_flags)
        )
        ret = output_queue.get()
        return ret
