from ivy_tests.test_ivy.pipeline.base.pipeline import Pipeline
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
from ivy_tests.test_ivy.pipeline.backend.runners import BackendFunctionTestCaseRunner


class BackendPipeline(Pipeline):
    @staticmethod
    def test_function(
        fn_name: str,
        on_device: str,
        backend_to_test: str,
        test_flags: FunctionTestFlags,
        input_dtypes,
        tolerance_dict: dict = None,
        rtol_: float = 1e-05,
        atol_: float = 1e-06,
        xs_grad_idxs=None,
        ret_grad_idxs=None,
        return_flat_np_arrays: bool = False,
        test_values: bool = True,
        **all_as_kwargs_np,
    ):
        runner = BackendFunctionTestCaseRunner(
            fn_name=fn_name,
            backend_handler=BackendPipeline.backend_handler,
            backend_to_test=backend_to_test,
            ground_truth_backend=test_flags.ground_truth_backend,
            on_device=on_device,
            test_values=test_values,
            tolerance_dict=tolerance_dict,
            rtol=rtol_,
            atol=atol_,
        )

        runner.run(input_dtypes, all_as_kwargs_np, test_flags)

    def test_method():
        pass
