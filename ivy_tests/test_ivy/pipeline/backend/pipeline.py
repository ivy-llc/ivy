from ivy_tests.test_ivy.pipeline.base.pipeline import Pipeline
from ivy_tests.test_ivy.pipeline.c_backend_handler import WithBackendHandler
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
from ivy_tests.test_ivy.pipeline.backend.runners import BackendTestCaseRunner


class BackendPipeline(Pipeline):
    def __init__(self):
        self._backend_handler = WithBackendHandler()

    @property
    def backend_handler(self):
        return self.backend_handler

    @staticmethod
    def test_function(
        fn_name: str,
        on_device: str,
        backend_to_test: str,
        test_flags: FunctionTestFlags,
        input_dtypes,
        rtol_: float = 1e-05,
        atol_: float = 1e-06,
        xs_grad_idxs=None,
        ret_grad_idxs=None,
        return_flat_np_arrays: bool = False,
        test_values: bool = True,
        **all_as_kwargs_np,
    ):
        runner = BackendTestCaseRunner(
            fn_name=fn_name,
            backend_handler=WithBackendHandler(),
            backend_to_test=backend_to_test,
            ground_truth_backend=test_flags.ground_truth_backend,
            on_device=on_device,
            rtol=rtol_,
            atol=atol_,
        )

        runner.run(input_dtypes, all_as_kwargs_np, test_flags)

    def test_method():
        pass
