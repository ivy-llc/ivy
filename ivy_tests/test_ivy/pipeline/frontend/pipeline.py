from ivy_tests.test_ivy.pipeline.base.pipeline import Pipeline
from ivy_tests.test_ivy.pipeline.c_backend_handler import WithBackendHandler
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
from ivy_tests.test_ivy.pipeline.frontend.runners import FrontendTestCaseRunner


class FrontendPipeline(Pipeline):
    traced_fn = None

    def __init__(self):
        self._backend_handler = WithBackendHandler()

    @property
    def backend_handler(self):
        return self.backend_handler

    @classmethod
    def set_traced_fn(cls, fn):
        cls.traced_fn = fn

    @staticmethod
    def test_function(
        fn_tree: str,
        on_device: str,
        backend_to_test: str,
        test_flags: FunctionTestFlags,
        input_dtypes,
        frontend: str,
        gt_fn_tree: str = None,
        rtol_: float = None,
        atol_: float = 1e-06,
        tolerance_dict: dict = None,
        test_values: bool = True,
        **all_as_kwargs_np,
    ):
        runner = FrontendTestCaseRunner(
            fn_tree=fn_tree,
            backend_handler=WithBackendHandler(),
            backend_to_test=backend_to_test,
            gt_fn_tree=gt_fn_tree,
            frontend=frontend,
            on_device=on_device,
            traced_fn=FrontendPipeline.traced_fn,
            tolerance_dict=tolerance_dict,
            rtol=rtol_,
            atol=atol_,
        )

        runner.run(input_dtypes, all_as_kwargs_np, test_flags)

    def test_method(self):
        pass
