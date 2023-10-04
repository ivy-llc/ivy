from ivy_tests.test_ivy.pipeline.base.pipeline import Pipeline
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
from ivy_tests.test_ivy.pipeline.frontend.multiprocessing import (
    FrontendTestCaseRunnerMP,
)
from ivy_tests.test_ivy.pipeline.frontend.runners import FrontendTestCaseRunner


class FrontendPipeline(Pipeline):
    mod_frontend = {
        "tensorflow": None,
        "numpy": None,
        "jax": None,
        "torch": None,
        "mindspore": None,
        "scipy": None,
        "paddle": None,
        "onnx": None,
        "pandas": None,
        "xgboost": None,
        "sklearn": None,
        "mxnet": None,
    }

    @classmethod
    def set_mod_frontend(cls, mod_frontend):
        for key in mod_frontend:
            cls.mod_frontend[key] = mod_frontend[key]

    @classmethod
    def unset_mod(cls):
        for key in cls.mod_backend:
            cls.mod_backend[key] = None
        for key in cls.mod_frontend:
            cls.mod_frontend[key] = None

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
        if not FrontendPipeline.multiprocessing_flag:
            runner = FrontendTestCaseRunner(
                fn_tree=fn_tree,
                backend_handler=FrontendPipeline.backend_handler,
                backend_to_test=backend_to_test,
                gt_fn_tree=gt_fn_tree,
                frontend=frontend,
                on_device=on_device,
                traced_fn=FrontendPipeline.traced_fn,
                test_values=test_values,
                tolerance_dict=tolerance_dict,
                rtol=rtol_,
                atol=atol_,
            )
        else:
            runner = FrontendTestCaseRunnerMP(
                fn_tree=fn_tree,
                backend_handler=FrontendPipeline.backend_handler,
                backend_to_test=backend_to_test,
                gt_fn_tree=gt_fn_tree,
                frontend=frontend,
                on_device=on_device,
                traced_fn=FrontendPipeline.traced_fn,
                test_values=test_values,
                tolerance_dict=tolerance_dict,
                rtol=rtol_,
                atol=atol_,
                mod_backend=FrontendPipeline.mod_backend,
                mod_frontend=FrontendPipeline.mod_frontend,
            )
        runner.run(input_dtypes, all_as_kwargs_np, test_flags)

    def test_method(self):
        pass
