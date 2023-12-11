from typing import List, Union

import ivy
from ivy_tests.test_ivy.pipeline.base.pipeline import Pipeline
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
from ivy_tests.test_ivy.pipeline.frontend.runners import (
    FrontendFunctionTestCaseRunner,
    FrontendMethodTestCaseRunner,
)

# from ivy_tests.test_ivy.helpers import FrontendMethodData


class FrontendPipeline(Pipeline):
    @staticmethod
    def test_function(
        *,
        fn_tree: str,
        on_device: str,
        backend_to_test: str,
        test_flags: FunctionTestFlags,
        input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
        frontend: str,
        gt_fn_tree: str = None,
        rtol_: float = None,
        atol_: float = 1e-06,
        tolerance_dict: dict = None,
        test_values: bool = True,
        **all_as_kwargs_np,
    ):
        runner = FrontendFunctionTestCaseRunner(
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
        runner.run(input_dtypes, all_as_kwargs_np, test_flags)

    @staticmethod
    def test_method(
        *,
        init_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]] = None,
        method_input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
        init_flags,
        method_flags,
        init_all_as_kwargs_np: dict = None,
        method_all_as_kwargs_np: dict,
        frontend: str,
        frontend_method_data,
        backend_to_test: str,
        on_device,
        rtol_: float = None,
        atol_: float = 1e-06,
        tolerance_dict: dict = None,
        test_values: Union[bool, str] = True,
    ):
        runner = FrontendMethodTestCaseRunner(
            frontend=frontend,
            frontend_method_data=frontend_method_data,
            backend_to_test=backend_to_test,
            backend_handler=FrontendPipeline.backend_handler,
            on_device=on_device,
            traced_fn=FrontendPipeline.traced_fn,
            rtol_=rtol_,
            atol_=atol_,
            tolerance_dict=tolerance_dict,
            test_values=test_values,
        )
        runner.run(
            init_input_dtypes=init_input_dtypes,
            method_input_dtypes=method_input_dtypes,
            init_flags=init_flags,
            method_flags=method_flags,
            init_all_as_kwargs_np=init_all_as_kwargs_np,
            method_all_as_kwargs_np=method_all_as_kwargs_np,
        )
