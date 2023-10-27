# global
from typing import List, Union

# local
import ivy
from ivy_tests.test_ivy.pipeline.base.pipeline import Pipeline
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags
from ivy_tests.test_ivy.pipeline.backend.runners import (
    BackendFunctionTestCaseRunner,
    BackendMethodTestCaseRunner,
)


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
            traced_fn=BackendPipeline.traced_fn,
        )

        runner.run(input_dtypes, all_as_kwargs_np, test_flags)

    @staticmethod
    def test_method(
        *,
        init_input_dtypes: List[ivy.Dtype] = None,
        method_input_dtypes: List[ivy.Dtype] = None,
        init_all_as_kwargs_np: dict = None,
        method_all_as_kwargs_np: dict = None,
        init_flags,
        method_flags,
        class_name: str,
        method_name: str = "__call__",
        rtol_: float = None,
        atol_: float = 1e-06,
        tolerance_dict=None,
        test_values: Union[bool, str] = True,
        test_gradients: bool = False,
        xs_grad_idxs=None,
        ret_grad_idxs=None,
        backend_to_test: str,
        ground_truth_backend: str,
        on_device: str,
        return_flat_np_arrays: bool = False,
    ):
        runner = BackendMethodTestCaseRunner(
            class_name=class_name,
            method_name=method_name,
            backend_handler=BackendPipeline.backend_handler,
            traced_fn=BackendPipeline.traced_fn,
            backend_to_test=backend_to_test,
            ground_truth_backend=ground_truth_backend,
            on_device=on_device,
            test_values=test_values,
            tolerance_dict=tolerance_dict,
            test_gradients=test_gradients,
            xs_grad_idxs=xs_grad_idxs,
            ret_grad_idxs=ret_grad_idxs,
            return_flat_np_arrays=return_flat_np_arrays,
            rtol_=rtol_,
            atol_=atol_,
        )
        runner.run(
            init_input_dtypes=init_input_dtypes,
            method_input_dtypes=method_input_dtypes,
            init_flags=init_flags,
            method_flags=method_flags,
            init_all_as_kwargs_np=init_all_as_kwargs_np,
            method_all_as_kwargs_np=method_all_as_kwargs_np,
        )
