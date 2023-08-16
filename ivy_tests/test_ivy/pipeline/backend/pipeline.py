from ivy_tests.test_ivy.pipeline.base.pipeline import Pipeline
from ivy_tests.test_ivy.pipeline.c_backend_handler import WithBackendHandler
from ivy_tests.test_ivy.helpers.test_parameter_flags import FunctionTestFlags


class BackendPipeline(Pipeline):
    def __init__(self):
        self._backend_handler = WithBackendHandler()

    @property
    def backend_handler(self):
        return self.backend_handler

    def test_function(
        *,
        fn_name: str,
        on_device: str,
        backend_to_test: str,
        test_flags: FunctionTestFlags,
        input_dtypes,
        rtol_: float = None,
        atol_: float = 1e-06,
        xs_grad_idxs=None,
        ret_grad_idxs=None,
        return_flat_np_arrays: bool = False,
        test_values: bool = True,
        **all_as_kwargs_np,
    ):
        pass

    def test_method():
        pass
