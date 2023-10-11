# global
import numpy as np

# local
from .runners import BackendMethodTestCaseRunner, MethodTestCaseSubRunner
import ivy_tests.test_ivy.helpers.globals as t_globals
from ..base.runners import TestCaseSubRunnerResult


class MethodTestCaseSubRunnerMP(MethodTestCaseSubRunner):
    def __init__(
        self,
        class_name,
        method_name,
        backend_handler,
        backend,
        on_device,
        traced_fn,
        v_np,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        is_gt,
    ):
        super().__init__(
            class_name=class_name,
            method_name=method_name,
            backend_handler=backend_handler,
            backend=backend,
            on_device=on_device,
            traced_fn=traced_fn,
            init_input_dtypes=init_input_dtypes,
            method_input_dtypes=method_input_dtypes,
            init_flags=init_flags,
            method_flags=method_flags,
            is_gt=is_gt,
        )
        self.v_np = v_np

    def _get_v_np(self, ins, args_constructor, kwargs_constructor, kwargs_method):
        v_np = None
        if isinstance(ins, self._ivy.Module):
            if self.init_flags.init_with_v:
                v = self._ivy.Container(
                    ins._create_variables(
                        device=self.on_device, dtype=self.method_input_dtypes[0]
                    )
                )
                ins = self._ivy.__dict__[self.class_name](
                    *args_constructor, **kwargs_constructor, v=v
                )
            v = ins.__getattribute__("v")
            v_np = v.cont_map(
                lambda x, kc: self._ivy.to_numpy(x) if self._ivy.is_array(x) else x
            )
            if self.method_flags.method_with_v:
                kwargs_method = dict(**kwargs_method, v=v)
        return kwargs_method, v_np

    def _get_function(self, init_args, init_kwargs, method_args, method_kwargs):
        if self.traced_fn is not None and self.method_flags.test_trace:
            return self.traced_fn

        ins = self._ivy.__dict__[self.class_name](*init_args, **init_kwargs)
        backend_fn = ins.__getattribute__(self.method_name)
        v_np = None
        if not self.is_gt:
            method_kwargs, v_np = self._get_v_np(
                ins, init_args, init_kwargs, method_kwargs
            )
        else:
            if isinstance(ins, self._ivy.Module):
                v_gt = self.v_np.cont_map(
                    lambda x, kc: (
                        self._ivy.asarray(x) if isinstance(x, np.ndarray) else x
                    )
                )
                method_kwargs = dict(**method_kwargs, v=v_gt)
                v_np = None
        if self.traced_fn is None and not self.is_gt:
            backend_fn = self.trace_if_required(
                backend_fn,
                test_trace=self.method_flags.test_trace,
                args=method_args,
                kwargs=method_kwargs,
            )
        return backend_fn, method_kwargs, v_np

    def _call_function(self, init_args, init_kwargs, method_args, method_kwargs):
        # determine the target backend_fn
        backend_fn, method_kwargs, v_np = self._get_function(
            init_args, init_kwargs, method_args, method_kwargs
        )

        with self._ivy.PreciseMode(self.method_flags.precision_mode):
            ret = backend_fn(*method_args, **method_kwargs)

        def map_fn(x):
            if self._ivy.is_native_array(x) or isinstance(x, np.ndarray):
                return self._ivy.to_ivy(x)
            return x

        ret = self._ivy.nested_map(map_fn, ret, include_derived={"tuple": True})
        if self.method_flags.test_trace and not self.is_gt:
            return (
                self._get_results_from_ret(ret, store_types=True, raw=True),
                backend_fn,
                v_np,
            )
        return self._get_results_from_ret(ret, store_types=True, raw=True), None, v_np


class BackendMethodTestCaseRunnerMP(BackendMethodTestCaseRunner):
    def __init__(
        self,
        *,
        class_name,
        method_name,
        rtol_,
        atol_,
        tolerance_dict,
        test_values,
        test_gradients,
        xs_grad_idxs,
        ret_grad_idxs,
        backend_to_test,
        ground_truth_backend,
        on_device,
        return_flat_np_arrays,
        backend_handler,
        traced_fn,
        mod_backend,
    ):
        super().__init__(
            class_name=class_name,
            method_name=method_name,
            backend_handler=backend_handler,
            traced_fn=traced_fn,
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
        self.mod_backend = mod_backend

    def _run_ground_truth(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        proc, input_queue, output_queue = self.mod_backend[self.ground_truth_backend]
        input_queue.put(
            (
                "_method_ground_truth",
                self.class_name,
                self.method_name,
                self.backend_handler,
                self.ground_truth_backend,
                self.on_device,
                self.traced_fn,
                MethodTestCaseSubRunner.v_np,
                init_input_dtypes,
                method_input_dtypes,
                init_flags,
                method_flags,
                init_all_as_kwargs_np,
                method_all_as_kwargs_np,
            )
        )
        (ret_np_flat, ret_shapes, ret_devices, ret_dtypes, ret_types), _, v_np = (
            output_queue.get()
        )
        MethodTestCaseSubRunner.v_np = v_np
        return TestCaseSubRunnerResult(
            flatten_elements_np=ret_np_flat,
            shape=ret_shapes,
            device=ret_devices,
            dtype=ret_dtypes,
            type=ret_types,
        )

    @staticmethod
    def _run_ground_truth_helper(
        class_name,
        method_name,
        backend_handler,
        ground_truth_backend,
        on_device,
        traced_fn,
        v_np,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        sub_runner_target = MethodTestCaseSubRunnerMP(
            class_name,
            method_name,
            backend_handler,
            ground_truth_backend,
            on_device,
            traced_fn,
            v_np,
            init_input_dtypes,
            method_input_dtypes,
            init_flags,
            method_flags,
            is_gt=True,
        )
        ret = sub_runner_target.get_results(
            init_all_as_kwargs_np, method_all_as_kwargs_np
        )
        sub_runner_target.exit()
        return ret

    def _run_target(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        proc, input_queue, output_queue = self.mod_backend[self.backend_to_test]
        input_queue.put(
            (
                "_method_backend",
                self.class_name,
                self.method_name,
                self.backend_handler,
                self.ground_truth_backend,
                self.on_device,
                self.traced_fn,
                MethodTestCaseSubRunner.v_np,
                init_input_dtypes,
                method_input_dtypes,
                init_flags,
                method_flags,
                init_all_as_kwargs_np,
                method_all_as_kwargs_np,
            )
        )
        (
            (ret_np_flat, ret_shapes, ret_devices, ret_dtypes, ret_types),
            traced_fn,
            v_np,
        ) = output_queue.get()
        MethodTestCaseSubRunner.v_np = v_np
        if method_flags.test_trace and self.traced_fn is None:
            t_globals.CURRENT_PIPELINE.set_traced_fn(traced_fn)
        return TestCaseSubRunnerResult(
            flatten_elements_np=ret_np_flat,
            shape=ret_shapes,
            device=ret_devices,
            dtype=ret_dtypes,
            type=ret_types,
        )

    @staticmethod
    def _run_target_helper(
        class_name,
        method_name,
        backend_handler,
        ground_truth_backend,
        on_device,
        traced_fn,
        v_np,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        sub_runner_target = MethodTestCaseSubRunnerMP(
            class_name,
            method_name,
            backend_handler,
            ground_truth_backend,
            on_device,
            traced_fn,
            v_np,
            init_input_dtypes,
            method_input_dtypes,
            init_flags,
            method_flags,
            is_gt=False,
        )
        ret = sub_runner_target.get_results(
            init_all_as_kwargs_np, method_all_as_kwargs_np
        )
        sub_runner_target.exit()
        return ret
