from ivy_tests.test_ivy.pipeline.base.argument_searcher import (
    TestArgumentsSearchResult,
    ArgumentsSearcher,
)
import numpy as np
import inspect
import copy
from ivy_tests.test_ivy.pipeline.base.runners import (
    TestCaseRunner,
    TestCaseSubRunner,
    TestCaseSubRunnerResult,
)
from ivy_tests.test_ivy.pipeline.base.assertion_checker import AssertionChecker
import ivy_tests.test_ivy.helpers.globals as t_globals


class BackendTestCaseRunner(TestCaseRunner):
    def __init__(
        self,
        backend_handler,
        backend_to_test,
        ground_truth_backend,
        on_device,
        tolerance_dict,
        rtol_,
        atol_,
        test_values,
        traced_fn,
    ):
        self.backend_handler = backend_handler
        self.backend_to_test = backend_to_test
        self.ground_truth_backend = ground_truth_backend
        self.on_device = on_device
        self.tolerance_dict = tolerance_dict
        self.rtol = rtol_
        self.atol = atol_
        self.test_values = test_values
        self.traced_fn = traced_fn

    def _check_assertions(self, target_results, ground_truth_results):
        if self.test_values:
            assertion_checker = AssertionChecker(
                target_results,
                ground_truth_results,
                self.backend_to_test,
                self.ground_truth_backend,
                self.tolerance_dict,
                self.rtol,
                self.atol,
            )
            assertion_checker.check_assertions()


class FunctionTestCaseSubRunner(TestCaseSubRunner):
    def __init__(
        self,
        fn_name,
        backend_handler,
        backend,
        device,
        input_dtypes,
        test_flags,
        traced_fn,
    ):
        self.fn_name = fn_name
        self._backend_handler = backend_handler
        self.backend = backend
        self.on_device = device
        self.__ivy = self._backend_handler.set_backend(backend)
        self.test_flags = test_flags
        self.input_dtypes = input_dtypes
        self.traced_fn = traced_fn

    @property
    def backend_handler(self):
        return self._backend_handler

    @property
    def _ivy(self):
        return self.__ivy

    def _find_instance_in_args(self, args, array_indices, mask):
        """
        Find the first element in the arguments that is considered to be an instance of
        Array or Container class.

        Parameters
        ----------
        args
            Arguments to iterate over
        array_indices
            Indices of arrays that exists in the args
        mask
            Boolean mask for whether the corrseponding element in (args) has a
            generated test_flags.native_array as False or test_flags.container as
            true

        Returns
        -------
            First found instance in the arguments and the updates arguments not
            including the instance
        """
        i = 0
        for i, a in enumerate(mask):
            if a:
                break
        instance_idx = array_indices[i]
        instance = self._ivy.index_nest(args, instance_idx)
        new_args = self._ivy.copy_nest(args, to_mutable=False)
        self._ivy.prune_nest_at_index(new_args, instance_idx)
        return instance, new_args

    def _preprocess_flags(self, total_num_arrays):
        # Make all array-specific test flags and dtypes equal in length
        if len(self.input_dtypes) < total_num_arrays:
            self.input_dtypes = [self.input_dtypes[0] for _ in range(total_num_arrays)]
        if len(self.test_flags.as_variable) < total_num_arrays:
            self.test_flags.as_variable = [
                self.test_flags.as_variable[0] for _ in range(total_num_arrays)
            ]
        if len(self.test_flags.native_arrays) < total_num_arrays:
            self.test_flags.native_arrays = [
                self.test_flags.native_arrays[0] for _ in range(total_num_arrays)
            ]
        if len(self.test_flags.container) < total_num_arrays:
            self.test_flags.container = [
                self.test_flags.container[0] for _ in range(total_num_arrays)
            ]
        # Update variable flags to be compatible with float dtype and with_out args
        self.test_flags.as_variable = [
            v if self._ivy.is_float_dtype(d) and not self.test_flags.with_out else False
            for v, d in zip(self.test_flags.as_variable, self.input_dtypes)
        ]

        # TODO this is not ideal, modifying Hypothesis generated value
        # May result in weird bugs. Should instead update strategies to
        # Not generate this in first place.
        # update instance_method flag to only be considered if the
        self.test_flags.instance_method = self.test_flags.instance_method and (
            not self.test_flags.native_arrays[0] or self.test_flags.container[0]
        )
        return self.test_flags

    def _compile_if_required(self, fn, args=None, kwargs=None):
        if self.test_flags.test_compile:
            fn = self._ivy.compile(fn, args=args, kwargs=kwargs)
        return fn

    def _flatten_and_to_np(self, *, ret):
        ret_flat = self._flatten(ret=ret)
        ret = [self._ivy.to_numpy(x) for x in ret_flat]
        return ret

    def _get_ret(self, fn, *args, **kwargs):
        """
        Run func with args and kwargs.

        Return the result along with its flattened version.
        """
        with self._ivy.PreciseMode(self.test_flags.precision_mode):
            ret = fn(*args, **kwargs)

        def map_fn(x):
            if self._ivy.is_native_array(x) or isinstance(x, np.ndarray):
                return self._ivy.to_ivy(x)
            return x

        return self._ivy.nested_map(map_fn, ret, include_derived={"tuple": True})

    def _test_out(self, target_fn, ret_from_target, *args, **kwargs):

        # If function doesn't have an out argument but an out argument is given
        # or a test with out flag is True
        if (
            "out" in kwargs or self.test_flags.with_out
        ) and "out" not in inspect.signature(
            getattr(self._ivy, self.fn_name)
        ).parameters:
            raise Exception(f"Function {self.fn_name} does not have an out parameter")

        # TODO should be moved outside of get_results
        # Assert indices of return if the indices of the out array provided
        if self.test_flags.with_out and not self.test_flags.test_trace:
            test_ret = (
                ret_from_target[getattr(self._ivy.__dict__[self.fn_name], "out_index")]
                if hasattr(self._ivy.__dict__[self.fn_name], "out_index")
                else ret_from_target
            )
            out = self._ivy.nested_map(
                self._ivy.zeros_like, test_ret, to_mutable=True, include_derived=True
            )
            ret_from_target = self._get_ret(
                target_fn,
                *args,
                **kwargs,
                out=out,
            )
            test_ret = (
                ret_from_target[getattr(self._ivy.__dict__[self.fn_name], "out_index")]
                if hasattr(self._ivy.__dict__[self.fn_name], "out_index")
                else ret_from_target
            )
            assert not self._ivy.nested_any(
                self._ivy.nested_multi_map(lambda x, _: x[0] is x[1], [test_ret, out]),
                lambda x: not x,
            ), "the array in out argument does not contain same value as the returned"
            if not max(self.test_flags.container) and self._ivy.native_inplace_support:
                # these backends do not always support native inplace updates
                assert not self._ivy.nested_any(
                    self._ivy.nested_multi_map(
                        lambda x, _: x[0].data is x[1].data, [test_ret, out]
                    ),
                    lambda x: not x,
                ), (
                    "the array in out argument does not contain same value as the"
                    " returned"
                )

    def _get_target_fn(self, args, kwargs):
        if self.test_flags.instance_method:
            # TODO all this argument handling should be moved to _preprocess_args
            array_or_container_mask = [
                (not native_flag) or container_flag
                for native_flag, container_flag in zip(
                    self.test_flags.native_arrays, self.test_flags.container
                )
            ]

            # Boolean mask for args and kwargs True if an entry's
            # test Array flag is True or test Container flag is true
            args_instance_mask = array_or_container_mask[
                : self.test_flags.num_positional_args
            ]
            kwargs_instance_mask = array_or_container_mask[
                self.test_flags.num_positional_args :
            ]

            if any(args_instance_mask):
                instance, args = self._find_instance_in_args(
                    args, arrays_args_indices, args_instance_mask  # noqa: F821
                )
            else:
                instance, kwargs = self._find_instance_in_args(
                    kwargs, arrays_kwargs_indices, kwargs_instance_mask  # noqa: F821
                )

            if self.test_flags.test_trace and self.traced_fn is not None:
                args = [instance, *args]
                target_fn = self.traced_fn
                return target_fn, args

            if self.test_flags.test_trace and self.traced_fn is None:
                target_fn = lambda instance, *args, **kwargs: instance.__getattribute__(
                    self.fn_name
                )(*args, **kwargs)
                args = [instance, *args]
            else:
                target_fn = instance.__getattribute__(self.fn_name)
        else:
            target_fn = self._ivy.__dict__[self.fn_name]

        target_fn = self.trace_if_required(
            target_fn, test_trace=self.test_flags.test_trace, args=args, kwargs=kwargs
        )
        return target_fn, args

    def _search_args(self, all_as_kwargs_np):
        # init args searching
        arg_searcher = ArgumentsSearcher(all_as_kwargs_np)
        return arg_searcher.search_args(self.test_flags.num_positional_args)

    def _preprocess_args(
        self,
        args_result: TestArgumentsSearchResult,
        kwargs_result: TestArgumentsSearchResult,
    ):
        """
        Create arguments and keyword-arguments for the function to test.

        Returns
        -------
        Backend specific arguments, keyword-arguments
        """
        ret = []
        for result, start_index_of_arguments in zip(
            [args_result, kwargs_result], [0, len(args_result.values)]
        ):
            temp = self._ivy.copy_nest(result.original, to_mutable=False)
            self._ivy.set_nest_at_indices(
                temp,
                result.indices,
                self.test_flags.apply_flags(
                    result.values,
                    self.input_dtypes,
                    start_index_of_arguments,
                    backend=self.backend,
                    on_device=self.on_device,
                ),
            )
            ret.append(temp)
        return ret[0], ret[1]

    def _call_function(self, args, kwargs):
        target_fn, args = self._get_target_fn(args=args, kwargs=kwargs)

        # Make copy of arguments for functions that might use inplace update by default
        copy_args = copy.deepcopy(args)
        copy_kwargs = copy.deepcopy(kwargs)

        ret_from_target = self._get_ret(
            target_fn,
            *copy_args,
            **copy_kwargs,
        )

        # TODO move from here to assertions
        assert self._ivy.nested_map(
            lambda x: self._ivy.is_ivy_array(x) if self._ivy.is_array(x) else True,
            ret_from_target,
        ), "Ivy function returned non-ivy arrays: {}".format(ret_from_target)

        self._test_out(target_fn, ret_from_target, *args, **kwargs)

        return self._get_results_from_ret(ret_from_target)

    def get_results(self, test_arguments):
        args_result, kwargs_result, total_num_arrays = self._search_args(test_arguments)
        self._preprocess_flags(total_num_arrays)
        args, kwargs = self._preprocess_args(args_result, kwargs_result)

        return self._call_function(args, kwargs)


class MethodTestCaseSubRunner(TestCaseSubRunner):
    v_np = None

    def __init__(
        self,
        class_name,
        method_name,
        backend_handler,
        backend,
        on_device,
        traced_fn,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        is_gt,
    ):
        self.class_name = class_name
        self.method_name = method_name
        self._backend_handler = backend_handler
        self.backend = backend
        self.on_device = on_device
        self.__ivy = self._backend_handler.set_backend(backend)
        self.traced_fn = traced_fn
        self.init_input_dtypes = init_input_dtypes
        self.method_input_dtypes = method_input_dtypes
        self.init_flags = init_flags
        self.method_flags = method_flags
        self.is_gt = is_gt

    @property
    def backend_handler(self):
        return self._backend_handler

    @property
    def _ivy(self):
        return self.__ivy

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
        MethodTestCaseSubRunner.v_np = v_np
        return kwargs_method

    def _get_function(self, init_args, init_kwargs, method_args, method_kwargs):
        if self.traced_fn is not None and self.method_flags.test_trace:
            return self.traced_fn

        ins = self._ivy.__dict__[self.class_name](*init_args, **init_kwargs)
        backend_fn = ins.__getattribute__(self.method_name)
        if not self.is_gt:
            method_kwargs = self._get_v_np(ins, init_args, init_kwargs, method_kwargs)
        else:
            if isinstance(ins, self._ivy.Module):
                v_gt = MethodTestCaseSubRunner.v_np.cont_map(
                    lambda x, kc: (
                        self._ivy.asarray(x) if isinstance(x, np.ndarray) else x
                    )
                )
                method_kwargs = dict(**method_kwargs, v=v_gt)
                MethodTestCaseSubRunner.v_np = None
        if self.traced_fn is None and not self.is_gt:
            backend_fn = self.trace_if_required(
                backend_fn,
                test_trace=self.method_flags.test_trace,
                args=method_args,
                kwargs=method_kwargs,
            )
            t_globals.CURRENT_PIPELINE.set_traced_fn(backend_fn)
        return backend_fn, method_kwargs

    def _search_args(self, init_all_as_kwargs_np, method_all_as_kwargs_np):
        # init args searching
        arg_searcher = ArgumentsSearcher(init_all_as_kwargs_np)
        init_args = arg_searcher.search_args(self.init_flags.num_positional_args)

        # method args searching
        arg_searcher = ArgumentsSearcher(method_all_as_kwargs_np)
        method_args = arg_searcher.search_args(self.method_flags.num_positional_args)

        return init_args, method_args

    def _preprocess_init_flags(self, init_total_num_arrays):
        # Make all array-specific test flags and dtypes equal in length
        if len(self.init_input_dtypes) < init_total_num_arrays:
            self.init_input_dtypes = [
                self.init_input_dtypes[0] for _ in range(init_total_num_arrays)
            ]
        if len(self.init_flags.as_variable) < init_total_num_arrays:
            self.init_flags.as_variable = [
                self.init_flags.as_variable[0] for _ in range(init_total_num_arrays)
            ]
        if len(self.init_flags.native_arrays) < init_total_num_arrays:
            self.init_flags.native_arrays = [
                self.init_flags.native_arrays[0] for _ in range(init_total_num_arrays)
            ]

        self.init_flags.as_variable = [
            v if self._ivy.is_float_dtype(d) else False
            for v, d in zip(self.init_flags.as_variable, self.init_input_dtypes)
        ]

    def _preprocess_method_flags(self, method_total_num_arrays):
        # Make all array-specific test flags and dtypes equal in length
        if len(self.method_input_dtypes) < method_total_num_arrays:
            self.method_input_dtypes = [
                self.method_input_dtypes[0] for _ in range(method_total_num_arrays)
            ]
        if len(self.method_flags.as_variable) < method_total_num_arrays:
            self.method_flags.as_variable = [
                self.method_flags.as_variable[0] for _ in range(method_total_num_arrays)
            ]
        if len(self.method_flags.native_arrays) < method_total_num_arrays:
            self.method_flags.native_arrays = [
                self.method_flags.native_arrays[0]
                for _ in range(method_total_num_arrays)
            ]

        self.method_flags.as_variable = [
            v if self._ivy.is_float_dtype(d) else False
            for v, d in zip(self.method_flags.as_variable, self.method_input_dtypes)
        ]

    def _preprocess_flags(self, init_total_num_arrays, method_total_num_arrays):
        self._preprocess_init_flags(init_total_num_arrays)
        self._preprocess_method_flags(method_total_num_arrays)

    def _preprocess_init_args(self, init_args_result, init_kwargs_result):
        ret = []
        for result, start_index_of_arguments in zip(
            [init_args_result, init_kwargs_result], [0, len(init_args_result.values)]
        ):
            temp = self._ivy.copy_nest(result.original, to_mutable=False)
            self._ivy.set_nest_at_indices(
                temp,
                result.indices,
                self.init_flags.apply_flags(
                    result.values,
                    self.init_input_dtypes,
                    start_index_of_arguments,
                    backend=self.backend,
                    on_device=self.on_device,
                ),
            )
            ret.append(temp)

        return ret[0], ret[1]

    def _preprocess_method_args(self, method_args_result, method_kwargs_result):
        ret = []
        for result, start_index_of_arguments in zip(
            [method_args_result, method_kwargs_result],
            [0, len(method_args_result.values)],
        ):
            temp = self._ivy.copy_nest(result.original, to_mutable=False)
            self._ivy.set_nest_at_indices(
                temp,
                result.indices,
                self.method_flags.apply_flags(
                    result.values,
                    self.method_input_dtypes,
                    start_index_of_arguments,
                    backend=self.backend,
                    on_device=self.on_device,
                ),
            )
            ret.append(temp)
        return ret[0], ret[1]

    def _preprocess_args(
        self,
        init_args_result,
        init_kwargs_result,
        method_args_result,
        method_kwargs_result,
    ):
        init_args, init_kwargs = self._preprocess_init_args(
            init_args_result, init_kwargs_result
        )
        method_args, method_kwargs = self._preprocess_method_args(
            method_args_result, method_kwargs_result
        )
        return init_args, init_kwargs, method_args, method_kwargs

    def _call_function(self, init_args, init_kwargs, method_args, method_kwargs):
        # determine the target backend_fn
        backend_fn, method_kwargs = self._get_function(
            init_args, init_kwargs, method_args, method_kwargs
        )

        with self._ivy.PreciseMode(self.method_flags.precision_mode):
            ret = backend_fn(*method_args, **method_kwargs)

        def map_fn(x):
            if self._ivy.is_native_array(x) or isinstance(x, np.ndarray):
                return self._ivy.to_ivy(x)
            return x

        ret = self._ivy.nested_map(map_fn, ret, include_derived={"tuple": True})
        return self._get_results_from_ret(ret, store_types=True)

    def get_results(self, init_all_as_kwargs_np, method_all_as_kwargs_np):
        # preprocess init and method args and kwargs
        (
            (init_args_result, init_kwargs_result, init_total_num_arrays),
            (method_args_result, method_kwargs_result, method_total_num_arrays),
        ) = self._search_args(init_all_as_kwargs_np, method_all_as_kwargs_np)

        self._preprocess_flags(init_total_num_arrays, method_total_num_arrays)
        init_args, init_kwargs, method_args, method_kwargs = self._preprocess_args(
            init_args_result,
            init_kwargs_result,
            method_args_result,
            method_kwargs_result,
        )

        return self._call_function(init_args, init_kwargs, method_args, method_kwargs)


class BackendFunctionTestCaseRunner(BackendTestCaseRunner):
    def __init__(
        self,
        backend_handler,
        fn_name,
        backend_to_test,
        ground_truth_backend,
        on_device,
        tolerance_dict,
        test_values,
        rtol,
        atol,
        traced_fn,
    ):
        self.fn_name = fn_name
        super().__init__(
            backend_handler,
            backend_to_test,
            ground_truth_backend,
            on_device,
            tolerance_dict,
            rtol,
            atol,
            test_values,
            traced_fn,
        )

    def _run_target(self, input_dtypes, test_arguments, test_flags):
        sub_runner_target = FunctionTestCaseSubRunner(
            self.fn_name,
            self.backend_handler,
            self.backend_to_test,
            self.on_device,
            input_dtypes,
            test_flags,
            self.traced_fn,
        )
        results = sub_runner_target.get_results(test_arguments)
        sub_runner_target.exit()
        return results

    def _run_ground_truth(self, input_dtypes, test_arguments, test_flags):
        sub_runner_target = FunctionTestCaseSubRunner(
            self.fn_name,
            self.backend_handler,
            self.ground_truth_backend,
            self.on_device,
            input_dtypes,
            test_flags,
            self.traced_fn,
        )
        results = sub_runner_target.get_results(test_arguments)
        sub_runner_target.exit()
        return results

    def run(self, input_dtypes, test_arguments, test_flags):
        target_results: TestCaseSubRunnerResult = self._run_target(
            input_dtypes, test_arguments, test_flags
        )
        ground_truth_results: TestCaseSubRunnerResult = self._run_ground_truth(
            input_dtypes, test_arguments, test_flags
        )

        self._check_assertions(target_results, ground_truth_results)


class BackendMethodTestCaseRunner(BackendTestCaseRunner):
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
    ):
        self.class_name = class_name
        self.method_name = method_name
        self.return_flat_np_arrays = return_flat_np_arrays
        self.test_gradients = test_gradients
        self.xs_grad_idxs = xs_grad_idxs
        self.ret_grad_idxs = ret_grad_idxs
        super().__init__(
            backend_handler,
            backend_to_test,
            ground_truth_backend,
            on_device,
            tolerance_dict,
            rtol_,
            atol_,
            test_values,
            traced_fn,
        )

    def _run_target(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        sub_runner_target = MethodTestCaseSubRunner(
            self.class_name,
            self.method_name,
            self.backend_handler,
            self.backend_to_test,
            self.on_device,
            self.traced_fn,
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

    def _run_ground_truth(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        # no need to exit since we're not setting any backend for this
        sub_runner_target = MethodTestCaseSubRunner(
            self.class_name,
            self.method_name,
            self.backend_handler,
            self.ground_truth_backend,
            self.on_device,
            self.traced_fn,
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

    def run(
        self,
        init_input_dtypes,
        method_input_dtypes,
        init_flags,
        method_flags,
        init_all_as_kwargs_np,
        method_all_as_kwargs_np,
    ):
        # getting results from target and ground truth
        target_results: TestCaseSubRunnerResult = self._run_target(
            init_input_dtypes,
            method_input_dtypes,
            init_flags,
            method_flags,
            init_all_as_kwargs_np,
            method_all_as_kwargs_np,
        )
        ground_truth_results: TestCaseSubRunnerResult = self._run_ground_truth(
            init_input_dtypes,
            method_input_dtypes,
            init_flags,
            method_flags,
            init_all_as_kwargs_np,
            method_all_as_kwargs_np,
        )
        self._check_assertions(target_results, ground_truth_results)
