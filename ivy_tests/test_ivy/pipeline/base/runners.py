# global
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np  # for type hint only

# local
import ivy


@dataclass
class TestCaseSubRunnerResult:
    flatten_elements_np: List[np.ndarray]
    shape: List[Tuple] = None
    device: List[str] = None
    dtype: List[str] = None
    type: List[str] = None


class TestCaseRunner(ABC):
    @abstractmethod
    def _run_target(self):
        pass

    @abstractmethod
    def _run_ground_truth(self):
        pass

    @abstractmethod
    def run(self):
        pass


class TestCaseSubRunner(ABC):
    def exit(self):
        self._backend_handler.unset_backend()

    def trace_if_required(self, fn, test_trace=False, args=None, kwargs=None):
        if test_trace:
            with ivy.utils.backend.ContextManager(self.backend_to_test) as ivy_backend:
                fn = ivy_backend.trace_graph(fn, args=args, kwargs=kwargs)
        return fn

    def _flatten(self, *, ret):
        """Return a flattened arrays in ret."""
        if not isinstance(ret, tuple):
            ret = (ret,)
        ret_idxs = self._ivy.nested_argwhere(ret, self._ivy.is_ivy_array)
        # no ivy array in the returned values, which means it returned scalar
        if len(ret_idxs) == 0:
            ret_idxs = self._ivy.nested_argwhere(ret, self._ivy.isscalar)
            ret_flat = self._ivy.multi_index_nest(ret, ret_idxs)
            ret_flat = [
                self._ivy.asarray(x, dtype=self._ivy.Dtype(str(np.asarray(x).dtype)))
                for x in ret_flat
            ]
        else:
            ret_flat = self._ivy.multi_index_nest(ret, ret_idxs)
        return ret_flat

    def _flatten_and_to_np(self, *, ret):
        ret_flat = self._flatten(ret=ret)
        ret = [self._ivy.to_numpy(x) for x in ret_flat]
        return ret

    def _get_dev_str(self, x):
        return str(self._ivy.dev(x))

    def _is_array(self, x):
        return self._ivy.is_array(x)

    def _get_results_from_ret(self, ret, raw=False, store_types=False):
        ret_flat = self._flatten(ret=ret)
        ret_devices = [
            self._get_dev_str(ret) if self._is_array(ret) else None for ret in ret_flat
        ]
        ret_types = []
        if store_types:
            ret_types = [
                str(type(ret)) if self._is_array(ret) else None for ret in ret_flat
            ]
        ret_np_flat = self._flatten_ret_to_np(ret_flat)
        # for when it returns scalar
        ret_np_flat = [np.array(ret_np) for ret_np in ret_np_flat]
        ret_shapes = [
            ret_np.shape if isinstance(ret_np, np.ndarray) else None
            for ret_np in ret_np_flat
        ]
        ret_dtypes = [
            ret_np.dtype if isinstance(ret_np, np.ndarray) else None
            for ret_np in ret_np_flat
        ]
        if not raw:
            return TestCaseSubRunnerResult(
                flatten_elements_np=ret_np_flat,
                shape=ret_shapes,
                device=ret_devices,
                dtype=ret_dtypes,
                type=ret_types if store_types else None,
            )
        return ret_np_flat, ret_shapes, ret_devices, ret_dtypes, ret_types

    def _flatten_ret_to_np(self, ret_flat):
        return [self._ivy.to_numpy(x) for x in ret_flat]

    @abstractmethod
    def _search_args():
        pass

    @abstractmethod
    def _preprocess_args():
        pass

    @abstractmethod
    def _call_function():
        pass

    @abstractmethod
    def get_results():
        pass
