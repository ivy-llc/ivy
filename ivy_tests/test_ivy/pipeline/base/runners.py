from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
import numpy as np  # for type hint only


@dataclass
class TestCaseSubRunnerResult:
    flatten_elements_np: np.ndarray
    shape: tuple
    device: str
    dtype: str
    type: str


class TestCaseRunner(ABC):
    @abstractmethod
    def _assert_dtype(self):
        pass

    @abstractmethod
    def _assert_device(self):
        pass

    @abstractmethod
    def _assert_type(self):
        pass

    @abstractmethod
    def _assert_equal_elements(self):
        pass

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

    def compile_if_required(self, fn, test_compile=False, args=None, kwargs=None):
        if test_compile:
            fn = self._ivy.compile(fn, args=args, kwargs=kwargs)
        return fn

    @abstractproperty
    def backend_handler(self):
        pass

    @abstractproperty
    def _ivy(self):
        pass

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
