from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np  # for type hint only


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


@dataclass
class TestCaseSubRunnerResult:
    flatten_elements_np: np.ndarray
    shape: tuple
    device: str
    dtype: str
    type: str
