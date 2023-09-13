from abc import ABC, abstractmethod
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
    def split_args_to_args_and_kwargs(self, num_positional_args, test_arguments):
        """
        Split the kwargs into args and kwargs.

        The first num_positional_args ported to args.
        """
        args = [v for v in list(test_arguments.values())[:num_positional_args]]
        kwargs = {
            k: test_arguments[k]
            for k in list(test_arguments.keys())[num_positional_args:]
        }
        return args, kwargs

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
