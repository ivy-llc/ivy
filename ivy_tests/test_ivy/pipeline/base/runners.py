from abc import ABC, abstractmethod


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
