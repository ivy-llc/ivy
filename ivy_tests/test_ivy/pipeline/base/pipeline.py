from abc import ABC, abstractmethod, abstractproperty


class Pipeline(ABC):
    @abstractproperty
    def backend_handler(self):
        pass

    @abstractmethod
    def test_function(self):
        pass

    @abstractmethod
    def test_method(self):
        pass
