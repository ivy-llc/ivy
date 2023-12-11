from abc import ABC, abstractmethod


# TODO change methods to static methods
class BackendHandler(ABC):
    _backend = "None"

    @property
    def backend(self):
        return self._backend

    @abstractmethod
    def set_backend(self, backend: str):
        pass

    @abstractmethod
    def unset_backend(self):
        pass
