from dataclasses import dataclass


@dataclass
class BackendNativeObject:
    name: str
    namespace: str

    def full_name(self):
        if self.namespace == "":
            return self.name
        return f"{self.namespace}.{self.name}"
