# Mapping of standard objects we need to always translate. These objects
# are decoupled from the core recursive translation process and are always translated
# during the call to `ivy.transpile`. if the source framework is set to one of the frontends
# e.g. frontend torch.Tensor dunder methods etc.
STANDARD_METHODS_TO_TRANSLATE = {
    "torch_frontend": {
        "ivy.functional.frontends.torch.tensor": [
            "Tensor.__add__",
            "Tensor.__sub__",
            "Tensor.__mul__",
            "Tensor.__truediv__",
            "Tensor.__eq__",
            "Tensor.__ne__",
        ],
    },
}
STANDARD_FUNCTIONS_TO_TRANSLATE = {}
