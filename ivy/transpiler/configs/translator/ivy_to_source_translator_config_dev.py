# Mapping of standard objects we need to always translate. These objects
# are decoupled from the core recursive translation process and are always translated
# during the call to `ivy.transpile`. if the source framework is set to `ivy`
# e.g. handle_array_like_without_promotion etc.
STANDARD_METHODS_TO_TRANSLATE = {
    "ivy": {
        "ivy_transpiled_outputs.ivy_outputs.ivy.functional.frontends.torch.tensor": [
            "ivy___add___frnt_",
            "ivy___sub___frnt_",
            "ivy___mul___frnt_",
            "ivy___truediv___frnt_",
            "ivy___eq___frnt_",
            "ivy___ne___frnt_",
        ],
    },
}
STANDARD_FUNCTIONS_TO_TRANSLATE = {
    "ivy": {
        "ivy.utils.decorator_utils": [
            "handle_transpose_in_input_and_output",
        ],
        "ivy.func_wrapper": [
            "handle_array_like_without_promotion",
        ],
    },
}
