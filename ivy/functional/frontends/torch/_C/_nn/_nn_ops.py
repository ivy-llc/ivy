# global
import ivy.functional.frontends.torch as torch_frontend


# --- Helpers --- #
# --------------- #


def _parse_to(device=None, dtype=None, copy=False, *, memory_format=None):
    if isinstance(device, str):
        target_device = torch_frontend.device(device)

    # Return the tuple as per the required format
    return (target_device, dtype, copy, memory_format)
