import os
import logging
import json


def _get_paths_from_binaries(binaries, root_dir=""):
    """Get all the paths from the binaries.json into a list."""
    paths = []
    if isinstance(binaries, str):
        return [os.path.join(root_dir, binaries)]
    elif isinstance(binaries, dict):
        for k, v in binaries.items():
            paths += _get_paths_from_binaries(v, os.path.join(root_dir, k))
    else:
        for i in binaries:
            paths += _get_paths_from_binaries(i, root_dir)
    return paths


def check_for_binaries():
    folder_path = os.sep.join(__file__.split(os.sep)[:-3])
    binaries_path = os.path.join(folder_path, "binaries.json")
    available_configs_path = os.path.join(folder_path, "available_configs.json")
    initial = True
    if os.path.exists(binaries_path):
        binaries_dict = json.load(open(binaries_path))
        available_configs = json.load(open(available_configs_path))
        binaries_paths = _get_paths_from_binaries(binaries_dict)
        # verify if all binaries are available
        for _, path in enumerate(binaries_paths):
            if not os.path.exists(path):
                if initial:
                    config_str = "\n".join(
                        [
                            f"{module} : {', '.join(configs)}"
                            for module, configs in available_configs.items()
                        ]
                    )
                    logging.warning(
                        "\tSome binaries seem to be missing in your system. This could "
                        "be either because we don't have compatible binaries for your "
                        "system or that newer binaries were available. In the latter "
                        "case, running a ``pip install -e .`` should update the "
                        "binaries. Feel free to create an issue on "
                        "https://github.com/unifyai/ivy.git in case of the former\n"
                    )
                    logging.warning(
                        "\nFollowing are the supported configurations"
                        f" :\n{config_str}\n"
                    )
                    initial = False
                logging.warning(f"\t{path} not found.")
        if not initial:
            print()
