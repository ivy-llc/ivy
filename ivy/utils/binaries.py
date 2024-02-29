import os
import logging
import json
from packaging import tags
from urllib import request
from tqdm import tqdm


def _get_paths_from_binaries(binaries, root_dir=""):
    """Get all the paths from the binaries.json into a list."""
    paths = []
    ext = "pyd" if os.name == "nt" else "so"
    if isinstance(binaries, str):
        return [os.path.join(root_dir, binaries + "." + ext)]
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
        binaries_paths = _get_paths_from_binaries(binaries_dict, folder_path)
        # verify if all binaries are available
        for path in binaries_paths:
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
                        "case, calling ivy.utils.cleanup_and_fetch_binaries() should "
                        "fetch the binaries binaries. Feel free to create an issue on "
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


def cleanup_and_fetch_binaries(clean=True):
    folder_path = os.sep.join(__file__.split(os.sep)[:-3])
    binaries_path = os.path.join(folder_path, "binaries.json")
    available_configs_path = os.path.join(folder_path, "available_configs.json")
    if os.path.exists(binaries_path):
        binaries_dict = json.load(open(binaries_path))
        available_configs = json.load(open(available_configs_path))
        binaries_paths = _get_paths_from_binaries(binaries_dict, folder_path)
        binaries_exts = {path.split(".")[-1] for path in binaries_paths}

        # clean up existing binaries
        if clean:
            print("Cleaning up existing binaries...", end="\r")
            for root, _, files in os.walk(folder_path, topdown=True):
                for file in files:
                    if file.split(".")[-1] in binaries_exts:
                        os.remove(os.path.join(root, file))
            print("Cleaning up existing binaries --> done")

        print("Downloading new binaries...")
        all_tags = list(tags.sys_tags())

        version = os.environ["VERSION"] if "VERSION" in os.environ else "main"
        terminate = False

        # download binaries for the tag with highest precedence
        with tqdm(total=len(binaries_paths)) as pbar:
            for tag in all_tags:
                if terminate:
                    break
                for path in binaries_paths:
                    module = path[len(folder_path) :][1:].split(os.sep)[1]
                    if (
                        os.path.exists(path)
                        or str(tag) not in available_configs[module]
                    ):
                        continue
                    folders = path.split(os.sep)
                    _, file_path = os.sep.join(folders[:-1]), folders[-1]
                    ext = "pyd" if os.name == "nt" else "so"
                    file_name = f"{file_path[:-(len(ext)+1)]}_{tag}.{ext}"
                    search_path = f"{module}/{file_name}"
                    try:
                        response = request.urlopen(
                            "https://github.com/unifyai/binaries/raw/"
                            f"{version}/{search_path}",
                            timeout=40,
                        )
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "wb") as f:
                            f.write(response.read())
                        terminate = path == binaries_paths[-1]
                        pbar.update(1)
                    except request.HTTPError:
                        break
        if terminate:
            print("Downloaded all binaries!")
        else:
            print(
                "Couldn't download all binaries. Try importing ivy to get more "
                "details about the missing binaries."
            )
