import argparse
import json
import os
from packaging import tags
from urllib import request


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        type=str,
        help="Tag for the binaries to pull",
    )
    args = parser.parse_args()
    tag = args.tag

    folder_path = os.sep.join(__file__.split(os.sep)[:-2])
    binaries_path = os.path.join(folder_path, "binaries.json")
    available_configs_path = os.path.join(folder_path, "available_configs.json")
    print(binaries_path, os.path.exists(binaries_path))
    if os.path.exists(binaries_path):
        binaries_dict = json.load(open(binaries_path))
        available_configs = json.load(open(available_configs_path))
        binaries_paths = _get_paths_from_binaries(binaries_dict, folder_path)
        binaries_exts = {path.split(".")[-1] for path in binaries_paths}

        # remove existing binaries
        print("Removing existing binaries...", end="\r")
        for root, _, files in os.walk(folder_path, topdown=True):
            for file in files:
                if file.split(".")[-1] in binaries_exts:
                    os.remove(os.path.join(root, file))
        print("Removing existing binaries --> done")

        print("Downloading new binaries...")

        version = os.environ["VERSION"] if "VERSION" in os.environ else "main"
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
                    "https://github.com/ivy-llc/binaries/raw/"
                    f"{version}/{search_path}",
                    timeout=40,
                )
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    f.write(response.read())
                print("Downloaded", search_path)
            except request.HTTPError:
                print("Couldn't download", search_path)
                break
