# flake8: noqa
import os
import subprocess
import sys
import json


def directory_generator(req, base="/opt/fw/"):
    for versions in req:
        if "/" in versions:
            pkg, ver = versions.split("/")
            path = base + pkg + "/" + ver
            if not os.path.exists(path):
                install_pkg(path, pkg + "==" + ver)
        else:
            install_pkg(base + versions, versions)


def install_pkg(path, pkg, base="fw/"):
    if pkg.split("==")[0] if "==" in pkg else pkg == "torch":
        subprocess.run(
            (
                f"pip3 install --upgrade {pkg} --target {path} --default-timeout=100"
                " --extra-index-url https://download.pytorch.org/whl/cpu "
                " --no-cache-dir"
            ),
            shell=True,
        )
    elif pkg.split("==")[0] == "jax":
        subprocess.run(
            (
                f"pip install --upgrade {pkg} --target  {path}  -f"
                " https://storage.googleapis.com/jax-releases/jax_releases.html  "
                " --no-cache-dir"
            ),
            shell=True,
        )
    else:
        subprocess.run(
            (
                f"pip3 install --upgrade {pkg} --target {path} --default-timeout=100  "
                " --no-cache-dir"
            ),
            shell=True,
        )


def install_deps(pkgs, path_to_json, base="/opt/fw/"):
    for fw in pkgs:
        fw, ver = fw.split("/")
        path = base + fw + "/" + ver
        # check to see if this pkg has specific version dependencies
        with open(path_to_json, "r") as file:
            json_data = json.load(file)
            print(json_data.keys())
            for keys in json_data[fw]:
                print(keys, "here")
                # check if key is dict
                if isinstance(keys, dict):
                    # this is a dep with just one key
                    # being the dep
                    dep = list(keys.keys())[0]
                    # check if version is there in this
                    if ver in keys[dep].keys():
                        subprocess.run(
                            (
                                "pip3 install --upgrade"
                                f" {dep}=={keys[dep][ver]} --target"
                                f" {path} --default-timeout=100   --no-cache-dir"
                            ),
                            shell=True,
                        )
                    else:
                        subprocess.run(
                            (
                                f"pip3 install  {dep} --target"
                                f" {path} --default-timeout=100   --no-cache-dir"
                            ),
                            shell=True,
                        )
                else:
                    subprocess.run(
                        (
                            f"pip3 install  {keys} --target"
                            f" {path} --default-timeout=100   --no-cache-dir"
                        ),
                        shell=True,
                    )


if __name__ == "__main__":
    arg_lis = sys.argv

    json_path = (  # path to the json file storing version specific deps
        "requirement_mappings_multiversion.json"
    )

    directory_generator(arg_lis[1:])
    install_deps(arg_lis[1:], json_path)
