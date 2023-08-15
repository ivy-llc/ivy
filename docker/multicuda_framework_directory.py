# flake8: noqa
import os
import subprocess
import sys

# install requests only for build, and uninstall it later
subprocess.run(
    f"pip3 install requests",
    shell=True,
)

import requests


def get_latest_package_version(package_name):
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        response = requests.get(url)
        response.raise_for_status()
        package_info = response.json()
        return package_info["info"]["version"]
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch package information for {package_name}.")
        return None


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
                f"yes |pip3 install --upgrade {pkg} --target {path} --default-timeout=100"
                " --extra-index-url https://download.pytorch.org/whl/cu118 "
                " --no-cache-dir"
                " -y"
            ),
            shell=True,
        )
    elif pkg.split("==")[0] if "==" in pkg else pkg == "jax":
        subprocess.run(
            (
                f"yes |pip install --upgrade --target {path} 'jax[cuda11_local]' -f"
                " https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  "
                " --no-cache-dir"
                " -y"
            ),
            shell=True,
        )
    elif pkg.split("==")[0] if "==" in pkg else pkg == "paddle":
        subprocess.run(
            (
                "yes |pip install "
                f" paddlepaddle-gpu=={get_latest_package_version('paddlepaddle')}.post117"
                f" --target {path}  -f"
                " https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html  "
                " --no-cache-dir"
            ),
            shell=True,
        )
    else:
        subprocess.run(
            (
                f"yes |pip3 install --upgrade {pkg} --target {path} --default-timeout=100  "
                " --no-cache-dir"
            ),
            shell=True,
        )


if __name__ == "__main__":
    arg_lis = sys.argv
    if len(arg_lis) > 1:  # we have specified what frameworks to install
        directory_generator(arg_lis[1:], "")
    else:
        directory_generator(["tensorflow", "jax", "torch", "paddle"])

    # uninstall requests when done
    # install requests only for build, and uninstall it later
    subprocess.run(
        f"pip3 uninstall requests",
        shell=True,
    )
