# flake8: noqa
import os
import subprocess
import sys


def directory_generator(req, base="/opt/miniconda/fw/"):
    for versions in req:
        pkg, ver = versions.split("/")
        path = base + pkg + "/" + ver
        if not os.path.exists(path):
            install_pkg(path, pkg + "==" + ver)


def install_pkg(path, pkg, base="fw/"):
    if pkg.split("==")[0] == "torch":
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100 -f https://download.pytorch.org/whl/cpu --target={path} --no-cache-dir",
            shell=True,
        )
    elif pkg.split("==")[0] == "jaxlib":
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100 -f https://storage.googleapis.com/jax-releases/jax_releases.html  --target={path} --no-cache-dir",
            shell=True,
        )
    elif pkg.split("==")[0] == "tensorflow":
        subprocess.run(
            f"pip3 install tensorflow-cpu=={pkg.split('==')[1]} --default-timeout=100  --target={path} --no-cache-dir", shell=True
        )
    else:
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100  --target={path} --no-cache-dir", shell=True
        )


if __name__=="__main__":
    arg_lis=sys.argv
    directory_generator(arg_lis[1:])



