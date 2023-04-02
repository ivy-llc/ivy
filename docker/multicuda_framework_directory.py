# flake8: noqa
import os
import subprocess
import sys


def directory_generator(req, base="/fw/"):
    for versions in req:
        pkg, ver = versions.split("/")
        path = base + pkg + "/" + ver
        if not os.path.exists(path):
            install_pkg(path, pkg + "==" + ver)


def install_pkg(path, pkg, base="fw/"):
    if pkg.split("==")[0] == "torch":
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100 --extra-index-url https://download.pytorch.org/whl/cu116  --no-cache-dir",
            shell=True,
        )
    elif pkg.split("==")[0] == "jaxlib":
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html   --no-cache-dir",
            shell=True,
        )
    else:
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100   --no-cache-dir", shell=True
        )


if __name__ == "__main__":
    arg_lis = sys.argv
    directory_generator(arg_lis[1:], "")


# torch_req = ["torch/1.13.1"]
# tensorflow_req = [
#     "tensorflow/2.11.0",
# ]
# jax_req = ["jax/0.4.6"]
# jaxlib_req = ["jaxlib/0.4.6"]
# numpy_req = ["numpy/1.24.2"]
#
# directory_generator(torch_req)
# directory_generator(tensorflow_req)
# directory_generator(jax_req)
# directory_generator(numpy_req)
# directory_generator(jaxlib_req)
