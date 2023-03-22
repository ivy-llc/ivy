# flake8: noqa
import os
import subprocess
<<<<<<< HEAD
=======
import sys
>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead


def directory_generator(req, base="/opt/miniconda/fw/"):
    for versions in req:
        pkg, ver = versions.split("/")
        path = base + pkg + "/" + ver
        if not os.path.exists(path):
            install_pkg(path, pkg + "==" + ver)


def install_pkg(path, pkg, base="fw/"):
    if pkg.split("==")[0] == "torch":
        subprocess.run(
<<<<<<< HEAD
            f"pip3 install {pkg} --default-timeout=100 -f https://download.pytorch.org/whl/torch_stable.html --target={path}",
=======
            f"pip3 install {pkg} --default-timeout=100 -f https://download.pytorch.org/whl/cpu --target={path} --no-cache-dir",
>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead
            shell=True,
        )
    elif pkg.split("==")[0] == "jaxlib":
        subprocess.run(
<<<<<<< HEAD
            f"pip3 install {pkg} --default-timeout=100 -f https://storage.googleapis.com/jax-releases/jax_releases.html  --target={path}",
            shell=True,
        )
    else:
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100  --target={path}", shell=True
        )


torch_req = ["torch/1.4.0", "torch/1.5.0", "torch/1.10.1"]
tensorflow_req = [
    "tensorflow/2.2.0",
    "tensorflow/2.2.1",
    "tensorflow/2.2.2",
    "tensorflow/2.4.4",
    "tensorflow/2.9.0",
    "tensorflow/2.9.1",
]
jax_req = ["jax/0.1.60", "jax/0.1.61"]
jaxlib_req = ["jaxlib/0.1.50", "jaxlib/0.1.60", "jaxlib/0.1.61"]
numpy_req = ["numpy/1.17.3", "numpy/1.17.4", "numpy/1.23.1", "numpy/1.24.0"]

directory_generator(torch_req)
directory_generator(tensorflow_req)
directory_generator(jax_req)
directory_generator(numpy_req)
directory_generator(jaxlib_req)
=======
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



>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead
