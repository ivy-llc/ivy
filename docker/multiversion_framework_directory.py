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
            f"pip3 install {pkg} --default-timeout=100 --extra-index-url https://download.pytorch.org/whl/cpu --target={path} --no-cache-dir",
            shell=True,
        )
    elif pkg.split("==")[0] == "jaxlib":
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100 -f https://storage.googleapis.com/jax-releases/jax_releases.html  --target={path} --no-cache-dir",
            shell=True,
        )
    elif pkg.split("==")[0] == "tensorflow":
        subprocess.run(
            f"pip3 install tensorflow-cpu=={pkg.split('==')[1]} --default-timeout=100  --target={path} --no-cache-dir",
            shell=True,
        )
    else:
        subprocess.run(
            f"pip3 install {pkg} --default-timeout=100  --target={path} --no-cache-dir",
            shell=True,
        )


if __name__ == "__main__":
    arg_lis = sys.argv

    if "backend" in arg_lis:
        for i in arg_lis[2:]:
            if i.split("/")[0] == "torch":
                subprocess.run(
                    f"pip3 install torch=={i.split('/')[1]} --default-timeout=100 --extra-index-url https://download.pytorch.org/whl/cpu  --no-cache-dir",
                    shell=True,
                )
            elif i.split("/")[0] == "tensorflow":
                print(i, "heere")
                subprocess.run(
                    f"pip3 install tensorflow-cpu=={i.split('/')[1]} --default-timeout=100   --no-cache-dir",
                    shell=True,
                )
            elif i.split("/")[0] == "jaxlib":
                subprocess.run(
                    f"pip3 install {i} --default-timeout=100 -f https://storage.googleapis.com/jax-releases/jax_releases.html  --no-cache-dir",
                    shell=True,
                )
            else:
                subprocess.run(
                    f"pip3 install {i.split('/')[0]}=={i.split('/')[1]} --default-timeout=100   --no-cache-dir",
                    shell=True,
                )
        try:
            import tensorflow  # noqa
        except:
            subprocess.run(
                f"pip3 install tensorflow-cpu --default-timeout=100  --no-cache-dir",
                shell=True,
            )
        try:
            import torch  # noqa
        except:
            subprocess.run(
                f"pip3 install torch --default-timeout=100 --extra-index-url https://download.pytorch.org/whl/cpu  --no-cache-dir",
                shell=True,
            )
        try:
            import jaxlib  # noqa
        except:
            subprocess.run(
                f"pip3 install jaxlib --default-timeout=100 -f https://storage.googleapis.com/jax-releases/jax_releases.html  --no-cache-dir",
                shell=True,
            )
        try:
            import jax  # noqa
        except:
            subprocess.run(
                f"pip3 install jax  --default-timeout=100   --no-cache-dir", shell=True
            )
    else:
        directory_generator(arg_lis[1:])
