import time
import tempfile
import shutil
import ivy
import subprocess
import venv
import os
from typing import Callable, Optional, Tuple


SUPPORTED_FRAMEWORKS = ["torch", "tensorflow", "flax", "haiku"]  # flax, haiku?

FRAMEWORKS_TO_INSTALL_PIP = ["numpy","torch", "tensorflow", "jax", "jaxlib", "dm-haiku", "flax", "keras"]
FRAMEWORKS_TO_INSTALL_CONDA = ["numpy", "pytorch", "tensorflow", "jax", "jaxlib", "dm-haiku", "flax", "keras"]
ITERATIONS = (
    100  # number of iterations used when benchmarking runtime post transpilation
)


def create_virtual_environment():
    # Create a temporary virtual environment
    temp_env_dir = tempfile.mkdtemp(prefix="ivy_venv_")
    venv.create(temp_env_dir, with_pip=True)
    return temp_env_dir


def create_conda_environment(env_name):
    # Create a temporary conda environment
    subprocess.call(f"conda create -y -n {env_name}", shell=True)
    return env_name


def install_frameworks(environment, use_conda=False):
    if not use_conda:
        activate_script = os.path.join(environment, "bin", "activate")
        framework_names = " ".join(FRAMEWORKS_TO_INSTALL_PIP)
        subprocess.call(
            f". {activate_script} && pip install {framework_names}", shell=True, stdout=subprocess.DEVNULL
        )
    else:
        activate_script = f"conda activate {environment}"
        framework_names = " ".join(FRAMEWORKS_TO_INSTALL_CONDA)
        subprocess.call(
            f"{activate_script} && conda install -y {framework_names}", shell=True, stdout=subprocess.DEVNULL
        )


def cleanup_environment(environment, use_conda=False):
    if not use_conda:
        activate_script = os.path.join(environment, "bin", "activate")
        subprocess.call(f". {activate_script} && deactivate", shell=True, stdout=subprocess.DEVNULL)
        shutil.rmtree(environment)
    else:
        subprocess.call(f"conda deactivate", shell=True)
        subprocess.call(f"conda env remove -y -n {environment}", shell=True, stdout=subprocess.DEVNULL)


def autotune(
    *objs: Callable,
    install_all_frameworks=True,
    use_conda=False,
    source: Optional[str] = None,
    with_numpy: bool = False,
    args: Optional[Tuple] = None,
    kwargs: Optional[dict] = None,
):
    global SUPPORTED_FRAMEWORKS
    if with_numpy:
        SUPPORTED_FRAMEWORKS += ["numpy"]

    if install_all_frameworks:
        if use_conda:
            environment = create_conda_environment("ivy_env")
        else:
            environment = create_virtual_environment()

        install_frameworks(environment, use_conda=use_conda)

    import torch

    gpu_available = torch.cuda.device_count()

    devices = ["cpu"] + ["gpu:{}".format(i) for i in range(gpu_available)]

    results = {framework: {} for framework in SUPPORTED_FRAMEWORKS}
    min_runtime = None
    best_deployment = {
        "framework": None,
        "runtime": None,
        "memory_usage": None,
        "output": None,
    }

    ivy.transpile(
        *objs, source=source, to="torch", args=args, kwargs=kwargs
    )  # dummy transpile

    try:
        for device in devices:
            ivy.set_default_device(device)
            x = ivy.to_device(args, device)

            for framework in SUPPORTED_FRAMEWORKS:
                transpiled_function = ivy.transpile(
                    *objs, source=source, to=framework, args=x
                )
                (
                    ivy.set_backend(framework)
                    if framework not in ["flax", "haiku"]
                    else ivy.set_backend("jax")
                )

                start_time = time.perf_counter()
                for _ in range(ITERATIONS):
                    transpiled_function(*x)
                end_time = time.perf_counter()
                run_time = end_time - start_time
                run_time /= ITERATIONS

                memory_used = ivy.used_mem_on_dev(device, process_specific=True)

                if framework not in results:
                    results[framework] = {}
                if device not in results[framework]:
                    results[framework][device] = {}
                results[framework][device] = {
                    "runtime": run_time,
                    "memory_usage": memory_used,
                }
                if min_runtime is None or run_time < min_runtime:
                    min_runtime = run_time
                    best_deployment["framework"] = framework
                    best_deployment["runtime"] = run_time
                    best_deployment["memory_usage"] = memory_used
                    best_deployment["output"] = transpiled_function

                ivy.clear_cached_mem_on_dev(device)
    except Exception as e:
        print("exception: ", e)
        # Clean up environment
        if install_all_frameworks:
            cleanup_environment(environment, use_conda=use_conda)
        return
    # Clean up environment
    if install_all_frameworks:
        cleanup_environment(environment, use_conda=use_conda)

    return results, best_deployment
