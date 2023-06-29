import time
import tempfile
import shutil
import ivy
import subprocess
import venv
import os
from typing import Callable, Optional, Tuple
import matplotlib.pyplot as plt


SUPPORTED_FRAMEWORKS = ["torch", "tensorflow", "flax", "haiku"]  # flax, haiku?

FRAMEWORKS_TO_INSTALL_PIP = [
    "numpy",
    "torch",
    "tensorflow",
    "jax",
    "jaxlib",
    "dm-haiku",
    "flax",
    "keras",
]
FRAMEWORKS_TO_INSTALL_CONDA = [
    "numpy",
    "pytorch",
    "tensorflow",
    "jax",
    "jaxlib",
    "dm-haiku",
    "flax",
    "keras",
]
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
            f". {activate_script} && pip install {framework_names}",
            shell=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        activate_script = f"conda activate {environment}"
        framework_names = " ".join(FRAMEWORKS_TO_INSTALL_CONDA)
        subprocess.call(
            f"{activate_script} && conda install -y {framework_names}",
            shell=True,
            stdout=subprocess.DEVNULL,
        )


def cleanup_environment(environment, use_conda=False):
    if not use_conda:
        activate_script = os.path.join(environment, "bin", "activate")
        subprocess.call(
            f". {activate_script} && deactivate", shell=True, stdout=subprocess.DEVNULL
        )
        shutil.rmtree(environment)
    else:
        subprocess.call(f"conda deactivate", shell=True)
        subprocess.call(
            f"conda env remove -y -n {environment}",
            shell=True,
            stdout=subprocess.DEVNULL,
        )

def plot_graph(results, k=0):

    runtime_values = []
    memory_usage_values = []
    labels = []


    for framework in results.keys():
        for device in results[framework].keys():
            runtime = results[framework][device]["runtime"]
            memory_usage = results[framework][device]["memory_usage"]
            runtime_values.append(runtime)
            memory_usage_values.append(memory_usage)
            labels.append(framework + " - " + device)

    scores = {}

    for label, runtime, memory_usage in zip(labels, runtime_values, memory_usage_values):
        score = runtime + memory_usage * k
        if score in scores:
            scores[score].append(label)
        else:
            scores[score] = [label]

    plt.scatter(memory_usage_values, runtime_values)
    plt.xlabel("Memory Usage (GB)")
    plt.ylabel("Runtime (s)")
    plt.title("Performance Comparison")
    plt.grid(False)

    for label, x, y in zip(labels, memory_usage_values, runtime_values):
        plt.annotate(label, (x, y), xytext=(0,-12), textcoords='offset points', fontsize=8)
        plt.text(x, y+0.00002, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    for score, data_points in scores.items():
        if len(data_points) > 1:
            plt.plot(memory_usage_values, [score] * len(memory_usage_values), linestyle="--", alpha=0.5, label=f"Score: {score}")
            for data_point in data_points:
                index = labels.index(data_point)
                x = memory_usage_values[index]
    
    best_score = min(scores.keys())
    best_data_point = scores[best_score][0]
    best_index = labels.index(best_data_point)
    plt.scatter(memory_usage_values[best_index], runtime_values[best_index], marker="o", facecolors="none", edgecolors="red", s=100, label="Best Score")

    plt.savefig("performance_comparison.png")

    
def autotune(
    *objs: Callable,
    install_all_frameworks: bool = True,
    use_conda: bool = False,
    source: Optional[str] = None,
    with_numpy: bool = False,
    args: Optional[Tuple] = None,
    params_v = None, 
    k: float = 0,
    save_fig: bool = False,
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
        *objs, source=source, to="torch", args=args, params_v=params_v, kwargs=kwargs
    )  # dummy transpile

    try:
        for device in devices:
            ivy.set_default_device(device)
            x = ivy.to_device(args, device)

            for framework in SUPPORTED_FRAMEWORKS:
                try:
                    transpiled_function = ivy.transpile(
                        *objs, source=source, to=framework, args=x
                    )
                except Exception as e:
                    print(f"Transpilation to {framework} failed: {e}")
                    continue
                (
                    ivy.set_backend(framework)
                    if framework not in ["flax", "haiku"]
                    else ivy.set_backend("jax")
                )
                try:
                    start_time = time.perf_counter()
                    for _ in range(ITERATIONS):
                        transpiled_function(*x)
                    end_time = time.perf_counter()
                except Exception as e:
                    print("Function call failed with {framework}: {e}")
                    continue
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
    if save_fig:
        plot_graph(results, k)
    # Clean up environment
    if install_all_frameworks:
        cleanup_environment(environment, use_conda=use_conda)

    return results, best_deployment
