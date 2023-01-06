
import subprocess
import os
import sys
import importlib






subprocess.run('python --version',shell=True)

#install ivy
subprocess.run('conda install -y git',shell=True)
subprocess.run('pip install   git+https://github.com/RickSanchezStoic/ivy.git',shell=True)
subprocess.run('ls',shell=True)

#list of versions required for torch, and so on (maybe passed as args)
torch_req=['torch/1.4.0']
tensorflow_req=['tensorflow/2.2.0','tensorflow/2.2.1']
jax_req=['jax/0.1.60']
numpy_req=['numpy/1.17.3','numpy/1.17.4']

#we create a directory for each framework and install different versions in different directories as per requirements
def direcotry_generator(req,base='fw/'):
    for versions in req:
        pkg,ver=versions.split('/')
        path=base+pkg+'/'+ver
        if not os.path.exists(path):
            install_pkg(path,pkg+'=='+ver)


def install_pkg(path,pkg,base='fw/'):
    subprocess.run(f'pip install {pkg} --target={path}',shell=True)


# to import a specific pkg along with version name, to be used by the test functions
def custom_import(pkg,base='fw/'):     #format is pkg_name/version

    temp=sys.modules.copy()
    sys.path.insert(1, os.path.abspath(base+pkg))

    ret=importlib.import_module(pkg.split('/')[0])
    sys.path.remove(os.path.abspath(base+pkg))
    sys.modules.clear()
    sys.modules.update(temp)

    return ret


#we install numpy requirements
direcotry_generator(numpy_req)



numpy_v1=custom_import('numpy/1.17.4/')
numpy_v2=custom_import('numpy/1.17.3')

print(numpy_v1.__version__)
print(numpy_v2.__version__)


