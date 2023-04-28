FROM debian:buster
WORKDIR /ivy
ARG CLI
# python version for conda
ARG pycon=3.8.10

ENV DEBIAN_FRONTEND=noninteractive

# Install miniconda
ENV CONDA_DIR /opt/miniconda/

RUN apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y wget && \
    apt-get install git -y && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/miniconda


ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda create --name multienv python==$pycon

# to fix protobuf conflicts
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
ENV PATH=/opt/miniconda/envs/multienv/bin:$PATH
RUN apt-get update && \
    apt-get install -y python3-pip python3-tk && \
    apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx && \
    apt-get install -y python-opengl && \
    apt-get install -y git && \
    apt-get install -y rsync && \
    apt-get install -y libusb-1.0-0 && \
    apt-get install -y libglib2.0-0 && \
    pip3 install --upgrade pip && \
    pip3 install setuptools==58.5.3

RUN pip3 install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu
RUN export ver=$(pip show torch | grep Version | cut -d ' ' -f2) && \
    pip3 install --no-cache-dir --upgrade torch-scatter -f https://data.pyg.org/whl/torch-$ver.html

# Install Ivy Upstream
RUN git clone --progress --recurse-submodules https://github.com/unifyai/ivy --depth 1 && \
    cd ivy && \
    sed -i '/nvidia/d' requirements/requirements.txt && \
    cat requirements/requirements.txt | grep -v "ivy-" | pip3 install --upgrade --no-cache-dir -r /dev/stdin && \
    cat requirements/optional.txt | grep -v "ivy-" | pip3 install --upgrade --no-cache-dir -r /dev/stdin && \
    python3 -m pip install --user -e . && \
    cd ivy_tests/array_api_testing/test_array_api && \
    pip3 install --no-cache-dir -r requirements.txt


# Install local optional
COPY requirements/optional.txt .
COPY requirements/requirements.txt .
RUN pip3 install --upgrade --no-cache-dir -r optional.txt && \
    pip3 install --upgrade --no-cache-dir -r requirements.txt

COPY run_tests_CLI/test_dependencies.py .
RUN python3 test_dependencies.py -fp requirements.txt,optional.txt && \
    rm -rf requirements.txt && \
    rm -rf optional.txt && \
    rm -rf optional_gpu.txt && \
    rm -rf test_dependencies.py
