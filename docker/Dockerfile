FROM debian:buster
WORKDIR /ivy
ARG CLI
ARG pycon=3.10

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies and Miniconda
RUN apt-get update && \
    apt-get install -y wget git && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm ~/miniconda.sh

ENV CONDA_DIR /opt/miniconda/
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda create --name multienv python==$pycon

# Fix protobuf conflicts
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PATH=/opt/miniconda/envs/multienv/bin:$PATH

RUN apt-get update && \
    apt-get install -y python3-pip python3-tk libsm6 libxext6 libxrender-dev libgl1-mesa-glx python-opengl git rsync libusb-1.0-0 libglib2.0-0 jq && \
    pip3 install --upgrade pip && \
    pip3 install pip-autoremove setuptools==58.5.3

# Clone Ivy repository and remove cache
RUN git clone --progress --recurse-submodules https://github.com/ivy-llc/ivy --depth 1 && \
    rm -rf ivy/compiler/_cache

# Install Ivy dependencies
WORKDIR /ivy/ivy_tests/array_api_testing/test_array_api
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy and install local optional requirements
WORKDIR /ivy
COPY requirements/optional.txt .
COPY requirements/requirements.txt .
COPY /docker/requirement_mappings.json .
SHELL ["/bin/bash", "-c"]

RUN jq -r 'to_entries[] | select(.value != [""]) | .key as $dir | .value[] | @sh "/opt/fw/\($dir) \(.)"' requirement_mappings.json | \
    xargs -I {} sh -c 'printf "Installing %s\n" $2 && pip install --ignore-installed --target $1 $2 --extra-index-url  https://download.pytorch.org/whl/cpu --no-cache-dir' sh {} && \
    pip install --upgrade -r requirements.txt && \
    cp ./optional.txt tmp.txt && \
    jq -r 'to_entries[] | [.key] + .value | select(length > 0 or (. == "")) | .[]' requirement_mappings.json | \
    sort -u | xargs -I {} sed -i '/{}/d;/torch/d;/torch-scatter/d;/jax\[.*\]/d' tmp.txt && \
    pip install -r tmp.txt

# Set PYTHONPATH
ENV PYTHONPATH "/opt/fw/mxnet:/opt/fw/numpy:/opt/fw/tensorflow:/opt/fw/jax:/opt/fw/torch:/opt/fw/paddle:/opt/miniconda/envs/multienv/bin"

# Test dependencies
COPY scripts/test_dependencies.py .
RUN python3 test_dependencies.py -fp requirements.txt,optional.txt && \
    rm -rf requirements.txt optional.txt tmp.txt test_dependencies.py
