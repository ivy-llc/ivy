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
    apt-get install -y wget  \
    git -y && \
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
    apt-get install -y jq && \
    pip3 install --upgrade pip && \
    pip3 install pip-autoremove &&\
    pip3 install setuptools==58.5.3



# Install Ivy Upstream
RUN git clone --progress --recurse-submodules https://github.com/unifyai/ivy --depth 1 && \
    cd ivy && \
    cd ivy_tests/array_api_testing/test_array_api && \
    pip3 install --no-cache-dir -r requirements.txt


# Install local optional
COPY requirements/optional.txt .
COPY requirements/requirements.txt .
#setting torch path early on because torch-scatter needs it
ENV PYTHONPATH "/opt/fw/torch:/opt/miniconda/envs/multienv/bin"

#torch and torch scatter separate installation because they cause issues
RUN pip3 install --no-cache-dir torch --target '/opt/fw/torch' --extra-index-url https://download.pytorch.org/whl/cpu
RUN export ver=$(pip show torch --path '/opt/fw/torch' | grep Version | cut -d ' ' -f2) && \
    pip3 install --target '/opt/fw/torch' --no-cache-dir --upgrade torch-scatter -f https://data.pyg.org/whl/torch-$ver.html


# requirement mappings directs which dependency to be installed and where
COPY /docker/requirement_mappings.json .
SHELL ["/bin/bash", "-c"]

# installing requirements based on mappings in location /opt/fw/$framework
RUN jq -r 'to_entries[] | select(.value != [""]) | .key as $dir | .value[] | @sh "/opt/fw/\($dir) \(.)"' requirement_mappings.json | xargs -I {} sh -c 'printf "Installing %s\n" $2 && pip install --ignore-installed --target $1 $2' sh {}



# install the requirements.txt, optional.txt with the mapped dependencies filtered out
RUN pip install --upgrade -r  requirements.txt &&\
    cp ./optional.txt tmp.txt  &&\
    jq -r 'to_entries[] | [.key] + .value | select(length > 0 or (. == "")) | .[]' requirement_mappings.json | sort -u | xargs -I {} sed -i '/{}/d;/torch/d;/torch-scatter/d;/jax\[.*\]/d' tmp.txt && pip install -r tmp.txt


# add all the directories to environment path so that python knows where to find them
ENV PYTHONPATH "/opt/fw/mxnet:/opt/fw/numpy:/opt/fw/jax:/opt/fw/torch:/opt/fw/tensorflow:/opt/fw/paddle:/opt/miniconda/envs/multienv/bin"

COPY run_tests_CLI/test_dependencies.py .
RUN python3 test_dependencies.py -fp requirements.txt,optional.txt && \
    rm -rf requirements.txt && \
    rm -rf optional.txt && \
    rm -rf tmp.txt && \
    rm -rf test_dependencies.py
