FROM ubuntu:20.04
WORKDIR /ivy

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-tk && \
    apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx && \
    apt-get install -y python-opengl && \
    apt-get install -y git && \
    apt-get install -y rsync && \
    apt-get install -y libusb-1.0-0 && \
    apt-get install -y libglib2.0-0 && \
    pip3 install --upgrade pip && \
    pip3 install setuptools==58.5.3

RUN pip3 install torch -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --upgrade torch-scatter -f https://pytorch-geometric.com/whl

# Install Ivy Upstream
RUN git clone --progress --recurse-submodules https://github.com/unifyai/ivy && \
    cd ivy && \
    cat requirements/requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat requirements/optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 -m pip install --user -e . && \
    cd ivy_tests/test_array_api && \
    pip3 install --no-cache-dir -r requirements.txt

# Install local requirements
COPY requirements/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install local optional
COPY requirements/optional.txt .
RUN pip3 install --no-cache-dir -r optional.txt

COPY run_tests_CLI/test_dependencies.py .
RUN python3 test_dependencies.py -fp requirements.txt,optional.txt && \
    rm -rf requirements.txt && \
    rm -rf optional.txt && \
    rm -rf test_dependencies.py
