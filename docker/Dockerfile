FROM ubuntu:20.04
WORKDIR /ivy

# Install Docker CE CLI
RUN apt-get update \
    && apt-get install -y apt-transport-https ca-certificates curl gnupg2 lsb-release \
    && curl -fsSL https://download.docker.com/linux/$(lsb_release -is | tr '[:upper:]' '[:lower:]')/gpg | apt-key add - 2>/dev/null \
    && echo "deb [arch=amd64] https://download.docker.com/linux/$(lsb_release -is | tr '[:upper:]' '[:lower:]') $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y docker-ce-cli

# Install Docker Compose
RUN LATEST_COMPOSE_VERSION=$(curl -sSL "https://api.github.com/repos/docker/compose/releases/latest" | grep -o -P '(?<="tag_name": ").+(?=")') \
    && curl -sSL "https://github.com/docker/compose/releases/download/${LATEST_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose \
    && chmod +x /usr/local/bin/docker-compose

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

RUN pip3 install --no-cache-dir torch -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir --upgrade torch-scatter -f https://pytorch-geometric.com/whl

# Install Ivy Upstream
RUN git clone --progress --recurse-submodules https://github.com/unifyai/ivy --depth 1 && \
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
