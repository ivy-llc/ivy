#!/bin/bash
#pip install virtualenv
cd XLA/rust_api/
#mkdir xla_build && virtualenv xla_build
#source xla_build/bin/activate
wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-x86_64-linux-gnu-cuda111.tar.gz
tar -xzvf xla_extension-x86_64-linux-gnu-cuda111.tar.gz
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
pip install maturin
apt-get update
apt install llvm-dev libclang-dev clang
export LIBCLANG_PATH=/usr/local/lib
# maturin develop
