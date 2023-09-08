#!/bin/bash
pip install virtualenv
cd XLA/rust_api/
mkdir xla_build && virtualenv xla_build
source xla_build/bin/activate
wget https://github.com/elixir-nx/xla/releases/download/v0.4.4/xla_extension-x86_64-linux-gnu-cuda111.tar.gz
tar -xzvf xla_extension-x86_64-linux-gnu-cuda111.tar.gz
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
source "$HOME/.cargo/env"
apt install llvm-dev libclang-dev clang
export LIBCLANG_PATH=/usr/local/lib
pip install maturin
maturin develop
