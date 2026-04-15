# =============================================================================
# Anyscale DEV image for vLLM v0.18 with ROCm
# =============================================================================
# Simplified image: Anyscale/Ray base + prebuilt vLLM ROCm wheel + RIXL/UCX.
#
# Build (Anyscale base, default):
#   docker build --platform linux/amd64 -f Dockerfile.v0.18.dev \
#     -t kouroshhahkha/rocm-vllm-ray:v0.18-dev .
#
# Build (OSS Ray base):
#   docker build --platform linux/amd64 \
#     --build-arg BASE_IMAGE=rayproject/ray:nightly-py312-cu128 \
#     -f Dockerfile.v0.18.dev -t kouroshhahkha/rocm-vllm-ray:v0.18-dev .
# =============================================================================

# -----------------------------------------------------------------------------
# Build arguments
# -----------------------------------------------------------------------------
ARG BASE_IMAGE=anyscale/ray:nightly-py312-cu128
ARG VLLM_VERSION=0.18.0
ARG VLLM_ROCM_VARIANT=rocm700

# =============================================================================
# Stage 1: build_rixl - Build UCX and RIXL wheel
# =============================================================================
FROM rocm/dev-ubuntu-22.04:7.0-complete AS build_rixl

ARG RIXL_BRANCH="f33a5599"
ARG RIXL_REPO="https://github.com/ROCm/RIXL.git"
ARG UCX_BRANCH="da3fac2a"
ARG UCX_REPO="https://github.com/ROCm/ucx.git"

ENV ROCM_PATH=/opt/rocm
ENV UCX_HOME=/usr/local/ucx
ENV RIXL_HOME=/usr/local/rixl
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:$PATH

RUN apt-get update && apt-get install -y \
        software-properties-common git curl sudo vim python3 python3-pip python3-venv \
        autoconf libtool pkg-config \
        libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
        libcpprest-dev libaio-dev \
        librdmacm1 librdmacm-dev libibverbs1 libibverbs-dev \
        ibverbs-utils rdmacm-utils ibverbs-providers && \
    rm -rf /var/lib/apt/lists/*

RUN pip install meson auditwheel patchelf tomlkit pybind11 ninja

RUN cd /usr/local/src && \
    git clone ${UCX_REPO} && \
    cd ucx && \
    git checkout ${UCX_BRANCH} && \
    ./autogen.sh && \
    mkdir build && cd build && \
    ../configure \
        --prefix=/usr/local/ucx \
        --enable-shared \
        --disable-static \
        --disable-doxygen-doc \
        --enable-optimizations \
        --enable-devel-headers \
        --with-rocm=/opt/rocm \
        --with-verbs \
        --with-dm \
        --enable-mt && \
    make -j$(nproc) && \
    make install

ENV PATH=/usr/local/ucx/bin:$PATH
ENV LD_LIBRARY_PATH=${UCX_HOME}/lib:${LD_LIBRARY_PATH}

RUN git clone ${RIXL_REPO} /opt/rixl && \
    cd /opt/rixl && \
    git checkout ${RIXL_BRANCH} && \
    meson setup build --prefix=${RIXL_HOME} \
        -Ducx_path=${UCX_HOME} \
        -Drocm_path=${ROCM_PATH} && \
    cd build && \
    ninja && \
    ninja install

# _ucx_install_dir must be exported - build-wheel.sh has a bug where it's undefined
RUN pip install uv && \
    cd /opt/rixl && \
    export _ucx_install_dir=${UCX_HOME} && \
    export LD_LIBRARY_PATH=${RIXL_HOME}/lib:${RIXL_HOME}/lib/x86_64-linux-gnu:${UCX_HOME}/lib:${LD_LIBRARY_PATH} && \
    ./contrib/build-wheel.sh \
        --output-dir /app/install \
        --rocm-dir ${ROCM_PATH} \
        --ucx-plugins-dir ${UCX_HOME}/lib/ucx \
        --nixl-plugins-dir ${RIXL_HOME}/lib/x86_64-linux-gnu/plugins

# =============================================================================
# Stage 2: build_triton - Build ROCm Triton wheel with Python 3.12
# =============================================================================
FROM python:3.12-slim AS build_triton

ARG TRITON_ROCM_COMMIT="f9e5bf54"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential zlib1g-dev libzstd-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install cmake wheel ninja pybind11 && \
    git clone https://github.com/ROCm/triton.git /tmp/triton && \
    cd /tmp/triton && \
    git checkout ${TRITON_ROCM_COMMIT} && \
    if [ ! -f setup.py ] && [ ! -f pyproject.toml ]; then cd python; fi && \
    MAX_JOBS=8 pip wheel --no-deps --wheel-dir /app/triton_wheel . && \
    rm -rf /tmp/triton

# =============================================================================
# Stage 3: final - Runtime image
# =============================================================================
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG VLLM_VERSION
ARG VLLM_ROCM_VARIANT

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies: RDMA libs (for RIXL/UCX) + debugging tools
RUN sudo apt-get update -y && \
    sudo apt-get install -y --no-install-recommends \
        sudo \
        openssh-client \
        openssh-server \
        git \
        gdb \
        curl \
        libopenmpi-dev \
        libpci-dev \
        infiniband-diags \
        libdrm2 \
        libdrm-dev \
        libdrm-amdgpu1 \
        libnuma1 \
        libelf1 \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-utils \
        rdmacm-utils \
        openmpi-bin \
        iputils-ping \
        net-tools \
        netcat-traditional \
        htop \
        lldb \
        lsof \
        psmisc \
        haproxy \
        socat && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

RUN sudo mkdir -p /var/run/sshd

# ROCm runtime libraries from build stage (needed by PyTorch ROCm / vLLM)
COPY --from=build_rixl /opt/rocm /opt/rocm
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}

# Install vLLM from prebuilt ROCm wheels
# https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
RUN pip install uv && \
    uv pip install --system vllm==${VLLM_VERSION} \
        --extra-index-url https://wheels.vllm.ai/rocm/${VLLM_VERSION}/${VLLM_ROCM_VARIANT}

# ROCm-optimized Triton (replaces PyPI triton installed by vLLM)
COPY --from=build_triton /app/triton_wheel/ /tmp/triton_wheel/
RUN pip uninstall -y triton && \
    pip install /tmp/triton_wheel/*.whl && \
    sudo rm -rf /tmp/triton_wheel

# UCX libraries from build stage (required for RIXL)
COPY --from=build_rixl /usr/local/ucx /usr/local/ucx
ENV UCX_HOME=/usr/local/ucx
ENV LD_LIBRARY_PATH=/usr/local/ucx/lib:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/ucx/bin:${PATH}

# RIXL wheel
COPY --from=build_rixl /app/install/ /tmp/rixl_install/
RUN pip install /tmp/rixl_install/*.whl && \
    sudo rm -rf /tmp/rixl_install

# Shell config
RUN echo 'PROMPT_COMMAND="history -a"' >> ~/.bashrc && \
    echo '[ -e ~/.workspacerc ] && source ~/.workspacerc' >> ~/.bashrc

WORKDIR /home/ray
