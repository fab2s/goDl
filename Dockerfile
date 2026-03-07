# goDl development image
# Go + libtorch + CUDA, ready for CGo development and testing.
#
# Usage (see Makefile for shortcuts):
#   make image       — build the image
#   make test        — run all tests (CPU + CUDA)
#   make shell       — interactive shell
#
# Layer ordering is optimized for Docker cache: slowest-changing layers first.
# Source code is mounted at runtime (not COPY'd), so code changes never
# invalidate the image cache.
#
# Override build args for different configurations:
#   docker build --build-arg CUDA_TAG=cu130 --build-arg LIBTORCH_VERSION=2.10.0 .

# --- Layer 1: Base image (changes ~never) ---
# CUDA 12.6 supports Pascal (SM 6.x, GTX 1060+) through Blackwell.
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# --- Layer 2: System dependencies (changes ~never) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    ca-certificates \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 3: Go installation (changes on Go version bump) ---
ARG GO_VERSION=1.24.1
RUN wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz \
    && rm go${GO_VERSION}.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:/root/go/bin:${PATH}"

# --- Layer 4: libtorch installation (changes on libtorch version bump) ---
# This is the largest download (~2GB with CUDA deps), so it must stay cached.
ARG LIBTORCH_VERSION=2.10.0
ARG CUDA_TAG=cu126
RUN wget -q https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2B${CUDA_TAG}.zip \
    && unzip -q libtorch-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_TAG}.zip -d /usr/local \
    && rm libtorch-shared-with-deps-${LIBTORCH_VERSION}+${CUDA_TAG}.zip

# --- Layer 5: Environment (changes ~never, cheap to rebuild) ---
ENV LIBTORCH_PATH="/usr/local/libtorch"
ENV LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${LIBTORCH_PATH}/lib:${LIBRARY_PATH}"
ENV C_INCLUDE_PATH="${LIBTORCH_PATH}/include:${LIBTORCH_PATH}/include/torch/csrc/api/include:${C_INCLUDE_PATH}"
ENV CPLUS_INCLUDE_PATH="${LIBTORCH_PATH}/include:${LIBTORCH_PATH}/include/torch/csrc/api/include:${CPLUS_INCLUDE_PATH}"

# CGo build flags
ENV CGO_CFLAGS="-I${LIBTORCH_PATH}/include -I${LIBTORCH_PATH}/include/torch/csrc/api/include"
ENV CGO_LDFLAGS="-L${LIBTORCH_PATH}/lib"
ENV CGO_ENABLED=1

# Build tag for backend selection — Go source links CUDA libs only with -tags cuda
ENV GOFLAGS="-tags=cuda"

# --- Layer 6: Dev tools (changes on version bump) ---
ARG GOLANGCI_LINT_VERSION=v2.1.6
RUN wget -q -O- https://raw.githubusercontent.com/golangci/golangci-lint/HEAD/install.sh | sh -s -- -b /usr/local/bin ${GOLANGCI_LINT_VERSION}

# pkgsite — local Go documentation server (same engine as pkg.go.dev)
RUN GOBIN=/usr/local/bin go install golang.org/x/pkgsite/cmd/pkgsite@latest

# Suppress NVIDIA license banner on every container run.
# The license is accepted by using the image.
ENV NVIDIA_PRODUCT_NAME=""
ENTRYPOINT []

WORKDIR /workspace
