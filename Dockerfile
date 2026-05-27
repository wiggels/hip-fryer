FROM rocm/dev-ubuntu-24.04:7.0.2

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    build-essential \
    clang \
    libclang-dev \
    rocblas-dev \
    rocm-smi-lib \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src

RUN cargo build --release && \
    cp target/release/hip-fryer /usr/local/bin/hip-fryer && \
    rm -rf target /root/.cargo/registry

ENTRYPOINT ["hip-fryer"]
