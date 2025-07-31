FROM rocm/dev-ubuntu-24.04:latest

RUN apt-get update && apt-get install -y \
    curl \
    git \
    libssl-dev \
    pkg-config \
    build-essential \
    clang \
    libclang-dev \
    llvm-dev \
    rocblas-dev \
    rocsolver-dev \
    rocfft-dev \
    miopen-hip-dev \
    hip-dev \
    rocm-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .

RUN cargo build --release

ENV PATH="/app/target/release/:${PATH}"
ENTRYPOINT ["hip-fryer"]
