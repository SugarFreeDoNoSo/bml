# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# Build only with --platform=linux/amd64; no model belongs in this image.
FROM rust:1.98.1-slim-bookworm@sha256:ff521445a372125ed4f76e1453a1f8098f2d05332d1601d30db1c1f62757e730 AS compiler
ARG SOURCE_COMMIT=unknown
ARG SOURCE_DIRTY=unknown
LABEL org.opencontainers.image.title="BML native ELF fragment compiler" \
      org.opencontainers.image.revision=$SOURCE_COMMIT
ENV CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 \
    RUSTFLAGS="-Ctarget-cpu=x86-64 -Ctarget-feature=-fma"
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates ./crates
COPY tests ./tests
# Populate the exact toolchain and dependency cache at image-build time.
# Generated artifact builds subsequently use --offline and --locked.
RUN rustup component add --toolchain 1.98.1 rustfmt clippy llvm-tools-preview && \
    cargo +1.98.1 build --release --locked --target x86_64-unknown-linux-gnu \
      -p bml-inference --bin bml-model
RUN test "$(rustc +1.98.1 --version | cut -d ' ' -f 2)" = 1.98.1 && \
    sha256sum Cargo.lock target/x86_64-unknown-linux-gnu/release/bml-model > /src/compiler.sha256
# Source/cache COPY modes may be private on the host. Generated builds run as
# the caller's UID against a read-only image and copy Cargo's cache to /work.
RUN chmod -R a+rX /src /usr/local/cargo /usr/local/rustup
ENTRYPOINT ["/src/target/x86_64-unknown-linux-gnu/release/bml-model"]

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS runtime
ARG SOURCE_COMMIT=unknown
ARG SOURCE_DIRTY=unknown
ENV PYTHONDONTWRITEBYTECODE=1 SOURCE_COMMIT=$SOURCE_COMMIT SOURCE_DIRTY=$SOURCE_DIRTY
LABEL org.opencontainers.image.title="BML native ELF fragment validation" \
      org.opencontainers.image.revision=$SOURCE_COMMIT
WORKDIR /opt/bml
COPY --from=compiler /src/target/x86_64-unknown-linux-gnu/release/bml-model ./bml-model
COPY --from=compiler /src/compiler.sha256 ./compiler.sha256
COPY tools/validate_fragment_elf.py ./validate_fragment_elf.py
COPY tools/validate_fragment_tcp.py ./validate_fragment_tcp.py
COPY tools/fragment_private_transport.py ./fragment_private_transport.py
COPY tools/validate_fragment_private.py ./validate_fragment_private.py
RUN chmod -R a+rX /opt/bml
ENTRYPOINT ["python", "/opt/bml/validate_fragment_elf.py"]
