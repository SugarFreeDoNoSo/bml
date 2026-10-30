# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
FROM rust:1.98.1-slim-bookworm@sha256:ff521445a372125ed4f76e1453a1f8098f2d05332d1601d30db1c1f62757e730 AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY docker/gguf-validation/reference ./docker/gguf-validation/reference
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export CARGO_HTTP_CAINFO=/run/secrets/proxy_ca; fi; \
    cargo build -p bml-gguf --example bounded_probe --release --locked
RUN sha256sum Cargo.lock crates/gguf/examples/bounded_probe.rs > /src/source.sha256
RUN cc -std=c11 -O2 docker/gguf-validation/reference/shim.c -lm -o /src/q4_reference
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
ARG SOURCE_COMMIT=unknown
ARG SOURCE_DIRTY=unknown
ENV SOURCE_COMMIT=$SOURCE_COMMIT SOURCE_DIRTY=$SOURCE_DIRTY
LABEL org.opencontainers.image.title="BML bounded GGUF validation" org.opencontainers.image.revision=$SOURCE_COMMIT
WORKDIR /opt/bml
COPY --from=build /src/target/release/examples/bounded_probe /src/q4_reference ./
COPY --from=build /src/source.sha256 ./source.sha256
COPY docker/gguf-validation/run.py ./run.py
COPY docker/gguf-validation/reference ./reference
RUN python -c 'import json,os,platform; json.dump({"source_commit":os.environ["SOURCE_COMMIT"],"source_dirty":os.environ["SOURCE_DIRTY"],"source_sha256":open("source.sha256").read(),"python":platform.python_version(),"rust":"1.98.1","scope":"CPU only"},open("build.json","w"))'
RUN chmod -R a+rX /opt/bml
ENTRYPOINT ["python", "/opt/bml/run.py"]
