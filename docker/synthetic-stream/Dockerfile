FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS py
FROM rust:1.98.1-slim-bookworm@sha256:ff521445a372125ed4f76e1453a1f8098f2d05332d1601d30db1c1f62757e730
COPY --from=py /usr/local /usr/local
COPY numpy /usr/local/lib/python3.12/site-packages/numpy
COPY numpy.libs /usr/local/lib/python3.12/site-packages/numpy.libs
COPY probe /opt/stream/probe
RUN chmod -R a+rX /opt/stream /usr/local/lib/python3.12/site-packages/numpy /usr/local/lib/python3.12/site-packages/numpy.libs && chmod a+rx /opt/stream/probe
ENV PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
