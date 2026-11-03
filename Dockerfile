# Reproduce the final LOOKOUT-VO entry from the artifacts on Linux x86-64.
#   docker build --platform linux/amd64 -t lookout-vo-repro .
#   docker run --rm --platform linux/amd64 -v /path/to/lookout-vo-test-v0.1:/data/test:ro -v /path/to/3_challenge:/data/challenge:ro \
#     -v "$PWD/out:/out" lookout-vo-repro
# The chain needs numpy only. Its numbers depend on the BLAS kernel and numpy's SIMD paths, so both are fixed below
# (OpenBLAS Nehalem kernels, numpy without AVX-512) -- the x86 setup in which the run was measured; see README.md.
FROM python:3.11.17-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
RUN pip install --no-cache-dir numpy==1.26.4
ENV OPENBLAS_CORETYPE=Nehalem \
    NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL" \
    PYTHONDONTWRITEBYTECODE=1
COPY . /repo
WORKDIR /repo
ENTRYPOINT ["bash", "reproduce.sh", "/data/test", "/data/challenge/pack_submission.py", "/out"]
