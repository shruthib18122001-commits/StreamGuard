# UNVERIFIED: no Docker available on this dev machine to actually build
# this image. Written against nvidia/cuda's documented tag scheme; the
# exact tag may need adjusting to match the target GPU's driver (Colab T4
# today). Jetson Orin Nano needs an aarch64/L4T base image instead of this
# one -- per CLAUDE.md, that untested path isn't written yet.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY streamguard ./streamguard
COPY scripts ./scripts
COPY configs ./configs

# TensorRT/CuPy/pynvml aren't in pyproject.toml's base dependencies -- the
# stage-4 modules import them lazily specifically so the package installs
# and imports fine without them (see tensorrt_backend.py, cuda_timer.py).
# Install them explicitly here, version-matched to the CUDA base above.
RUN pip3 install --no-cache-dir -e ".[viz]" \
    && pip3 install --no-cache-dir cupy-cuda12x tensorrt pynvml

EXPOSE 9108
ENTRYPOINT ["python3", "scripts/serve.py"]
CMD ["--config", "configs/default.yaml", "--metrics-port", "9108"]
