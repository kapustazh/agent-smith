FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /testbed
RUN printf 'def add(a, b):\n    return a - b\n' > calc.py \
    && git init -q && git config user.name fixture \
    && git config user.email fixture@example.test \
    && git add calc.py && git commit -qm baseline
