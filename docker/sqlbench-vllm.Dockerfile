ARG VLLM_IMAGE=vllm/vllm-openai:v0.22.1
FROM ${VLLM_IMAGE}

RUN python3 -m pip install --no-cache-dir google-cloud-storage==3.7.0

COPY docker/sqlbench-vllm-entrypoint.py /usr/local/bin/sqlbench-vllm-entrypoint.py

EXPOSE 8000

ENTRYPOINT ["python3", "/usr/local/bin/sqlbench-vllm-entrypoint.py"]
