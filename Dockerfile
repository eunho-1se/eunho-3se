FROM python:3.12

WORKDIR /workspace

RUN pip install jupyter numpy tiktoken

EXPOSE 4444

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0", "--port=4444", "--allow-root"]