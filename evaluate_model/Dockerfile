FROM drtools/dask:latest
ENV PYTHONWARNINGS="ignore"

COPY . /opt/evaluate_model/

RUN pip install -r /opt/evaluate_model/requirements.txt

WORKDIR /opt/evaluate_model/
