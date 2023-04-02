FROM python:3.9.10-buster

RUN pip install -U pip
RUN pip install optuna "scikit-learn>=0.19.0" "botorch>=0.4.0,<0.8.0" jupyter torch torchaudio torchvision "plotly>=4.9.0"
RUN pip install PyMySQL psycopg2-binary cryptography redis pymssql

WORKDIR /work
