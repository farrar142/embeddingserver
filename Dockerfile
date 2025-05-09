FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /usr/src/app

#Stage 1
COPY requirements.txt /usr/src/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt
#Stage 2
COPY . /usr/src/app
ENV PYTHONUNBUFFERED 1
ENTRYPOINT python3 main.py