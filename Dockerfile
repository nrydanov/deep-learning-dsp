FROM python:3.10

ARG DEVICE=cpu

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.4.1

RUN apt update && apt install -y libsndfile1-dev

WORKDIR /guitar-emulation

COPY poetry.lock pyproject.toml /guitar-emulation

RUN pip install poetry==${POETRY_VERSION}
RUN poetry install --no-interaction --no-ansi

COPY model /guitar-emulation/model