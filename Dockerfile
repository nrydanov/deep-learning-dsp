ARG BASE_CONTAINER=python:3.8-alpine
FROM $BASE_CONTAINER

USER root

ENV POETRY_VERSION=1.4.0
WORKDIR /app
RUN apk update && apk add --no-cache --virtual build-deps musl-dev libffi-dev gcc libc-dev g++ python3-dev
COPY poetry.lock pyproject.toml /app/
RUN pip install poetry==${POETRY_VERSION}
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction

