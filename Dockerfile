ARG PLATFORM="amd64"
ARG HOST_IMAGE=ubuntu22.04
ARG CUDA_VERSION=11.8.0


FROM --platform=$PLATFORM nvidia/cuda:$CUDA_VERSION-runtime-$HOST_IMAGE
ENV WD_NAME=/guitar-effects-emulation

ENV PIP_DEFAULT_TIMEOUT=200 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.4.1 \
    TORCH_DEVICE=$DEVICE 

WORKDIR $WD_NAME

RUN apt update && apt install -y python3-pip
COPY poetry.lock pyproject.toml $WD_NAME

RUN pip install poetry==${POETRY_VERSION}
RUN poetry config installer.max-workers 10 \
        && poetry install --no-dev --no-interaction --no-ansi -vvv

COPY nn $WD_NAME/nn
ENTRYPOINT ["poetry", "run", "sh", "entrypoint.sh"]
CMD ["train"]
