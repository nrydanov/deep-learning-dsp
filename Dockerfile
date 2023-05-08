ARG PLATFORM="amd64"
ARG DEVICE="gpu"
ARG WD_NAME=/guitar-effects-emulation

FROM --platform=$PLATFORM python:3.10-bullseye as builder
ARG PLATFORM
ARG DEVICE
ARG GPU_REQUIRED=0
ARG WD_NAME

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.4.1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    TORCH_DEVICE=$DEVICE

RUN if [ "$PLATFORM" = "amd64" ] && [ "$DEVICE" = "gpu" ]; \
        then \
            distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
                && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \ 
                    gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
                && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
                && apt update \
                && apt install -y nvidia-container-toolkit; \
    fi


WORKDIR $WD_NAME

COPY poetry.lock pyproject.toml $WD_NAME
RUN pip install poetry==${POETRY_VERSION} \
        && poetry install --no-dev --no-root --no-interaction --no-ansi

FROM python:3.10-slim-bullseye as runtime
ARG WD_NAME

RUN apt update && apt install -y libgomp1 libsndfile-dev

ENV VIRTUAL_ENV=$WD_NAME/.venv \
    PATH=$WD_NAME/.venv/bin:$PATH

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY model $WD_NAME/model
COPY config $WD_NAME/config

COPY train_example.sh $WD_NAME/train_example.sh
RUN chmod +x $WD_NAME/train_example.sh
# ENTRYPOINT /bin/bash $WD_NAME/entrypoint.sh