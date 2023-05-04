ARG PLATFORM="amd64"
ARG DEVICE="gpu"

FROM --platform=$PLATFORM python:3.10-bullseye as builder
ARG PLATFORM
ARG DEVICE
ARG GPU_REQUIRED=0
ENV WD_NAME=/guitar-emulation

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.4.1 \
    TORCH_DEVICE=$DEVICE

RUN apt update && apt install -y libsndfile1-dev
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
        && poetry config installer.max-workers 10 \
        && poetry install --no-dev --no-interaction --no-ansi -vvv


COPY model $WD_NAME/model
COPY config $WD_NAME/config
COPY entrypoint.sh $WD_NAME/entrypoint.sh
RUN chmod +x $WD_NAME/entrypoint.sh
# ENTRYPOINT /bin/bash $WD_NAME/entrypoint.sh