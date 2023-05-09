# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.8    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# ==================================================================
# tools
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y git wget

# ==================================================================
# python
# ------------------------------------------------------------------
RUN apt-get install -y python3.8 \
        python3.8-dev \
        python3.8-distutils \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.8 ~/get-pip.py && \
    ln -s /usr/bin/python3.8 /usr/local/bin/python && \
    python -m pip --no-cache-dir install --upgrade \
        numpy \
        matplotlib \
        pandas \
        tqdm \
        sphinx==2.4.4 \
        https://github.com/ceshine/shap/archive/master.zip \
        jupyterlab 
# ==================================================================
# pytorch
# ------------------------------------------------------------------
RUN python -m pip --no-cache-dir install --upgrade \
        --pre torch torchvision torchaudio -f \
        https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html \