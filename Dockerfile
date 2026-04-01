# ---------- Base (CUDA + build tools) ----------
# FROM nvidia/cuda:12.1.1-cudnn9-devel-ubuntu22.04
# FROM docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04
FROM docker.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive \
    # PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # PIP_NO_CACHE_DIR=1 \
    # Hugging Face caches (persist by mounting a volume to these)
    HF_HOME=/opt/hf \
    HUGGINGFACE_HUB_CACHE=/opt/hf/hub \
    TRANSFORMERS_CACHE=/opt/hf/transformers \
    # Optional: set your token at runtime with `-e HF_TOKEN=...`
    HF_TOKEN=""

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev \
    git git-lfs \
    ffmpeg \
    libgl1 libglib2.0-0 \
    build-essential pkg-config \
    curl ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (nice for mapping host volumes)
# ARG USERNAME=appuser
# ARG UID=1000
# ARG GID=1000
# RUN groupadd -g $GID $USERNAME && \
#     useradd -m -s /bin/bash -u $UID -g $GID $USERNAME

# # Tini for sane signal handling
# RUN wget -qO /usr/bin/tini https://github.com/krallin/tini/releases/download/v0.19.0/tini && \
#     chmod +x /usr/bin/tini
# ENTRYPOINT ["/usr/bin/tini","--"]

# ---------- Python & PyTorch ----------
# Use pip from python3
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch CUDA 12.1 wheels + common ML deps
# (Pin or bump as needed; these match CUDA 12.1 base)
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Core packages often required by SAM/BLIP projects
RUN pip install \
    numpy scipy matplotlib \
    timm \
    opencv-python-headless \
    pillow \
    pycocotools \
    onnx onnxruntime-gpu \
    sentencepiece \
    accelerate \
    transformers>=4.44 \
    bitsandbytes \
    gradio uvicorn fastapi \
    einops \
    # xformers improves attention perf; version should match torch
    xformers -f https://download.pytorch.org/whl/cu121

# ---------- Workdir & caches ----------
WORKDIR /workspace
RUN mkdir -p /workspace /opt/hf
# && \
    # chown -R $USERNAME:$USERNAME /workspace /opt/hf
# USER $USERNAME

# ---------- Describe Anything (DAM) ----------
# Clone Describe Anything and install with SAM/SAM2 + eval extras
RUN git clone https://github.com/NVlabs/describe-anything.git /workspace/describe-anything && \
    pip install -e "/workspace/describe-anything[sam,sam2,eval]"

# ---------- Clone & install SAM 2 ----------
# Repo name assumed; adjust if you use a fork or different branch
RUN git clone https://github.com/facebookresearch/segment-anything-2.git sam2 && \
    pip install -e ./sam2


# Run /workspace/sam2/checkpoints/download_ckpts.sh from that directory to download SAM2 checkpoints
# Also do so silently to avoid hanging on prompts

RUN bash -c "cd /workspace/sam2/checkpoints && ./download_ckpts.sh"

# ---------- Clone & install SAM 3 ----------
# Note: SAM3 officially recommends PyTorch >=2.7.0 with CUDA 12.6 wheels,
# but the library itself doesn't hard-pin torch in pyproject.toml.
# This installs SAM3 against the torch you already have.
RUN git clone https://github.com/facebookresearch/sam3.git /workspace/sam3 && \
    pip install -e /workspace/sam3 && \
    # Optional: notebook / dev / train extras if you want them
    pip install -e "/workspace/sam3[notebooks]" && \
    pip install -e "/workspace/sam3[train,dev]"


# ---------- OpenMMLab stack (required for Semantic-SAM) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ninja-build \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

#RUN pip install -U openmim && \
#    pip install -U pip setuptools wheel && \
 #   mim install "mmcv==2.1.0" && \
 #   mim install "mmdet==3.3.0"

# ---------- Detectron2 ----------
RUN pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# # ---------- Clone & install Semantic-SAM ----------
# RUN git clone https://github.com/UX-Decoder/Semantic-SAM.git /workspace/semantic-sam && \
#     pip install -e /workspace/semantic-sam

# ---------- Clone & install Semantic-SAM (code only, no deps) ----------
RUN git clone https://github.com/UX-Decoder/Semantic-SAM.git /workspace/semantic-sam && \
    pip install -e /workspace/semantic-sam --no-deps

RUN pip install wandb flash_attn qwen-vl-utils==0.0.14

# ---------- OpenAI CLIP ----------
RUN pip install ftfy regex tqdm && \
    pip install git+https://github.com/openai/CLIP.git
# Maybe pip install -U --force-reinstall --no-cache-dir "pydantic-core==2.27.2"

# # Download the Semantic-SAM checkpoints
# RUN mkdir -p /workspace/semantic-sam/checkpoints && \
#     wget -O /workspace/semantic-sam/checkpoints/swinl_only_sam_many2many.pth \
#     https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth

# wget https://github.com/UX-Decoder/Semantic-SAM/releases/download/checkpoint/swinl_only_sam_many2many.pth

# # ---------- (Optional) Pre-download Semantic-SAM checkpoints ----------
# RUN mkdir -p /workspace/checkpoints && \
#     wget -O /workspace/checkpoints/swinl_only_sam_many2many.pth \
#     https://huggingface.co/UX-Decoder/Semantic-SAM/resolve/main/swinl_only_sam_many2many.pth


# # ---------- Install InstructBLIP (via LAVIS) ----------
# # InstructBLIP is available through the LAVIS toolkit
# RUN pip install "git+https://github.com/salesforce/LAVIS.git#egg=lavis"

# Optional: small sanity scripts directory
RUN mkdir -p /workspace/examples

RUN ln -s /usr/bin/python3 /usr/bin/python

# (Optional) example launcher placeholders you can overwrite in your project
# COPY ./examples /workspace/examples

# Expose common demo ports (Gradio, FastAPI, etc.)
EXPOSE 7860 8000

# Default shell
CMD ["/bin/bash"]


# HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python3 test_directory_of_imgs.py --config-name init_dataset_gen_qwen2_cache ++image_dir_path=/workspace/data/