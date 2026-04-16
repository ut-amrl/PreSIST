# Running on TACC (Stampede3)

## One-time setup

### 1. Get an allocation
Request access via the [ACCESS portal](https://allocations.access-ci.org/) or the TACC Portal. The current allocation is `IRI26004` on the `amd-rtx` partition.

### 2. Set up directory structure
TACC has three storage tiers. Use them as follows:

| Path | Use for |
|------|---------|
| `$WORK/workspaces/semantic-persistence-filter` | Repo (code, configs) |
| `$SCRATCH/data/semantic-persistence-filter` | Datasets |
| `$SCRATCH/containers/` | Apptainer SIF files |
| `$SCRATCH/.cache/huggingface` | HuggingFace model cache |

```bash
mkdir -p $SCRATCH/containers $SCRATCH/data/semantic-persistence-filter $SCRATCH/.cache/huggingface
```

Clone or transfer the repo to `$WORK`:
```bash
# On TACC login node
cd $WORK/workspaces
git clone <repo-url> semantic-persistence-filter
```

Transfer your datasets to `$SCRATCH/data/semantic-persistence-filter`. The image root inside the container is `/workspace/data`, so paths in manifests like `train_dataset_manifests/train_mix_0.txt` should be relative to that root.

### 3. Build or transfer the SPF container

On a machine with Docker, build and export the SIF:
```bash
docker build -t spf:latest .
apptainer build spf_latest.sif docker-daemon://spf:latest
scp spf_latest.sif <user>@stampede3.tacc.utexas.edu:$SCRATCH/containers/
```

Or build directly on TACC if you have internet access on a compute node:
```bash
apptainer build $SCRATCH/containers/spf_latest.sif docker://mandi1267/spf:latest
```

### 4. Set up micromamba + vLLM env (vLLM variant only)

The vLLM variant runs vLLM natively (not in a container) using a micromamba environment.

```bash
# Install micromamba if not already present
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
export MAMBA_ROOT_PREFIX=$WORK/micromamba

# Create the vllm env
micromamba create -n vllm2 python=3.11 -y
micromamba activate vllm2
pip install vllm
```

### 5. Pull additional SIFs (TRT-LLM + DAM variant only)

```bash
# TRT-LLM server (~20 GB, takes a while)
apptainer pull $SCRATCH/containers/tritonserver_26.03-trtllm-python-py3.sif \
    docker://nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3

# DAM server
apptainer pull $SCRATCH/containers/describe-anything_latest.sif \
    docker://mandi1267/describe-anything:latest
```

---

## vLLM variant

Runs all pipeline phases (segmentation → object descriptions → persistence). The LLM persistence querier is served via vLLM in a micromamba env; the pipeline runs in the SPF container.

**Scripts:**
- `tacc_run_spf_with_vllm_batch.slurm` — sbatch wrapper
- `tacc_run_spf_with_vllm.sh` — driver (starts vLLM, waits, runs pipeline)
- `tacc_run_spf_container.sh` — reusable helper that runs a command in the SPF container

**Key config (edit in `tacc_run_spf_with_vllm.sh`):**

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE` | `$SCRATCH/containers/spf_latest.sif` | SPF container SIF |
| `CODE_DIR` | `$WORK/workspaces/semantic-persistence-filter` | Repo root |
| `DATA_DIR` | `$SCRATCH/data/semantic-persistence-filter` | Dataset root |
| `HF_CACHE_DIR` | `$SCRATCH/.cache/huggingface` | HF model cache |
| `VLLM_MODEL` | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Model to serve |
| `VLLM_PORT` | `9006` | Port for vLLM server |
| `VLLM_TP` | `4` | Tensor parallel size |
| `VLLM_GPUS` | `0,1,2,3,4,5,6,7` | GPUs for vLLM |
| `PIPELINE_GPUS` | `0,1,2,3,4,5,6,7` | GPUs for pipeline |
| `CONFIG_NAME` | *(set in script)* | Hydra config name |
| `DATASET_NAME` | `openloris_location_all` | Image dataset config |

### sbatch submission

```bash
cd $WORK/workspaces/semantic-persistence-filter
sbatch tacc_run_spf_with_vllm_batch.slurm
```

To override variables without editing the script, uncomment the export lines at the top of `tacc_run_spf_with_vllm_batch.slurm`, or pass them inline:

```bash
DATASET_NAME=train_mix_0 sbatch tacc_run_spf_with_vllm_batch.slurm
```

Logs go to `logs/spf/spf_job.<jobid>.out` / `.err`.

### idev session (interactive testing)

```bash
# Request a node
idev -p amd-rtx -N 1 -n 1 -t 02:00:00 -A IRI26004

# Once on the node:
cd $WORK/workspaces/semantic-persistence-filter
./tacc_run_spf_with_vllm.sh
```

Override any variable before running:
```bash
DATASET_NAME=train_mix_0 VLLM_PORT=9007 ./tacc_run_spf_with_vllm.sh
```

---

## TRT-LLM + DAM variant

Runs segmentation + object descriptions phases only. The VLM object discoverer (QwenVL) is served via TRT-LLM; the object describer (DAM) is served via the DAM container. Both servers run in Apptainer. The pipeline runs in the SPF container.

**Scripts:**
- `tacc_run_spf_trtllm_dam_seg_desc_batch.slurm` — sbatch wrapper
- `tacc_run_spf_trtllm_dam_seg_desc.sh` — driver (starts TRT-LLM + DAM, waits, runs pipeline)
- `tacc_run_spf_container.sh` — reusable helper (shared with vLLM variant)

**Key config (edit in `tacc_run_spf_trtllm_dam_seg_desc.sh`):**

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE` | `$SCRATCH/containers/spf_latest.sif` | SPF container SIF |
| `TRTLLM_IMAGE` | `$SCRATCH/containers/tritonserver_26.03-trtllm-python-py3.sif` | TRT-LLM SIF |
| `DAM_IMAGE` | `$SCRATCH/containers/describe-anything_latest.sif` | DAM SIF |
| `CODE_DIR` | `$WORK/workspaces/semantic-persistence-filter` | Repo root |
| `DATA_DIR` | `$SCRATCH/data/semantic-persistence-filter` | Dataset root |
| `HF_CACHE_DIR` | `$SCRATCH/.cache/huggingface` | HF model cache |
| `TRTLLM_MODEL` | `Qwen/Qwen3-VL-8B-Thinking` | VLM model to serve |
| `TRTLLM_PORT` | `9011` | Port for TRT-LLM server |
| `TRTLLM_TP` | `4` | Tensor parallel size |
| `DAM_PORT` | `9014` | Port for DAM server |
| `TRTLLM_GPUS` | `0,1,2,3` | GPUs for TRT-LLM |
| `DAM_GPUS` | `4` | GPU(s) for DAM |
| `PIPELINE_GPUS` | `5,6,7` | GPUs for pipeline (SAM3) |
| `CONFIG_NAME` | `qwen3_30b_t_lang_q3T_pU3_sam3batch_maxObj_30_minConf_0p35_mA0p001_dedupe0p9_v3_iqr_only` | Hydra config name |
| `DATASET_NAME` | `train_mix_0` | Image dataset config |

> **Note on first-run time:** TRT-LLM JIT-compiles GPU kernels for the model on first use. This can take **10–30 minutes** before the server is ready. The script waits up to 30 minutes. Subsequent runs reuse compiled kernels from the HF cache and start much faster.

### sbatch submission

```bash
cd $WORK/workspaces/semantic-persistence-filter
sbatch tacc_run_spf_trtllm_dam_seg_desc_batch.slurm
```

To override variables, uncomment the export lines at the top of `tacc_run_spf_trtllm_dam_seg_desc_batch.slurm`, or pass inline:

```bash
DATASET_NAME=other_dataset sbatch tacc_run_spf_trtllm_dam_seg_desc_batch.slurm
```

Logs go to `logs/spf/spf-seg-desc.<jobid>.out` / `.err`, with separate server logs at `logs/spf/trtllm_<jobid>.out` and `logs/spf/dam_<jobid>.out`.

### idev session (interactive testing)

```bash
# Request a node
idev -p amd-rtx -N 1 -n 1 -t 02:00:00 -A IRI26004

# Once on the node:
cd $WORK/workspaces/semantic-persistence-filter
./tacc_run_spf_trtllm_dam_seg_desc.sh
```

Override any variable before running:
```bash
DATASET_NAME=train_mix_0 TRTLLM_PORT=9012 ./tacc_run_spf_trtllm_dam_seg_desc.sh
```

---

## Checking job status

```bash
# List your running/pending jobs
squeue -u $USER

# Follow live output from a running job
tail -f logs/spf/spf-seg-desc.<jobid>.out

# Follow TRT-LLM or DAM server logs specifically
tail -f logs/spf/trtllm_<jobid>.out
tail -f logs/spf/dam_<jobid>.out

# Cancel a job
scancel <jobid>
```
