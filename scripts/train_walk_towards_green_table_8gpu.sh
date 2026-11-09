#!/usr/bin/env bash
# Skill-1: walk_towards_the_green_table — 8-GPU finetune (UNITREE_G1_SONIC)
#
# Inputs: ego_view + head cameras + joint/state (see unitree_g1_sonic modality).
# User params:
#   lr=1e-4, 4 epochs, save 1 ckpt / epoch, keep last 3, WandB on
#   Dataset: ALL sessions under walk_towards_the_green_table
#     (incl. previously excluded 2026-11-05-13-35-11 / 14-05-17 / 14-17-12)
#
# Epoch→step mapping (action horizon=40, valid = length-AH+1):
#   14 sessions, valid steps ≈ 59016, GLOBAL_BATCH_SIZE=32
#   steps/epoch = 59016 // 32 = 1844
#   max_steps   = 1844 * 4    = 7376
#   save_steps  = 1844
#   save_total_limit = 3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLA_ROOT="$(cd "${REPO}/.." && pwd)"

# shellcheck disable=SC1091
source "${VLA_ROOT}/activate_gr00t.sh"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  _WANDB_KEY_FILE="${VLA_ROOT}/.wandb_api_key"
  if [[ -f "${_WANDB_KEY_FILE}" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "${_WANDB_KEY_FILE}")"
    export WANDB_API_KEY
  fi
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set. Export it or put it in ${VLA_ROOT}/.wandb_api_key" >&2
  exit 1
fi

ROOT="/home/zhenyu/workspace/data/demo/dataset/walk_towards_the_green_table"
# Include real dirs and symlinked session dirs (find -type d skips symlinks).
# Exclude token-backup folders (not LeRobot sessions).
DATASET_PATH="$(
  {
    find "${ROOT}" -mindepth 1 -maxdepth 1 -type d
    find "${ROOT}" -mindepth 1 -maxdepth 1 -type l
  } | sort -u | while read -r d; do
    [[ -d "${d}" ]] || continue
    base="$(basename "${d}")"
    [[ "${base}" == _* ]] && continue
    [[ -f "${d}/meta/info.json" ]] || continue
    printf '%s\n' "${d}"
  done | paste -sd: -
)"
BASE_MODEL_PATH="${GR00T_MODEL_PATH:-${REPO}/checkpoints/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhenyu/workspace/data/demo/outputs/walk_towards_the_green_table_skill1_all_2cam}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-walk_towards_green_table_skill1_all_2cam}"
WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7}"
MASTER_PORT="${MASTER_PORT:-29511}"
# Bind c10d/TCPStore to loopback. Using the pod hostname previously left
# orphaned workers retrying a dead store after the torchrun parent exited
# (seen as: TCP client timed out connecting to <hostname>:<port>).
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-7376}"
SAVE_STEPS="${SAVE_STEPS:-1844}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO}"

# Online wandb needs a live local HTTP proxy; 7890 is often dead on this box.
# Override, e.g.: HTTPS_PROXY=http://127.0.0.1:7897 WANDB_MODE=online bash ...
PROXY_URL="${HTTPS_PROXY:-${https_proxy:-http://127.0.0.1:7890}}"
if [[ "${WANDB_MODE:-offline}" == "online" ]]; then
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  # Keep distributed / localhost / cluster traffic off the proxy.
  export no_proxy="localhost,127.0.0.1,${MASTER_ADDR},10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.aliyuncs.com,*.aliyun.com"
  export NO_PROXY="$no_proxy"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export no_proxy='*' NO_PROXY='*'
fi

export USE_WANDB=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export GROOT_HF_LOCAL_FIRST=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_START_METHOD=thread
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
export WANDB_SILENT=false
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Force store/rendezvous onto loopback for every rank.
export MASTER_ADDR
export MASTER_PORT
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

echo "DATASET sessions: $(echo "${DATASET_PATH}" | tr ':' '\n' | wc -l)"
echo "${DATASET_PATH}" | tr ':' '\n' | sed 's|.*/||' | sed 's/^/  - /'
echo "BASE_MODEL: ${BASE_MODEL_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}  lr=1e-4  max_steps=${MAX_STEPS}  save_steps=${SAVE_STEPS}  save_total_limit=${SAVE_TOTAL_LIMIT}  gbs=32"
echo "RDZV: ${MASTER_ADDR}:${MASTER_PORT}"
echo "WANDB_MODE=${WANDB_MODE} proxy=${https_proxy:-none}; TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}; visible GPUs:"
nvidia-smi -L
echo "WANDB_API_KEY is set (len=${#WANDB_API_KEY})"
if [[ "${WANDB_MODE}" == "online" ]]; then
  if ! curl -fsS --max-time 15 -x "${https_proxy}" https://api.wandb.ai/healthz >/dev/null; then
    echo "ERROR: wandb online requested but proxy ${https_proxy} cannot reach api.wandb.ai." >&2
    echo "       Start the local proxy (or SSH -D/-L tunnel), then rerun with HTTPS_PROXY=..." >&2
    exit 2
  fi
  echo "wandb proxy healthz OK via ${https_proxy}"
fi

# Explicit c10d rdzv on loopback (avoid --standalone hostname store).
exec torchrun \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${EXPERIMENT_NAME}" \
  gr00t/experiment/launch_finetune.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --embodiment_tag UNITREE_G1_SONIC \
  --num_gpus "${NUM_GPUS}" \
  --output_dir "${OUTPUT_DIR}" \
  --experiment_name "${EXPERIMENT_NAME}" \
  --wandb_project "${WANDB_PROJECT}" \
  --use_wandb \
  --learning_rate 1e-4 \
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --global_batch_size 32 \
  --dataloader_num_workers 4 \
  --shard_size 1024 \
  --num_shards_per_epoch 100000 \
  --episode_sampling_rate 0.1
