#!/usr/bin/env bash
# Skill-1 (server 32): walk_towards_the_green_table — 8-GPU finetune WITH training-time RTC
#
# Dataset: 2026-11-05-15-17-43_new .. 2026-11-05-20-36-57
#   (exclude 13-35-11, 14-05-17, 14-17-12)
#
# Training-time RTC (arXiv:2512.05964 Algorithm 1):
#   H=40, s=12, max_delay=H-s=28 (=overlap), deploy meta frozen=5, ramp=3.0
#
# Match non-RTC schedule: gbs=32, lr=1e-4, 4 epochs, 1 ckpt/epoch, keep last 3.
# Epoch→step (valid≈50197 with 11 sessions incl. 2026-11-06-*, gbs=32):
#   steps/epoch = 50197 // 32 = 1568
#   max_steps   = 1568 * 4    = 6272
#   save_steps  = 1568
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
  echo "ERROR: WANDB_API_KEY is not set." >&2
  exit 1
fi

ROOT="/home/zhenyu/workspace/data/demo/dataset/walk_towards_the_green_table"
EXCLUDE_SESSIONS=(
  "2026-11-05-13-35-11"
  "2026-11-05-14-05-17"
  "2026-11-05-14-17-12"
)
# Include real dirs and symlinked session dirs (find -type d skips symlinks).
DATASET_PATH="$(
  {
    find "${ROOT}" -mindepth 1 -maxdepth 1 -type d
    find "${ROOT}" -mindepth 1 -maxdepth 1 -type l
  } | sort -u | while read -r d; do
    # Keep only entries that resolve to a directory.
    [[ -d "${d}" ]] || continue
    base="$(basename "${d}")"
    skip=0
    for ex in "${EXCLUDE_SESSIONS[@]}"; do
      if [[ "${base}" == "${ex}" ]]; then skip=1; break; fi
    done
    if [[ "${skip}" -eq 0 ]]; then
      printf '%s\n' "${d}"
    fi
  done | paste -sd: -
)"
BASE_MODEL_PATH="${GR00T_MODEL_PATH:-${REPO}/checkpoints/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhenyu/workspace/data/demo/outputs/walk_towards_the_green_table_skill1_rtc}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-walk_towards_green_table_skill1_rtc}"
WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29541}"
NUM_GPUS="${NUM_GPUS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO}"

export WANDB_MODE="${WANDB_MODE:-offline}"
if [[ "${WANDB_MODE}" == "online" ]]; then
  PROXY_URL="${HTTPS_PROXY:-${https_proxy:-http://127.0.0.1:7890}}"
  export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}"
  export no_proxy="localhost,127.0.0.1,${MASTER_ADDR},10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
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
export WANDB_START_METHOD=thread
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MASTER_ADDR MASTER_PORT

echo "DATASET sessions: $(echo "${DATASET_PATH}" | tr ':' '\n' | wc -l)"
echo "${DATASET_PATH}" | tr ':' '\n' | sed 's|.*/||' | sed 's/^/  - /'
echo "BASE_MODEL: ${BASE_MODEL_PATH}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "RDZV: ${MASTER_ADDR}:${MASTER_PORT}"
echo "NUM_GPUS=${NUM_GPUS}  gbs=32  lr=1e-4  max_steps=6272  save_steps=1568  save_total_limit=3  workers=${DATALOADER_NUM_WORKERS}"
echo "RTC training: H=40 s=12 max_delay=28 | deploy meta overlap=28 frozen=5 ramp=3.0"
echo "WANDB_MODE=${WANDB_MODE}"
nvidia-smi -L

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
  --max_steps 6272 \
  --save_steps 1568 \
  --save_total_limit 3 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-5 \
  --global_batch_size 32 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --shard_size 1024 \
  --num_shards_per_epoch 100000 \
  --episode_sampling_rate 0.1 \
  --rtc_training \
  --rtc_execution_horizon 12 \
  --rtc_max_delay 28 \
  --rtc_overlap_steps 28 \
  --rtc_frozen_steps 5 \
  --rtc_ramp_rate 3.0
