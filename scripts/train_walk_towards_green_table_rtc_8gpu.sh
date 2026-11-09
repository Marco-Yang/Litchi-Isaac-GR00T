#!/usr/bin/env bash
# Skill-1: walk_towards_the_green_table — 8-GPU finetune WITH training-time RTC
#
# Inputs: ego_view + head cameras + joint/state (see unitree_g1_sonic modality).
# Dataset: ALL sessions under walk_towards_the_green_table
#   (incl. previously excluded 2026-11-05-13-35-11 / 14-05-17 / 14-17-12)
#
# Training-time RTC (arXiv:2512.05964 Algorithm 1):
#   H=40, s=12, max_delay=H-s=28 (=overlap), deploy meta frozen=5, ramp=3.0
#
# Match non-RTC schedule: gbs=32, lr=1e-4, 4 epochs, 1 ckpt/epoch, keep last 3.
# Epoch→step (valid≈59016 with 14 sessions, gbs=32):
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
  echo "ERROR: WANDB_API_KEY is not set." >&2
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
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhenyu/workspace/data/demo/outputs/walk_towards_the_green_table_skill1_all_rtc_2cam}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-walk_towards_green_table_skill1_all_rtc_2cam}"
WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29541}"
NUM_GPUS="${NUM_GPUS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
MAX_STEPS="${MAX_STEPS:-7376}"
SAVE_STEPS="${SAVE_STEPS:-1844}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO}"

export WANDB_MODE="${WANDB_MODE:-online}"
if [[ "${WANDB_MODE}" == "online" ]]; then
  # Prefer live local proxy (7897 on this box; 7890 often dead).
  # activate_gr00t.sh sets no_proxy='*' — must override so wandb can egress.
  PROXY_URL="${HTTPS_PROXY:-${https_proxy:-http://127.0.0.1:7897}}"
  export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}"
  export no_proxy="localhost,127.0.0.1,${MASTER_ADDR},10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
  export NO_PROXY="$no_proxy"
  export WANDB_ENTITY="${WANDB_ENTITY:-3155853275-national-university-of-singapore-students-union}"
  # Use python/requests (curl UA is often 403'd by the local proxy/CF).
  # Local proxy is flaky (SSL EOF / timeouts) — retry before aborting.
  if ! python - <<'PY'
import os, sys, time, requests
proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
proxies = {"http": proxy, "https": proxy}
key = os.environ.get("WANDB_API_KEY", "")
last = None
for attempt in range(1, 8):
    try:
        r = requests.get("https://api.wandb.ai/healthz", proxies=proxies, timeout=45)
        r.raise_for_status()
        g = requests.post(
            "https://api.wandb.ai/graphql",
            proxies=proxies,
            timeout=45,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"query": "query { viewer { username entity } }"},
        )
        g.raise_for_status()
        data = g.json()
        viewer = (data.get("data") or {}).get("viewer") or {}
        if not viewer.get("username"):
            raise RuntimeError(f"wandb viewer empty: {data}")
        print(
            f"wandb online OK user={viewer.get('username')} "
            f"entity={viewer.get('entity')} via {proxy} (attempt {attempt})"
        )
        break
    except Exception as e:
        last = e
        print(f"wandb online check attempt {attempt}/7 failed: {e}", file=sys.stderr)
        time.sleep(min(2 * attempt, 10))
else:
    print(f"ERROR: wandb online check failed after retries: {last}", file=sys.stderr)
    sys.exit(1)
PY
  then
    exit 1
  fi
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
echo "NUM_GPUS=${NUM_GPUS}  gbs=32  lr=1e-4  max_steps=${MAX_STEPS}  save_steps=${SAVE_STEPS}  save_total_limit=${SAVE_TOTAL_LIMIT}  workers=${DATALOADER_NUM_WORKERS}"
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
  --max_steps "${MAX_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
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
