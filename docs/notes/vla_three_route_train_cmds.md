# VLA 三条路线 · 训练指令速查

工作目录：`code/HumanoidWBM/demo/VLA/Litchi-Isaac-GR00T`  
数据根：`/home/zhenyu/workspace/data/demo/dataset/<route>/<skill>/`  
默认 skill：`walk_towards_the_green_table`

## 三条路线（必选其一）

| 旗标 | 监督 token 来源 | 数据路径 |
|------|-----------------|----------|
| `--origin` | 数采 live `action.motion_token` | `dataset/origin/<skill>/` |
| `--rewrite` | 事后 sonic_v1_1 离线 ONNX 重编码 | `dataset/rewrite/<skill>/`（需先 prepare） |
| `--sim` | 事后 MuJoCo 跑 Litchi online-v3 encode 覆写 | `dataset/sim/<skill>/`（需先 prepare） |

相机 / proprio 三路相同，差的只是 token。输出目录自动带后缀 `_origin` / `_rewrite` / `_sim`。

## 训练（最简）

```bash
cd /home/zhenyu/workspace/code/HumanoidWBM/demo/VLA/Litchi-Isaac-GR00T

# --- 带 training RTC（推荐；默认 WANDB online）---
WANDB_MODE=online bash scripts/train_walk_towards_green_table_rtc_8gpu.sh --origin
WANDB_MODE=online bash scripts/train_walk_towards_green_table_rtc_8gpu.sh --rewrite
WANDB_MODE=online bash scripts/train_walk_towards_green_table_rtc_8gpu.sh --sim

# --- 不带 training RTC（默认 WANDB offline）---
bash scripts/train_walk_towards_green_table_8gpu.sh --origin
WANDB_MODE=online bash scripts/train_walk_towards_green_table_8gpu.sh --origin

# 只跑选路/数据闸门，不启动 torchrun
GATE_ONLY=1 bash scripts/train_walk_towards_green_table_rtc_8gpu.sh --origin
```

默认输出（RTC / origin 例）：  
`/home/zhenyu/workspace/data/demo/outputs/walk_towards_the_green_table_skill1_all_rtc_2cam_origin`

## 参数 / 环境变量含义

**脚本后 CLI**

| 参数 | 含义 |
|------|------|
| `--origin` / `--rewrite` / `--sim` | 选路线（互斥；也可用 `--route origin` 或 `ROUTE=origin`） |
| `--skill NAME` | skill 名，默认 `walk_towards_the_green_table`；决定读哪棵 `dataset/<route>/<skill>/` |

**环境变量（常用）**

| 变量 | 含义 | 默认 |
|------|------|------|
| `WANDB_MODE` | `online` 上传 / `offline` 本地 | RTC 脚本=`online`；无 RTC=`offline` |
| `HTTPS_PROXY` | online 时访问 wandb 的代理 | RTC 倾向 `7897`；无 RTC 倾向 `7890` |
| `WANDB_API_KEY` | wandb key（或写在 `../.wandb_api_key`） | 必填 |
| `WANDB_PROJECT` / `WANDB_ENTITY` | wandb 项目 / entity | 脚本内默认 |
| `OUTPUT_DIR` / `EXPERIMENT_NAME` | 覆盖输出目录与实验名 | 按 skill+rtc+route 自动拼 |
| `NUM_GPUS` / `MAX_STEPS` / `SAVE_STEPS` | 卡数 / 总步 / 存盘间隔 | `8` / `7376` / `1844` |
| `GR00T_MODEL_PATH` | 基座权重 | `checkpoints/GR00T-N1.7-3B` |
| `GATE_ONLY=1` | 只校验路线与数据，不训练 | 关 |
| `DATASET_BASE` | 数据根父目录 | `.../data/demo/dataset` |

**两套脚本差别**

| | `..._rtc_8gpu.sh` | `..._8gpu.sh`（无 rtc） |
|--|------------------|------------------------|
| training RTC | 开：`--rtc_training`，H=40，s=12，overlap/max_delay=28，frozen=5，ramp=3.0 | 关 |
| 默认输出名 | `..._skill1_all_rtc_2cam_<route>` | `..._skill1_all_2cam_<route>` |
| 默认 wandb | online | offline |

## prepare（rewrite / sim 才需要）

```bash
VLA=/home/zhenyu/workspace/code/HumanoidWBM/demo/VLA
bash $VLA/scripts/prepare_route_dataset.sh --rewrite --force
NET_IFACE=eth1 DEPLOY_CUDA_VISIBLE_DEVICES=7 \
  bash $VLA/scripts/prepare_route_dataset.sh --sim --force
```

更完整的路线说明见：`demo/VLA/scripts/ROUTES.md`。
