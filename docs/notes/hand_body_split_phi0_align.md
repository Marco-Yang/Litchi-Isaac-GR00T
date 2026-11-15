# Hand/body action-head split + Phi0 RTC align

Branch: `hand-body-split`

## Phi0 事实（对齐对象）

Phi0（`Phi_0_wpy`）**没有**独立的 hand/body 神经网络头：单 ACT 输出 512-d，用 slice / 加权 loss 区分 sonic64 与 revo2_12。

| 项 | Phi0 生产/蒸馏 | 本分支落点 |
|---|---|---|
| 网络头 | 单 head | **扩展**：共享 DiT + **双 decoder**（body/hand） |
| 权重 | 离线 flat 逐维；蒸馏常用 `W_Z:W_HAND=1:1`（组均值） | 默认 `body_w=hand_w=1.0`（组均值，抬高手部相对 flat 64:14） |
| 训练 RTC | hard prefix，`max_delay=8`，`s=15`，`d≈6` | 同：`s=15`，`max_delay=8`，`frozen=6` |
| 部署 RTC | Phi0 仅 hard（yaml `exponential` 被忽略） | Litchi 仍保留 soft meta：`overlap=25`（`H-s`），`ramp=3.0` |
| H | 常用 32 | 保持 **40**（modality / 现有 ckpt） |

手模态：Phi0 Revo2=12；Litchi SONIC=`7+7=14`。权重按 **hand group**，不硬套 12 维。

## 实现

1. **双 decoder**（共享 DiT）  
   - body: `motion_token` `[0:64)`  
   - hand: L/R joints `[64:78)`  
   - finetune：从 base joint `action_decoder` 切片拷贝初始化

2. **组均值 MSE**  
   `loss = body_w * mean(body) + hand_w * mean(hand)`  
   （对齐 Phi0 distill 1:1；相对 legacy flat 等效抬高手部）

3. **RTC 默认**（相对旧 promptfix `s=12/delay=28/frozen=5`）  
   - `rtc_execution_horizon=15`  
   - `rtc_max_delay=8`（需 `≤ H-s=25`）  
   - `rtc_frozen_steps=6`  
   - `rtc_overlap_steps=25`  
   - `rtc_ramp_rate=3.0`

## 训练

```bash
cd /home/zhenyu/workspace/code/HumanoidWBM/demo/VLA/Litchi-Isaac-GR00T

WANDB_MODE=online bash scripts/train_walk_towards_green_table_rtc_8gpu.sh \
  --origin --skill pick_toy
```

默认输出：`.../data/demo/outputs/skill2_pick_toy_hand_body_split`

常用覆盖：

```bash
# 旧 RTC
RTC_EXECUTION_HORIZON=12 RTC_MAX_DELAY=28 RTC_OVERLAP_STEPS=28 RTC_FROZEN_STEPS=5 ...

# 关双 decoder（退回 joint + 仍可用权重需改代码；当前关 split 即 flat MSE）
SPLIT_HAND_BODY_ACTION_HEAD=0 ...

# 再抬高手
HAND_LOSS_WEIGHT=2.0 ...
```

## 参考路径

- Phi0 RTC：`Phi_0_wpy/configs/model/phi0_wholebody_best_rtc.yaml`，`src/phi0/inference/rtc.py`
- Phi0 distill 权重：`Phi_0_wpy/src/phi0/online/losses.py`（`w_z` / `w_hand`）
- Litchi：`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`，`scripts/train_walk_towards_green_table_rtc_8gpu.sh`
