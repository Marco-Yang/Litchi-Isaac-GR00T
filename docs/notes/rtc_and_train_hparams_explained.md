# RTC / 训练超参通俗说明

本文解释本仓库 GR00T N1.7 微调与部署里常见的一组参数：  
`action_horizon`、`rtc_execution_horizon`、训练 `max_delay`、`rtc_frozen_steps`、`rtc_overlap_steps`、`rtc_ramp_rate`，以及 `gpu batch_size`、`dataloader workers`。

默认数值以当前 RTC 训练脚本为准：

```text
H=40, s=12, max_delay=28, overlap=28, frozen=5, ramp=3.0
```

---

## 先建立一张时间轴直觉

把机器人动作想成「一次开出一张未来行程单」：

```text
一次预测的动作块（chunk），长度 = action_horizon = H = 40 步

时间 →
[ 0  1  2 ... 11 | 12 13 ... 39 ]
  └─ 马上要执行 ─┘   └── 还没执行到、留给下一次衔接 ──┘
       s=12                  典型 overlap = H - s = 28
```

部署时机器人一边执行旧行程，一边在后台算新行程。两张行程单必须「接得住」，这就是 RTC（Real-Time Chunking）。

---

## 1. `action_horizon`（H）

**一句话：** 模型一次预测多少步未来动作。

**逻辑：**  
VLA 不是每一步都重新想一遍，而是一次性吐出一串动作（chunk）。这个串有多长，就是 `action_horizon`。

**本仓库默认：** `H = 40`（约 40 个控制拍）。

**例子：**  
假如控制频率是 20Hz，`H=40` 大约覆盖 2 秒未来。  
模型看一眼相机和状态，说：「接下来 40 拍请这样动。」

**调大 / 调小：**

| 更大 H | 更小 H |
|--------|--------|
| 看得更远，衔接窗口更大 | 反应更快、更短视 |
| 算力/显存更贵 | 更容易抖、更频繁重规划 |

---

## 2. `rtc_execution_horizon`（s）

**一句话：** 这一张行程单里，计划「先老老实实执行」的前缀长度。

**逻辑：**  
异步推理时，你不会把整整 40 步都跑完再换新计划；通常执行前 `s` 步左右，就该换上下一张（或开始融合重叠区）。  
训练里它主要用来约束：`max_delay ≤ H - s`，保证「延迟」不会长到把整块都冻死、后面没东西可学。

**本仓库默认：** `s = 12`。

**例子：**  
`H=40, s=12` ≈ 「先按旧计划走 12 拍；同时后台算新计划；剩下 28 拍是两张计划的重叠缓冲区。」

```text
旧计划:  [========执行 s=12========|========重叠区 28========]
新计划:                            [========重叠区 28========|====新尾巴====]
```

---

## 3. 训练 `rtc_max_delay`（Train max_delay）

**一句话：** 训练时模拟「推理延迟最多有多长」——随机把前缀当成「已经定了、别改了」。

**逻辑（训练专用）：**  
真实部署时，你发起推理的那一刻，前面若干步其实已经在执行或即将执行，新模型不该再去改它们。  
训练开启 `--rtc_training` 后，每个样本随机抽延迟  
`d ~ Uniform[0, max_delay)`，然后：

- 前 `d` 步：当成硬条件（prefix），几乎不再靠新噪声去学
- 后面的步：正常去噪，loss 主要打在这里

约束：`max_delay ≤ H - s`。  
默认常取 `max_delay = H - s = 28`。

**注意：**  
`max_delay` 是**训练**里学「带延迟的衔接」；  
下面的 `overlap / frozen / ramp` 是**推理部署**里怎么软融合。  
脚本里 overlap 也常写成 28，数值碰巧相同，但职责不同。

**例子：**  
某次训练抽到 `d=5`：  
「假装推理花了 5 拍时间，前 5 拍已经钉死，模型只负责把第 6–40 拍补圆滑。」  
下次抽到 `d=20`：钉死前 20 拍，学更长延迟下的衔接。

---

## 4. `rtc_overlap_steps`（overlap）

**一句话：** 推理时，新计划要和旧计划重叠多少步来「对缝」。

**逻辑（部署）：**  
新 chunk 开头不是从纯噪声瞎生成，而是把旧 chunk **还没执行完的尾巴**贴到新 chunk 开头，当作 inpainting 条件，避免两段动作在接缝处跳变。

**本仓库默认：** `overlap = 28`（常见取 `H - s`）。

**例子：**  
旧计划还剩最后 28 步没跑完；新计划前 28 步先用旧尾巴填上，再在融合规则下改/保。  
如果 `overlap=0`，就是硬切到全新预测，手臂可能突然抖一下。

```text
旧 chunk 尾巴 (28步) ──┐
                       ├─→ 贴进新 chunk 前缀，做 RTC 融合
新观测算出的意图 ──────┘
```

**训练时：** 这个值会写进 checkpoint 元数据，**不直接进 train loss**。

---

## 5. `rtc_frozen_steps`（frozen）

**一句话：** 重叠区里，最前面几步「完全不许改」——通常对应推理延迟。

**逻辑（部署）：**  
overlap 区不是整段都软改。最前面 `frozen` 步对应「你算新计划这段时间里，机器人其实已经在走/马上必走」的动作，必须锁死；  
后面 overlap 里剩下的步，才允许按 ramp 逐渐放开。

**本仓库默认：** `frozen = 5`。

**例子：**  
推理大约要 5 个控制拍。那 overlap 前 5 步 `vel_strength=0`（新模型的修正速度为 0），等于「这 5 步照旧计划走」；  
从第 6 步开始才慢慢允许向新计划靠拢。

```text
overlap=28:
[ 0..4 冻结 frozen=5 ][ 5..27 按 ramp 逐渐放开 ][ 28..39 全新生成 ]
```

`frozen` 太小：延迟还没结束就改动作 → 可能抖。  
`frozen` 太大：新观测迟迟进不来 → 反应钝。

---

## 6. `rtc_ramp_rate`（ramp）

**一句话：** 冻结区之后，从「完全听旧计划」过渡到「完全听新计划」有多快。

**逻辑（部署）：**  
在 `frozen` 与 `overlap` 之间的中间段，用指数 ramp 控制修正强度：

```text
ramp ≈ 归一化后的 (1 - exp(-rate * t))
```

- `rate` 更大：很快就贴近新计划（过渡陡）
- `rate` 更小：更长时间迁就旧计划（过渡缓）

**本仓库默认：** `ramp_rate = 3.0`。

**例子（直觉）：**  
像两车道并线：

| ramp | 体感 |
|------|------|
| 很小 | 很晚才并过去，平滑但可能偏旧意图 |
| 3.0（默认） | 中间一段平滑并线 |
| 很大 | 很快并到新车道，更跟新观测，但接缝可能更「硬」 |

---

## 训练 RTC vs 部署 RTC（别混）

| 参数 | 主要用在 | 干什么 |
|------|----------|--------|
| `action_horizon` H | 训练 + 部署 | 一次预测多长 |
| `rtc_execution_horizon` s | 训练约束 / 部署节奏 | 计划执行前缀；约束 `max_delay ≤ H-s` |
| `rtc_max_delay` | **仅训练** | 随机模拟延迟，硬条件 prefix |
| `rtc_overlap_steps` | **主要部署** | 新旧 chunk 重叠多长 |
| `rtc_frozen_steps` | **主要部署** | 重叠里锁死前几步 |
| `rtc_ramp_rate` | **主要部署** | 重叠里软过渡斜率 |

你们脚本里的典型搭配：

```text
训练：H=40, s=12, max_delay=28
部署元数据：overlap=28, frozen=5, ramp=3.0
```

---

## 7. GPU `batch_size`（本仓库里的 global / per-GPU）

**一句话：** 每一步优化，GPU 一次看多少条样本。

**逻辑：**  
- `per_gpu_batch_size`：单卡一个 step 的样本数  
- `global_batch_size`：所有卡加起来（≈ `per_gpu × num_gpus`，再乘 gradient accumulation）

一步里样本越多，梯度越稳，显存越贵；**不等于效果一定更好**。

**例子（你们 pick_toy 8 卡）：**

```text
per_gpu_batch = 32
num_gpus = 8
→ global_batch = 256
```

脚本还会按 batch 缩放 `max_steps`，让「看过的样本总量 / epoch」大致可比：

```text
batch 变大 → 同样 epoch 需要的 step 变少
```

所以：

- **只加大 batch、不减 steps**：墙钟未必更快，每步还可能更慢  
- **加大 batch 且按脚本缩 steps**：才有机会更早跑完（前提是吞吐真的上去）

显存还空很多时，可以试更大 batch；但要盯 samples/s 和 loss/下游，不要只看显存占用。

---

## 8. `dataloader_num_workers`（每 rank）

**一句话：** 每张 GPU 旁边有几个 CPU 进程专门「提前把数据准备好」。

**逻辑：**  
训练主进程在 GPU 上算；workers 在 CPU 上读 shard、做增广、拼 batch。  
本仓库的 `ShardedMixtureDataset` 还会按

```text
world_size × num_workers
```

把 shard 分给各卡各 worker，避免大家抢同一份。

**例子：**

```text
8 卡 × 16 workers/rank = 128 个数据进程
单卡 × 24 workers/rank = 24 个数据进程
```

| 太小 | 太大 |
|------|------|
| GPU 经常空等数据 | CPU/内存/磁盘打满，收益变小甚至变慢 |
| step 抖动 | 每个 worker 还 cache shard，内存涨 |

经验：本机 8 卡常见 **8–16/rank**；单卡常见 **8–12**，再高要靠实测。  
**CPU 利用率高本身不会把模型训坏**；改 worker 数会轻微改变数据切分顺序（类似换 seed）。

---

## 一张总图（部署 RTC）

```text
action_horizon H = 40
┌──────────────────────────────────────────────────────────┐
│ 新 chunk                                                  │
│ [ frozen=5 | ramp 过渡 (overlap-frozen) | 全新尾部 ]       │
│ └──────────── overlap=28 ────────────┘                    │
└──────────────────────────────────────────────────────────┘
         ↑ 前 overlap 步来自「旧 chunk 未执行尾巴」
```

训练时对应直觉：用 `max_delay` 随机假装「前面已经走了一截」，逼模型学会在延迟下把后面接好。

---

## 快速对照表

| 名字 | 默认 | 人话 |
|------|------|------|
| `action_horizon` | 40 | 一次预测多长 |
| `rtc_execution_horizon` | 12 | 计划先执行多长 / 训练延迟上限锚点 |
| `rtc_max_delay` | 28 | 训练时最多模拟多长延迟 |
| `rtc_overlap_steps` | 28 | 部署时新旧计划重叠多长 |
| `rtc_frozen_steps` | 5 | 重叠里前几步完全锁死（≈推理延迟） |
| `rtc_ramp_rate` | 3.0 | 锁死后向新计划并线的陡峭程度 |
| GPU batch size | 视任务 | 一步看多少样本（影响显存/步数/稳定性） |
| dataloader workers | 2–16 常见 | 每卡几个 CPU 帮你预取数据 |

---

## 相关代码入口

- 训练 RTC loss：`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`（`rtc_training` / `rtc_max_delay`）
- 推理 soft RTC：同文件 `get_action` 中 `rtc_overlap_steps` / `frozen` / `ramp`
- 微调 CLI 默认：`gr00t/configs/finetune_config.py`
- 8 卡 RTC 脚本：`scripts/train_walk_towards_green_table_rtc_8gpu.sh`
