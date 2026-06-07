# Nanochat 训练环境部署与模型训练指南

---

## 目录

1. [环境准备与依赖安装](#1-环境准备与依赖安装)
2. [数据准备](#2-数据准备)
3. [分词器训练与评估](#3-分词器训练与评估)
4. [基础模型训练](#4-基础模型预训练)
5. [继续训练（中期训练）](#5-继续训练中期训练)
6. [SFT 微调](#6-sft-微调监督微调)
7. [模型服务部署](#7-模型服务部署)
8. [常见问题排查](#8-常见问题排查)

---

## 1. 环境准备与依赖安装

### 1.1 基础环境要求

- GPU：NVIDIA GPU（支持 CUDA）
- 宿主机 OS：Ubuntu 22.04 或类似 Linux 发行版
- 容器启动命令（必须包含 GPU 支持和特权模式）

### 1.2 安装基础工具

```bash
apt update && apt install -y git pciutils

# 安装 pyarrow（数据集处理依赖）
pip install pyarrow
```

> **说明**：若不安装 pyarrow，运行 `python -m nanochat.dataset` 时会报 `ModuleNotFoundError`。

### 1.3 创建目录并克隆项目

```bash
# 创建项目目录和缓存目录
mkdir -p ~/projects
mkdir -p ~/.cache/nanochat

# 安装 uv 包管理器
pip install uv

# 克隆 nanochat 仓库
cd ~/projects
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

> **说明**：
> - `~/projects` - 用于存放项目代码
> - `~/.cache/nanochat` - nanochat 的缓存目录，用于存放数据集、分词器、模型检查点等

### 1.4 安装额外依赖

```bash
pip install rustbpe tiktoken wandb fastapi uvicorn
```

### 1.5 创建 Python 虚拟环境

> **重要**：避免回退到系统默认的 Python 3.10

```bash
# 删除可能残留的旧虚拟环境
rm -rf .venv

# 强制使用 Python 3.12 创建虚拟环境
uv venv --python 3.12

# 激活虚拟环境
source .venv/bin/activate
```

> **踩坑点**：直接执行 `uv sync` 可能会因为系统默认 Python 3.10 而重建虚拟环境，导致环境版本回退。务必先用 `uv venv --python 3.12` 锁定版本。

### 1.6 同步项目依赖

```bash
uv sync --python 3.12
```

成功后会输出类似 `Installed 65 packages in 221ms` 的信息。

### 1.7 安装 CUDA 库

运行 `python -m nanochat.dataset -n 8` 时可能遇到 `libcusparseLt.so.0` 和 `libcupti.so.12` 缺失错误。使用 `uv add` 安装所需的 CUDA 库：

```bash
uv add --python 3.12 nvidia-cuda-cupti-cu12
```

成功安装后会输出：

```
Installed 16 packages in 15ms
  + nvidia-cublas-cu12==12.8.4.1
  + nvidia-cuda-cupti-cu12==12.8.90
  + nvidia-cuda-nvrtc-cu12==12.8.93
  + nvidia-cuda-runtime-cu12==12.8.90
  + nvidia-cudnn-cu12==9.10.2.21
  + nvidia-cufft-cu12==11.3.3.83
  + nvidia-cufile-cu12==1.13.1.3
  + nvidia-curand-cu12==10.3.9.90
  + nvidia-cusolver-cu12==11.7.3.90
  + nvidia-cusparse-cu12==12.5.8.93
  + nvidia-cusparselt-cu12==0.7.1
  + nvidia-nccl-cu12==2.27.5
  + nvidia-nvjitlink-cu12==12.8.93
  + nvidia-nvshmem-cu12==3.3.20
  + nvidia-nvtx-cu12==12.8.90
  + triton==3.5.1
```

> 若仍有问题，可手动查找库路径并设置 `LD_LIBRARY_PATH`：
> ```bash
> find / -name "libcusparseLt.so*" 2>/dev/null
> find / -name "libcupti.so.12*" 2>/dev/null
> export LD_LIBRARY_PATH=<库所在目录>:$LD_LIBRARY_PATH
> ```

---

## 2. 数据准备

### 2.1 下载数据集

```bash
python -m nanochat.dataset -n 8
```

成功后会看到多线程下载进度：

```
Downloading 9 shards using 4 workers...
Target directory: /root/.cache/nanochat/base_data_climbmix
Downloading shard_00001.parquet...
Downloading shard_00002.parquet...
...
Successfully downloaded shard_00000.parquet
...
Done! Downloaded: 9/9 shards to /root/.cache/nanochat/base_data_climbmix
```

---

## 3. 分词器训练与评估

### 3.1 训练分词器

```bash
python -m scripts.tok_train --max-chars=2000000000
```

> **注意**：参数使用 `--max-chars=` 格式（带等号），不是 `--max_chars=`

成功训练后会输出：

```
max_chars: 2,000,000,000
doc_cap: 10,000
vocab_size: 32,768
...
Progress: 100% (32503/32503 merges) - Last merge: (115, 4714) -> 32758 (frequency: 842)
Finished training: 32503 merges completed
Training time: 82.89s
Saved tokenizer encoding to /root/.cache/nanochat/tokenizer/tokenizer.pkl
Saved token_bytes to /root/.cache/nanochat/tokenizer/token_bytes.pt
```

### 3.2 评估分词器

```bash
python -m scripts.tok_eval
```

成功评估后输出：

```
Vocab sizes:
GPT-2: 50257
GPT-4: 100277
Ours: 32768

Comparison with GPT-2:
==============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     404     4.50    405     4.49       -0.2%     GPT-2
korean     893      745     1.20    741     1.21       +0.5%     Ours
code       1259     576     2.19    396     3.18      +31.2%     Ours
math       1834     936     1.96    911     2.01       +2.7%     Ours
science    1112     260     4.28    247     4.50       +5.0%     Ours
fwe-train  2948778  631304  4.67    622511  4.74       +1.4%     Ours
fwe-val    3024593  653067  4.63    644939  4.69       +1.2%     Ours

Comparison with GPT-4:
==============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better
                    Tokens  Ratio   Tokens  Ratio   Diff %
-----------------------------------------------------------------------------------------------
news       1819     387     4.70    405     4.49       -4.7%     GPT-4
korean     893      364     2.45    741     1.21     -103.6%     GPT-4
code       1259     309     4.07    396     3.18      -28.2%     GPT-4
math       1834     832     2.20    911     2.01       -9.5%     GPT-4
science    1112     249     4.47    247     4.50       +0.8%     Ours
fwe-train  2948778  611619  4.82    622511  4.74       -1.8%     GPT-4
fwe-val    3024593  631183  4.79    644939  4.69       -2.2%     GPT-4
```

---

## 4. 基础模型预训练

### 4.1 启动训练命令

```bash
python -m scripts.base_train --depth=4 --device-batch-size=32 --num-iterations=500
```

### 4.2 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--depth=4` | 4 | 模型的层数（深度） |
| `--device-batch-size=32` | 32 | 每个设备（GPU）一次处理的数据批次大小 |
| `--num-iterations=500` | 500 | 总共训练的步数 |

### 4.3 训练日志解读

训练开始时会显示模型配置信息：

```
Autodetected device type: cuda
GPU: Tesla V100S-PCIE-32GB | Peak FLOPS (BF16): inf
COMPUTE_DTYPE: torch.float32 (auto-detected: CUDA SM 70 (pre-Ampere, bf16 not supported, using fp32))

Vocab size: 32,768
Model config:
{
  "sequence_len": 2048,
  "vocab_size": 32768,
  "n_layer": 4,
  "n_head": 2,
  "n_kv_head": 2,
  "n_embd": 256,
  "window_pattern": "SSSL"
}
Parameter counts:
wte                     : 8,388,608
value_embeds            : 16,777,216
lm_head                 : 8,388,608
transformer_matrices    : 3,145,776
scalars                 : 34
total                   : 36,700,242
```

训练过程中会输出类似如下的进度日志：

```
step 00100/00500 (20.00%) | loss: 5.178412 | lrm: 1.00 | dt: 2512.04ms | tok/sec: 104,355 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 52 | total time: 3.79m | eta: 16.8m
```

**日志字段含义**：

| 字段 | 含义 |
|------|------|
| `step 00100/00500` | 当前步数/总步数 |
| `(20.00%)` | 完成百分比 |
| `loss: 5.178412` | 当前步的损失值 |
| `lrm: 1.00` | 学习率乘数 (learning rate multiplier) |
| `dt: 2512.04ms` | 该步耗时（毫秒） |
| `tok/sec: 104,355` | 每秒处理的token数量 |
| `bf16_mfu: 0.00` | BF16矩阵运算利用率 |
| `epoch: 1` | 当前训练轮次 |
| `pq: 0` | 前向传递计数 (pass queue) |
| `rg: 52` | 梯度累积步骤 (gradient accumulation steps) |
| `total time: 3.79m` | 训练已用总时间 |
| `eta: 16.8m` | 预计剩余时间 (estimated time of arrival) |

### 4.4 检查点保存

训练完成后会显示最终损失值和验证集结果：

```
Step 00500 | Validation bpb: 1.147239
Step 00500 | CORE metric: 0.0321
2026-06-05 14:09:50,307 - nanochat.checkpoint_manager - INFO - Saved model parameters to: /root/.cache/nanochat/base_checkpoints/d4/model_000500.pt
2026-06-05 14:09:50,308 - nanochat.checkpoint_manager - INFO - Saved metadata to: /root/.cache/nanochat/base_checkpoints/d4/meta_000500.json
2026-06-05 14:09:50,763 - nanochat.checkpoint_manager - INFO - Saved optimizer state to: /root/.cache/nanochat/base_checkpoints/d4/optim_000500_rank0.pt
Peak memory usage: 21601.56MiB
Total training time: 20.87m
Minimum validation bpb: 1.147146
```

### 4.5 检查点文件说明

检查点保存在 `~/.cache/nanochat/base_checkpoints/{model_tag}/`，包含：
- `model_{step:06d}.pt` - 模型参数
- `optim_{step:06d}_rank{rank}.pt` - 优化器状态
- `meta_{step:06d}.json` - 元数据（包含 dataloader 状态、最佳验证损失等）

---

## 5. 继续训练（中期训练）

### 5.1 继续训练命令

如果训练中断或需要继续训练，可以使用 `--resume-from-step` 参数从检查点恢复：

```bash
python -m scripts.base_train --depth=4 --device-batch-size=32 --num-iterations=1000 --resume-from-step=500
```

### 5.2 参数说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `--resume-from-step=500` | 500 | 从第500步继续（加载检查点） |
| `--num-iterations=1000` | 1000 | 总共训练到1000步（即再训练500步） |

### 5.3 执行流程

1. 从 `~/.cache/nanochat/base_checkpoints/default/` 加载第500步的检查点
2. 恢复模型参数、优化器状态、dataloader 位置
3. 继续训练到第1000步

### 5.4 其他可选参数

```bash
# 指定模型标签（如果之前训练时用了自定义标签）
python -m scripts.base_train --depth=4 --device-batch-size=32 --num-iterations=1000 --resume-from-step=500 --model-tag="my_model"

# 指定检查点目录
python -m scripts.base_train --depth=4 --device-batch-size=32 --num-iterations=1000 --resume-from-step=500 --checkpoint-dir=/path/to/checkpoints
```

---

## 6. SFT 微调（监督微调）

### 6.1 SFT 简介

基础模型训练完成后，可以使用 SFT（Supervised Fine-Tuning）对模型进行微调，使其更适合对话任务。

### 6.2 启动命令

```bash
python -m scripts.chat_sft --device-batch-size=16
```

### 6.3 命令输出说明

执行后会显示以下信息：

```
2026-06-05 15:08:44,273 - nanochat.checkpoint_manager - INFO - No model tag provided, guessing model tag: d4
2026-06-05 15:08:44,274 - nanochat.checkpoint_manager - INFO - Loading model from /root/.cache/nanochat/base_checkpoints/d4 with step 1000
2026-06-05 15:08:44,528 - nanochat.checkpoint_manager - INFO - Building model with config: {'sequence_len': 2048, 'vocab_size': 32768, 'n_layer': 4, 'n_head': 2, 'n_kv_head': 2, 'n_embd': 256, 'window_pattern': 'SSSL'}
Inherited max_seq_len=2048 from pretrained checkpoint
NOTE: --device-batch-size=16 overrides pretrained value of 32
Inherited total_batch_size=262144 from pretrained checkpoint
Inherited embedding_lr=0.3 from pretrained checkpoint
Inherited unembedding_lr=0.008 from pretrained checkpoint
Inherited matrix_lr=0.02 from pretrained checkpoint
```

**关键信息解读**：
- `guessing model tag: d4` - 自动检测模型标签（depth=4）
- `Loading model from ... with step 1000` - 从第1000步的检查点加载
- `Inherited ...` - 从预训练检查点继承的超参数

### 6.4 SFT 可选参数

| 参数 | 说明 |
|------|------|
| `--model-tag` | 指定模型标签（默认为 d4） |
| `--model-step` | 指定从哪个检查点加载（默认自动检测最新） |
| `--device-batch-size` | 设备批次大小（可覆盖预训练值） |
| `--load-optimizer` | 是否加载优化器状态（0=不加载，1=加载） |

### 6.5 完整示例

```bash
# 使用指定模型和检查点进行 SFT
python -m scripts.chat_sft --model-tag=d4 --model-step=1000 --device-batch-size=16 --load-optimizer=1
```

### 6.6 SFT 输出示例

```
Step 01938 | Validation bpb: 0.5537
Final: 608/2376 (25.59%)
  ARC-Easy: 25.59%
Final: 356/1172 (30.38%)
  ARC-Challenge: 30.38%
Final: 3913/14042 (27.87%)
  MMLU: 27.87%
Rank 0 | 0/24 (0.00%)
Final: 0/24 (0.00%)
  GSM8K: 0.00%
Rank 0 | 0/24 (0.00%)
Final: 0/24 (0.00%)
  HumanEval: 0.00%
Rank 0 | 22/24 (91.67%)
Final: 22/24 (91.67%)
  SpellingBee: 91.67%
Step 01938 | ChatCORE: 0.1724 | ChatCORE_cat: 0.0392
2026-06-05 17:25:22,533 - nanochat.checkpoint_manager - INFO - Saved model parameters to: /root/.cache/nanochat/chatsft_checkpoints/d4/model_001938.pt
Peak memory usage: 11095.37MiB
Total training time: 86.19m
```

这是个很小的模型，效果有限，但已经具备基本的对话能力了。

---

## 7. 模型服务部署

### 7.1 启动 GPT 服务

```bash
# 开启防火墙（可选）
sudo ufw allow 8000

# 启动服务
python -m scripts.chat_web
```

### 7.2 访问服务

服务启动后，可以通过浏览器访问 `http://localhost:8000` 来与训练好的模型对话。

---

## 8. 常见问题排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'pyarrow'` | 执行 `pip install pyarrow` |
| `Multiple top-level packages discovered` | 在 `pyproject.toml` 添加 `[tool.setuptools] packages = ["nanochat"]` |
| Python 版本回退到 3.10 | 使用 `uv venv --python 3.12` 强制指定版本 |
| `unrecognized arguments: --max_chars` | 使用 `--max-chars=` 格式（带等号） |
| `libcusparseLt.so.0` 或 `libcupti.so.12` 缺失 | 使用 `uv add nvidia-cuda-cupti-cu12` 安装 |

### 8.1 SDPA dtype 不匹配错误

**现象**：训练过程中在 `engine.generate_batch` 阶段崩溃，报错：
```
RuntimeError: Expected query, key, and value to have the same dtype, but got
query.dtype: float key.dtype: c10::BFloat16 and value.dtype: c10::BFloat16 instead.
```
即使设置 `export TORCH_DEFAULT_DTYPE=float32` 也无法解决。

**根本原因**：
- 模型使用混合精度训练，kv_cache 中的 key/value 被保存为 bfloat16
- 部分路径的 query 仍是 float32（单 token 生成 Tq==1 或需要显式掩码的分支）
- `_sdpa_attention` 函数仅在"Full context"分支做了 dtype 转换，其他分支遗漏

**解决方案**：修改 `nanochat/flash_attention.py` 中的 `_sdpa_attention` 函数，在函数开头统一所有输入的 dtype：

```python
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    # 统一 dtype（关键修复）
    if q.dtype != k.dtype:
        q = q.to(k.dtype)
    if v.dtype != k.dtype:
        v = v.to(k.dtype)

    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]
    # 后续所有分支均使用统一后的 q/k/v
    ...
```

---

你已经从零完成了一个大模型的训练。现在可以与他对话，来唤醒这个刚刚问世的大模型了。