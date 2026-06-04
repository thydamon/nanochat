# Nanochat 训练环境部署记录

> 参考 [火山引擎教程](https://developer.volcengine.com/articles/7563928608470007851) 在 Docker 容器中部署 nanochat 训练环境，并解决 CUDA 库缺失、Python 版本冲突、setuptools 打包错误等问题的完整过程。

---

## 1. 环境准备与容器启动

- GPU：NVIDIA GeForce RTX 3090 (24GB 显存)
- 宿主机 OS：Ubuntu 22.04
- 容器启动命令（必须包含 GPU 支持和特权模式）：

## 2. 安装基础工具与依赖

```bash
# 安装 git 和 pciutils（可选，用于查看 GPU）
apt update && apt install -y git pciutils

# 安装 pyarrow（数据集处理依赖）
pip install pyarrow
```

> 若不安装 pyarrow，运行 `python -m nanochat.dataset` 时会报 `ModuleNotFoundError`。

## 3. 安装 uv 并克隆项目

```bash
# 安装 uv 包管理器
pip install uv

# 克隆 nanochat 仓库
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/karpathy/nanochat.git
cd nanochat
```

## 3.1 安装额外依赖

```bash
pip install rustbpe tiktoken wandb fastapi uvicorn
```

## 4. 创建 Python 3.12 虚拟环境

> 避免回退到系统默认的 Python 3.10

```bash
# 删除可能残留的旧虚拟环境
rm -rf .venv

# 强制使用 Python 3.12 创建虚拟环境
uv venv --python 3.12

# 激活虚拟环境
source .venv/bin/activate
```

> **踩坑点**：直接执行 `uv sync` 可能会因为系统默认 Python 3.10 而重建虚拟环境，导致环境版本回退。务必先用 `uv venv --python 3.12` 锁定版本。

## 5. 同步项目依赖

```bash
uv sync --python 3.12
```

执行成功后，会输出类似 `Installed 65 packages in 221ms` 的信息。

## 6. 安装 CUDA 库

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

## 7. 下载数据集

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

## 8. 训练分词器

```bash
python -m scripts.tok_train --max-chars=2000000000
```

> 注意：参数使用 `--max-chars=` 格式（带等号），不是 `--max_chars=`

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

## 9. 评估tokenizer
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
===============================================================================================
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
===============================================================================================
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


## 10. 训练模型

### 启动训练命令

```bash
python -m scripts.base_train --depth=4 --device-batch-size=32 --num-iterations=500
```

参数说明：
- `--depth=4`：模型的层数（深度）设置为4
- `--device-batch-size=32`：每个设备（如一张GPU卡）一次处理的数据批次大小为32
- `--num-iterations=500`：总共训练500个步数

### 训练日志详解

训练过程中会输出类似如下的日志：

```
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    
Autodetected device type: cuda
2026-06-03 15:07:42,870 - nanochat.common - INFO - Distributed world size: 1
2026-06-03 15:07:42,871 - nanochat.common - WARNING - Peak flops undefined for: Tesla V100S-PCIE-32GB, MFU will show as 0%
GPU: Tesla V100S-PCIE-32GB | Peak FLOPS (BF16): inf
COMPUTE_DTYPE: torch.float32 (auto-detected: CUDA SM 70 (pre-Ampere, bf16 not supported, using fp32))
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback
WARNING: Training will be less efficient without FA3
WARNING: SDPA has no support for sliding window attention (window_pattern='SSSL'). Your GPU utilization will be terrible.
WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
Estimated FLOPs per token: 8.021635e+07
Auto-computed optimal batch size: 262,144 tokens
Scaling LRs by 0.7071 for batch size 262,144 (reference: 524,288)
Scaling weight decay from 0.280000 to 1.889903 for depth 4
Scaling the LR for the AdamW parameters ∝1/√(256/768) = 1.732051
Using user-provided number of iterations: 500
Total number of training tokens: 131,072,000
Tokens : Scaling params ratio: 11.36
Total training FLOPs estimate: 1.051412e+16
Tokens / micro-batch / rank: 32 x 2048 = 65,536
Tokens / micro-batch: 65,536
Total batch size 262,144 => gradient accumulation steps: 4
Step 00000 | Validation bpb: 3.169725
Step 00000 | Validation bpb: 3.169725
step 00000/00500 (0.00%) | loss: 10.397340 | lrm: 0.03 | dt: 26087.22ms | tok/sec: 10,048 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00001/00500 (0.20%) | loss: 10.392339 | lrm: 0.05 | dt: 2485.81ms | tok/sec: 105,456 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.00m
step 00002/00500 (0.40%) | loss: 10.383359 | lrm: 0.07 | dt: 2496.59ms | tok/sec: 105,000 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.00m
step 00003/00500 (0.60%) | loss: 10.370199 | lrm: 0.10 | dt: 2489.50ms | tok/sec: 105,299 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 2 | total time: 0.00m
step 00004/00500 (0.80%) | loss: 10.349018 | lrm: 0.12 | dt: 2501.42ms | tok/sec: 104,797 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 3 | total time: 0.00m
step 00005/00500 (1.00%) | loss: 10.318948 | lrm: 0.15 | dt: 2489.56ms | tok/sec: 105,297 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 3 | total time: 0.00m
step 00006/00500 (1.20%) | loss: 10.273894 | lrm: 0.17 | dt: 2492.91ms | tok/sec: 105,155 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 4 | total time: 0.00m
step 00007/00500 (1.40%) | loss: 10.206611 | lrm: 0.20 | dt: 2498.73ms | tok/sec: 104,911 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 4 | total time: 0.00m
step 00008/00500 (1.60%) | loss: 10.110173 | lrm: 0.23 | dt: 2504.07ms | tok/sec: 104,687 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 0.00m
step 00009/00500 (1.80%) | loss: 9.975765 | lrm: 0.25 | dt: 2506.58ms | tok/sec: 104,582 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 5 | total time: 0.00m
step 00010/00500 (2.00%) | loss: 9.798287 | lrm: 0.28 | dt: 2506.94ms | tok/sec: 104,567 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 6 | total time: 0.00m
step 00011/00500 (2.20%) | loss: 9.578386 | lrm: 0.30 | dt: 2490.23ms | tok/sec: 105,269 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 6 | total time: 0.04m | eta: 20.3m
step 00012/00500 (2.40%) | loss: 9.359259 | lrm: 0.33 | dt: 2494.11ms | tok/sec: 105,105 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 7 | total time: 0.08m | eta: 20.3m
step 00013/00500 (2.60%) | loss: 9.129915 | lrm: 0.35 | dt: 2505.90ms | tok/sec: 104,610 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 7 | total time: 0.12m | eta: 20.3m
step 00014/00500 (2.80%) | loss: 8.909904 | lrm: 0.38 | dt: 2512.35ms | tok/sec: 104,342 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 8 | total time: 0.17m | eta: 20.3m
...
step 00498/00500 (99.60%) | loss: 3.758718 | lrm: 0.06 | dt: 2531.15ms | tok/sec: 103,566 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 77 | total time: 20.59m | eta: 0.1m
step 00499/00500 (99.80%) | loss: 3.751152 | lrm: 0.05 | dt: 2533.14ms | tok/sec: 103,485 | bf16_mfu: 0.00 | epoch: 1 pq: 2 rg: 77 | total time: 20.63m | eta: 0.0m
Step 00500 | Validation bpb: 1.147239
Downloading https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip...

step 00100/00500 (20.00%) | loss: 5.178412 | lrm: 1.00 | dt: 2512.04ms | tok/sec: 104,355 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 52 | total time: 3.79m | eta: 16.8m

```

日志各字段含义：

| 字段 | 含义 |
|------|------|
| `step 00100/00500` | 当前步数/总步数 |
| `(20.00%)` | 完成百分比 |
| `loss: 5.178412` | 当前步的损失值 |
| `lrm: 1.00` | 学习率乘数 (learning rate multiplier) |
| `dt: 2512.04ms` | 该步耗时（毫秒） |
| `tok/sec: 104,355` | 每秒处理的token数量 |
| `bf16_mfu: 0.00` | BF16矩阵运算利用率 (BF16 MFU) |
| `epoch: 1` | 当前训练轮次 |
| `pq: 0` | 前向传递计数 (pass queue) |
| `rg: 52` | 梯度累积步骤 (gradient accumulation steps) |
| `total time: 3.79m` | 训练已用总时间 |
| `eta: 16.8m` | 预计剩余时间 (estimated time of arrival) |

训练开始时会显示模型配置信息：

```
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

训练完成后会显示最终损失值和验证集结果。

## 常见问题排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'pyarrow'` | 执行 `pip install pyarrow` |
| `Multiple top-level packages discovered` | 在 `pyproject.toml` 添加 `[tool.setuptools] packages = ["nanochat"]` |
| Python 版本回退到 3.10 | 使用 `uv venv --python 3.12` 强制指定版本 |
| `unrecognized arguments: --max_chars` | 使用 `--max-chars=` 格式（带等号） |
| `libcusparseLt.so.0` 或 `libcupti.so.12` 缺失 | 使用 `uv add nvidia-cuda-cupti-cu12` 安装 |
