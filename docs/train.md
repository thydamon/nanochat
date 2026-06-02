# Nanochat 训练环境部署记录

> 参考 [火山引擎教程](https://developer.volcengine.com/articles/7563928608470007851) 在 Docker 容器中部署 nanochat 训练环境，并解决 CUDA 库缺失、Python 版本冲突、setuptools 打包错误等问题的完整过程。

---

## 1. 环境准备与容器启动

- GPU：NVIDIA GeForce RTX 3090 (24GB 显存)
- 宿主机 OS：Ubuntu 22.04
- 容器启动命令（必须包含 GPU 支持和特权模式）：

```bash
docker run -it --gpus all --privileged your_image /bin/bash
```

> 注意：`--gpus all` 暴露所有 GPU，`--privileged` 提供访问 /dev、/sys 等系统目录的权限。

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

## 5. 修复 setuptools 打包错误

> 解决 flat-layout 问题

在 `pyproject.toml` 末尾添加以下内容，明确指定只打包 nanochat 目录：

```bash
cat >> pyproject.toml << 'EOF'

[tool.setuptools]
packages = ["nanochat"]
EOF
```

> 若不修复，执行 `pip install -e .` 或 `uv sync` 时会报 `Multiple top-level packages discovered` 错误。

## 6. 同步项目依赖

```bash
uv sync
```

执行成功后，会输出类似 `Installed 65 packages in 221ms` 的信息。

## 7. 安装 CUDA 库

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

## 8. 下载数据集

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

## 9. 训练分词器

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

---

## 常见问题排查

| 问题 | 解决方案 |
|------|---------|
| `ModuleNotFoundError: No module named 'pyarrow'` | 执行 `pip install pyarrow` |
| `Multiple top-level packages discovered` | 在 `pyproject.toml` 添加 `[tool.setuptools] packages = ["nanochat"]` |
| Python 版本回退到 3.10 | 使用 `uv venv --python 3.12` 强制指定版本 |
| `unrecognized arguments: --max_chars` | 使用 `--max-chars=` 格式（带等号） |
| `libcusparseLt.so.0` 或 `libcupti.so.12` 缺失 | 使用 `uv add nvidia-cuda-cupti-cu12` 安装 |