# nanochat 学习计划

基于项目结构，推荐以下学习路径：

## 学习路线

### 1. 入门：先跑通整个流程
```
scripts/base_train.py    → 预训练
scripts/chat_sft.py      → SFT微调
scripts/chat_web.py      → Web对话
```
先运行 `runs/speedrun.sh`，感受完整的训练-对话流程。
https://console.compshare.cn/light-gpu/console/resources 

### 2. 核心：理解模型架构
```
nanochat/gpt.py          → Transformer 实现 (27KB，核心)
nanochat/engine.py       → KV Cache + 推理
nanochat/common.py       → 工具函数
```
**推荐指数**: ⭐⭐⭐⭐⭐

GPT 模型是整个项目的心脏，掌握这里就理解了一半。

### 3. 训练系统
```
scripts/base_train.py    → 预训练入口
nanochat/optim.py        → 优化器 (AdamW + Muon)
nanochat/dataloader.py   → 数据加载
```
**推荐指数**: ⭐⭐⭐⭐

### 4. 分词器
```
nanochat/tokenizer.py    → BPE 分词器
scripts/tok_train.py     → 分词器训练
```
**推荐指数**: ⭐⭐⭐

### 5. 进阶：RLHF & 工具调用
```
scripts/chat_rl.py       → 强化学习
nanochat/execution.py    → Python 代码执行
```
**推荐指数**: ⭐⭐⭐

### 6. 评估任务
```
tasks/common.py          → 任务框架
tasks/gsm8k.py / mmlu.py → 具体任务实现
scripts/chat_eval.py     → 评估脚本
```
**推荐指数**: ⭐⭐

---

## 学习目标对照表

| 目标 | 起点 | 终点 |
|------|------|------|
| 理解 LLM 训练 | `gpt.py` | `base_train.py` |
| 复现 ChatGPT | `speedrun.sh` | `chat_web.py` |
| 改进训练速度 | `gpt.py` + `fp8.py` | `optim.py` |
| 添加新任务 | `tasks/common.py` | `tasks/gsm8k.py` |

---

## 模块依赖关系图

```
dataset.py / tokenizer.py
        ↓
dataloader.py → gpt.py (核心模型)
        ↓              ↓
base_train.py → optim.py (优化器)
        ↓              ↓
   checkpoint_manager.py
        ↓
chat_sft.py (SFT微调)
        ↓
chat_rl.py (RLHF) → execution.py (工具调用)
        ↓
chat_eval.py / chat_web.py (评估/对话)
```

---

## 建议学习顺序

1. **第一阶段** - 跑通流程 (1-2天)
   - 运行 `runs/speedrun.sh` 观察输出
   - 使用 `chat_web.py` 与训练好的模型对话

2. **第二阶段** - 理解核心 (1周)
   - 精读 `gpt.py` 的每一行代码
   - 阅读 `engine.py` 理解推理流程
   - 理解 `common.py` 中的工具函数

3. **第三阶段** - 掌握训练 (3-5天)
   - 理解 `base_train.py` 的训练循环
   - 学习 `optim.py` 中的优化器实现
   - 理解数据加载 `dataloader.py`

4. **第四阶段** - 扩展能力 (1周+)
   - 学习 SFT 微调 `chat_sft.py`
   - 理解 RLHF `chat_rl.py`
   - 添加新任务到 `tasks/`