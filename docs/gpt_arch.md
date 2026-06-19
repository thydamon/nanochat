# nanochat/gpt.py Transformer 架构详解

这是 nanochat 项目的核心模型文件，完整实现了现代 Transformer 架构，集成了 2023-2024 年的多项训练技巧。

---

## 目录

1. [模型整体结构](#1-模型整体结构)
2. [Token Embedding 层](#2-token-embedding-层)
3. [Block 层：注意力机制](#3-block-层注意力机制)
4. [Block 层：Multi-Head 与 GQA](#4-block-层multi-head-与-gqa)
5. [Block 层：RoPE 旋转位置编码](#5-block-层rope-旋转位置编码)
6. [Block 层：MLP 与 Block 结构](#6-block-层mlp-与-block-结构)
7. [前向传播流程](#7-前向传播流程)
8. [nanochat 特殊设计](#8-nanochat-特殊设计)
9. [GPTConfig 配置](#9-gptconfig-配置)
10. [权重初始化](#10-权重初始化)
11. [与 GPT-2 的对比](#11-与-gpt-2-的对比)

---

## 1. 模型概览

### 1.1 什么是 GPT 模型

#### 1.1.1 GPT 名称解释

**GPT = Generative Pre-trained Transformer**（生成式预训练 Transformer）

| 名称 | 含义 | 说明 |
|------|------|------|
| **Generative** | 生成式 | 给定前文，生成后续内容 |
| **Pre-trained** | 预训练 | 在大规模数据上预先训练好的 |
| **Transformer** | 核心架构 | 基于注意力机制的深度学习网络 |

**一句话概括**：GPT 是一个能根据前文预测下一个词的深度学习模型。

#### 1.1.2 GPT 模型解决什么问题？

**核心任务：语言建模 (Language Modeling)**

```
输入: "今天天气真"
              │
              ▼
模型预测: "好"  ← 最可能
         "坏"
         "冷"
```

**语言建模 = 已知前文，预测下一个词**

| 问题类型 | 示例 | GPT 如何解决 |
|----------|------|-------------|
| **文本生成** | 写文章、写代码 | 自回归生成 |
| **问答系统** | 回答问题 | 根据问题生成答案 |
| **文本补全** | 代码补全、句子补全 | 预测下一个词 |
| **对话系统** | 聊天机器人 | 根据对话历史生成回复 |

**本质理解**：GPT 解决的是 **"给定上下文，预测下一个词"** 这个看似简单的问题。当需要写一篇完整文章时，模型会一个词一个词地生成，每个词的生成都基于前面的所有词，最终形成连贯的文本。

#### 1.1.3 什么是传播？

**传播 = 信息在网络中流动和传递的过程**

在神经网络中，"传播"描述的是数据如何在网络的各层之间流动：

```
        信息传播
           │
           ▼
    ┌──────────────┐
    │    输入层     │  ← 接收原始数据
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   隐藏层 1    │  ← 处理、转换信息
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │   隐藏层 2    │  ← 继续处理
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │    输出层     │  ← 产生最终结果
    └──────────────┘
```

#### 1.1.4 有哪些传播类型？

神经网络中有**两种核心传播**：

| 类型 | 方向 | 作用 | 发生时机 |
|------|------|------|----------|
| **前向传播 (Forward)** | 输入 → 输出 | 计算预测结果 | 训练 & 推理 |
| **反向传播 (Backpropagation)** | 输出 → 输入 | 计算梯度 | 仅训练 |

```
┌─────────────────────────────────────────────────────────┐
│                    前向传播 (Forward)                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入 ──► 隐藏层1 ──► 隐藏层2 ──► 输出                    │
│  x      w1·x+b   w2·h1+b   预测结果                      │
│                                                         │
│  特点：数据从左到右单向流动                               │
│  目的：计算预测结果                                       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                   反向传播 (Backpropagation)              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输出 ──► 隐藏层2 ──► 隐藏层1 ──► 输入                    │
│  Loss   计算梯度   计算梯度   计算梯度                    │
│                                                         │
│  特点：数据从右到左反向流动                               │
│  目的：计算每个参数该"怎么调整"                           │
│  工具：链式法则 (Chain Rule)                             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 1.1.5 为什么使用前向传播？

**前向传播是模型处理信息的必经之路**：

| 场景 | 是否用前向传播 | 是否用反向传播 | 原因 |
|------|---------------|---------------|------|
| **训练时** | ✅ 需要 | ✅ 需要 | 学习如何正确预测 |
| **推理时** | ✅ 需要 | ❌ 不需要 | 模型已训练好，只需预测 |

**训练 vs 推理流程对比**：

```
┌─────────────────────────────────────────────────────────┐
│  训练阶段（学习如何预测）                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Step 1: 前向传播                                        │
│  "今天天气真" ──► GPT ──► 预测: "好"(0.8)                │
│                                                         │
│  Step 2: 计算损失                                        │
│  预测 vs 真实 → Loss = 0.22                              │
│                                                         │
│  Step 3: 反向传播                                        │
│  链式法则计算每个参数的梯度                               │
│                                                         │
│  Step 4: 更新权重                                        │
│  w = w - learning_rate × gradient                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
                           │
                           │ 训练完成后
                           ▼
┌─────────────────────────────────────────────────────────┐
│  推理阶段（使用模型预测）                                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  "今天天气真" ──► [前向传播] ──► "好"                     │
│                      │                                   │
│                      │ 只用前向传播！                     │
│                      │ 不需要反向传播                      │
│                      │ 不需要计算梯度                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**为什么推理时必须用前向传播？**
- 前向传播是**获取预测结果的唯一方式**
- 模型已经训练好了，不需要再学习
- 只需要把输入转成输出即可

#### 1.1.6 组件一览

GPT 模型由以下组件构成：

```
GPT
├── transformer.wte         # Token Embedding (词嵌入)
├── transformer.h [Block×12]  # 12层Transformer块
├── lm_head                   # 预测头 (输出层)
├── resid_lambdas            # 每层残差缩放系数
├── x0_lambdas              # 初始embedding混合系数
├── smear_gate + smear_lambda # Smear层 (类似bigram)
├── backout_lambda          # Backout层 (中间残差减法)
└── value_embeds            # Value Embeddings (ResFormer风格)
```

| 组件 | 通俗理解 | 作用 |
|------|----------|------|
| `wte` | 翻译员 | 把文字转成数字向量 |
| `Block×12` | 12层大脑 | 逐层理解语义 |
| `lm_head` | 决策者 | 把向量转成词表概率 |
| `resid_lambdas` | 思考深度控制器 | 可学习的残差缩放 |
| `smear_gate` | 上下文线索 | 混合前一个token的信息 |
| `backout_lambda` | 纠错员 | 减去中间层残差 |
| `value_embeds` | 知识注入 | ResFormer风格的额外知识 |

### 1.2 前向传播顺序

前向传播是数据从输入到输出的完整流程。下面对应 `nanochat/gpt.py` 行 428-472，详细解释每一步：

#### 步骤 1：Token Embedding（行 428-430）

```python
x = self.transformer.wte(idx)  # 把 token 转成向量
x = norm(x)                      # 归一化
```

**组件：`transformer.wte`**
- **通俗理解**：把文字转成数字向量，像一个"翻译员"
- **工作**：输入 token ID → 输出 n_embd 维向量
- **类比**：查词典，每个字对应一个 768 维的向量表示

#### 步骤 2：Smear 混合（行 436-449）

```python
gate = self.smear_lambda * sigmoid(self.smear_gate(x[:, 1:, :24]))
x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
```

**组件：`smear_gate + smear_lambda`**
- **通俗理解**：混合前一个词的信息，像提供"上下文线索"
- **工作**：每个位置的向量 = 当前词 + gate × 前一个词
- **类比**：读"林"字时，顺便记一下前面是"森"，帮助理解"森林"

#### 步骤 3：保存初始 Embedding（行 452）

```python
x0 = x  # 保存初始 embedding，用于后续残差混合
```

**组件：`x0_lambdas`**
- **通俗理解**：把最初的信息混合回来，像"复习笔记"
- **用途**：在每一层结束后，混合初始 embedding，防止信息丢失

#### 步骤 4：12 层 Transformer Block（行 456-461）

```python
for i, block in enumerate(self.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0  # 残差混合
    ve = self.value_embeds[str(i)](idx).to(x.dtype)          # 知识注入
    x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
```

**组件：`transformer.h (Block×12)` + `resid_lambdas` + `value_embeds`**

| 子组件 | 通俗理解 | 作用 |
|--------|----------|------|
| `resid_lambdas` | 控制每层的"思考深度" | 可学习的残差缩放系数，像音量旋钮 |
| `x0_lambdas` | 把初始信息混合回来 | 防止深层网络丢失原始信息 |
| `value_embeds` | 额外注入的"知识向量" | ResFormer 风格，像专家建议 |
| `Block×12` | 12层"思考单元" | 逐层理解语义，每层包含注意力+MLP |

**Block 内部结构**（每层都执行）：
```
输入 x
  ├── norm(x) 归一化
  ├── attention(norm(x)) → 注意力机制：让词知道其他词在说什么
  ├── x = x + attention_output  残差连接
  ├── norm(x) 归一化
  ├── MLP(norm(x)) → 前馈网络：非线性变换
  └── x = x + MLP_output  残差连接
```

#### 步骤 5：Backout 中间层（行 463-464）

```python
if x_backout is not None:
    x = x - self.backout_lambda.to(x.dtype) * x_backout
```

**组件：`backout_lambda`**
- **通俗理解**：去掉中间层的"低级特征"，像"纠错"
- **工作**：在第 6 层（n_layer//2）保存中间结果，最后减去
- **类比**：写完初稿后，回过头来删除拼写错误和格式问题

#### 步骤 6：预测下一个词（行 467-472）

```python
x = norm(x)
logits = self.lm_head(x)                    # 转成词表概率
logits = softcap * torch.tanh(logits / softcap)  # 裁剪到 [-15, 15]
```

**组件：`lm_head`**
- **通俗理解**：把处理后的向量转成词表概率，像"决策者"
- **工作**：n_embd 维向量 → vocab_size 维 logits
- **输出**：每个词表中词的得分，经过 softcap 裁剪防止数值爆炸

### 1.3 前向传播流程总览

```
输入: "走进森林"
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. wte: "走进森林" → [0.1, 0.3, ...] × 4                    │
│    翻译员：把文字转成数字向量                                 │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. smear: 混合前一个 token 的信息                            │
│    "林" = "林" + 0.3 × "森"                                  │
│    上下文线索：让每个词知道前一个词                           │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 3-5. Block×12: 12 轮深度思考                                │
│    ┌────────────────────────────────────────────────────┐   │
│    │ 第1层：残差混合 + 注意力 + MLP                      │   │
│    │ 第2层：残差混合 + 注意力 + MLP                      │   │
│    │ ...                                                │   │
│    │ 第6层：同时保存 x_backout（用于 backout）           │   │
│    │ ...                                                │   │
│    │ 第12层：最终表示                                    │   │
│    └────────────────────────────────────────────────────┘   │
│    12个大脑区域：逐层理解语义                                 │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. backout: 减去中间层残差，去掉低级特征                      │
│    纠错：删除拼写、格式等低级错误                             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. lm_head: 转成词表概率                                    │
│    决策者：输出 → "发现"(0.7), "遇到"(0.2), "看到"(0.1)     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
输出: 下一个词的概率分布
```

### 1.4 参数计算（默认配置约 85M~95M）

以 `n_embd=768, n_layer=12, n_head=6, vocab_size=32768` 为例：

| 组件 | 计算公式 | 参数量 |
|------|----------|--------|
| Token Embedding (wte) | vocab_size × n_embd | ≈ 25.2M |
| LM Head | n_embd × vocab_size | ≈ 25.2M |
| 每层 Attention | 3 × n_embd × (n_head + 2×n_kv_head) × head_dim | ≈ 1.8M |
| 每层 MLP | 2 × 4×n_embd² | ≈ 4.8M |
| 12层 Block 合计 | 12 × 7.4M | ≈ 88.8M |
| **总计** | 25.2 + 25.2 + 88.8 | **≈ 139M** |

---

## 2. Token Embedding 层

Token Embedding 层负责把文字转换成数字向量，是模型处理文本的入口和出口。本节按数据流向组织：

```
Token ID ──wte──► [batch, seq, n_embd] ──Block×12──► [batch, seq, n_embd] ──lm_head──► [batch, seq, vocab_size]
```

### 2.1 Embedding 维度（n_embd）

**n_embd 是 Token Embedding 的核心配置参数**，它决定了：
- wte 的输出向量维度
- lm_head 的输入向量维度
- 每层 Block 的隐藏层维度
- 所有注意力计算的向量长度

```python
# 每个 token 被表示为一个 n_embd 维的向量
n_embd = 768  # "学习" → [0.1, 0.5, ..., 0.2]  # 768 个数字
```

| n_embd | 模型容量 | 表达能力 |
|--------|----------|----------|
| 256 | 小 | 较弱 |
| 768 | 中 | 中等（nanochat 默认） |
| 4096 | 大 | 强（如 LLaMA-2 7B） |

### 2.2 wte：输入 Embedding

**作用**：把 token ID（文字的数字编号）转成 n_embd 维向量

**什么是 Token ID？**

Token ID = 文字的数字编号。计算机只能处理数字，需要把文字转换成数字：

```
"走进森林"
      │
      ▼ 分词（Tokenizer）
["走", "进", "森林"]
      │
      ▼ 查表（Vocab）
[1024, 3345, 876]  ← 每个词对应一个数字编号
```

**为什么需要 Token ID？**

| 问题 | 答案 |
|------|------|
| "走进森林"计算机认识吗？ | ❌ 不认识，计算机只认数字 |
| Token ID 从哪来？ | 分词器（Tokenizer）查词典得到 |
| Token 是什么？ | 文本的最小单位（字、词、或子词） |

**Token 的粒度**：

| 粒度 | 例子 | 特点 |
|------|------|------|
| **字级别** | "走", "进", "森", "林" | 简单，但词会被拆散 |
| **词级别** | "走进", "森林" | 语义完整，但词表大 |
| **子词级别** | "森林", "林" | 平衡效率与语义（常用） |

**一句话**：Token ID 就是文字的"数字身份证"，每个词在词表中都有一个唯一编号。

```python
x = transformer.wte(idx)  # [batch, seq_len] → [batch, seq_len, n_embd]
```

**输入**：token ID（整数）
**输出**：n_embd 维向量

```
"走进森林"
    │
    ▼ tokenizer
[1024, 3345, 876]  ← token ID 列表
    │
    ▼ wte
[[0.1, 0.3, ..., 0.2],   ← 每个 ID 对应一个 n_embd 维向量
 [0.2, 0.1, ..., 0.4],
 [0.5, 0.2, ..., 0.1]]
```

### 2.2.1 中间处理：Block×12

wte 输出的向量需要经过 **12 层 Transformer Block** 的深度处理，才能真正理解语义。

**什么是 Block？**

Block 是 Transformer 的核心处理单元，每层 Block 包含：
- **注意力机制 (Attention)**：让每个词知道其他词在说什么
- **前馈网络 (MLP)**：非线性变换，提取更深层的特征

```
wte 输出: [[0.1, ...], [0.2, ...], [0.5, ...]]  ← 只是简单的字向量
                │           │           │
                ▼           ▼           ▼
         ┌─────────────────────────────────┐
         │     Block×12：12层深度理解        │
         │  - 第1层：理解"森林"是自然环境     │
         │  - 第2层：理解"走进"是动作         │
         │  - ...                           │
         │  - 第12层：理解完整语义            │
         └─────────────────────────────────┘
                │           │           │
                ▼           ▼           ▼
Block 输出: [[0.8, ...], [0.3, ...], [0.9, ...]]  ← 真正的语义向量
```

**Block 能解决什么问题？**

| 问题 | 例子 | Block 如何解决 |
|------|------|---------------|
| 指代消解 | "他喜欢苹果" → 谁是"他"？"苹果"是什么？ | 注意力让"他"看向前面的词 |
| 一词多义 | "苹果很苹果" | 注意力根据上下文判断含义 |
| 长距离依赖 | "住在[北京]的人喜欢[北京]的胡同" | 跨越距离建立联系 |

**Block 的结构**（每层都执行）：

```
输入 x
  ├── norm(x) 归一化
  ├── attention(norm(x)) → 注意力：让词知道其他词在说什么
  ├── x = x + attention_output  残差连接
  ├── norm(x) 归一化
  ├── MLP(norm(x)) → 前馈网络：非线性变换
  └── x = x + MLP_output  残差连接
```

**一句话**：Block 是模型的"大脑"，12 层 Block 让模型能够逐层深入理解文本的语义。

### 2.3 lm_head：输出层

**作用**：把 Block 处理后的语义向量转换成**每个词的可能性得分**，用于预测下一个词

```python
logits = lm_head(norm(x))  # [batch, seq_len, n_embd] → [batch, seq_len, vocab_size]
```

**输入**：n_embd 维向量（Block 处理后的结果）
**输出**：vocab_size 维 logits（每个词的原始得分）

#### 为什么需要这个转换？

语言模型的任务是：**给定前文，预测下一个词**。

```
"小明走进__"
    │
    ├── 需要预测的候选词：森林、房间、学校、...
    └── 哪个词最合理？需要给每个词打分
```

lm_head 就是那个"打分器"：
- 输入：模型对"小明走进"的理解（n_embd 向量）
- 输出：每个词表中词的得分（vocab_size 维向量）

```
[0.1, 0.3, ..., 0.2]  ← 模型对上下文的理解（n_embd=768）
    │
    ▼ lm_head（一个线性变换：y = Wx + b）
[3.2, 1.5, -2.1, ..., 0.8]  ← 每个词的原始得分（vocab_size=32768）
     ↑     ↑        ↑
   "森林" "房间"   "学校"
   得分最高      得分较低
```

#### 什么是 logits？

**logits = 模型的"原始打分"**

lm_head 输出的不是概率，而是一串原始分数（logits）：

```
lm_head 输出: [3.2, 1.5, -2.1, ..., 0.8]
                ↑    ↑     ↑        ↑
              得分  得分  得分     得分
               │     │     │        │
              森林  房间  学校    苹果
              
这些就是 logits：模型对每个词的"喜好程度"
```

**为什么叫 logits？**

| 特点 | 说明 |
|------|------|
| **可以是任意实数** | 3.2、1.5、-2.1、999、-100... 都可以 |
| **没有限制范围** | 不是 [0,1]，也不要求和为 1 |
| **相对大小有意义** | 3.2 > 1.5，所以"森林"比"房间"更可能 |

**logits 不是概率，但可以通过 softmax 转成概率**：

```
logits:  [3.2, 1.5, -2.1]
             │     │     │
             ▼     ▼     ▼
         softmax
             │     │     │
             ▼     ▼     ▼
概率:    [0.85, 0.14, 0.01]  ← 每个词被选中的可能性（0~1之间，和为1）
            ↑     ↑     ↑
          森林   房间   学校
          最可能        最不可能
```

**类比**：logits 像考试分数，概率像排名百分比

```
考试得分 (logits):
  数学: 95分
  语文: 88分
  英语: 72分
  
排名百分比 (概率):
  数学: 95%  ← 最可能选数学
  语文: 4%
  英语: 1%
```

#### logits 和概率的区别

| 概念 | 说明 | 例子 |
|------|------|------|
| **logits** | 原始得分，未归一化，可以是任意实数 | [3.2, 1.5, -2.1] |
| **概率** | 经过 softmax 归一化后的值，范围 [0,1]，和为 1 | [0.85, 0.14, 0.01] |

```
logits: [3.2, 1.5, -2.1]
          │
          ▼ softmax
概率:   [0.85, 0.14, 0.01]  ← 所有概率之和 = 1
           ↑     ↑     ↑
         "森林" "房间" "学校"
```

**为什么用 logits 而不是直接输出概率？**
1. **数值稳定性**：softmax 对大数敏感，logits 更适合数值计算
2. **训练友好**：交叉熵损失可以直接用 logits 计算
3. **模型表达**：logits 的相对大小更重要，绝对值不直接代表概率

#### 后续处理

lm_head 输出的 logits 还会经过两个步骤：

```python
# 1. Softcap 裁剪：防止数值爆炸
softcap = 15
logits = softcap * torch.tanh(logits / softcap)  # 裁剪到 [-15, 15]

# 2. Softmax + 交叉熵：在损失计算时进行
loss = F.cross_entropy(logits, targets)  # 自动 softmax + 计算损失
```

### 2.4 输入与输出的对应关系

```
         ┌─────────────────────────────────────────────────────────┐
         │                      模型前向传播                         │
         └─────────────────────────────────────────────────────────┘

Token ID ──wte──► n_embd向量 ──Block×12──► n_embd向量 ──lm_head──► 词表概率
              ▲                           │                           │
              │                           │                           │
         输入维度                          │                           │
         由token数                          │                           │
         决定                               │                           │
                                            │                           │
                               n_embd是统一的隐藏层维度                  │
                               wte的输出 = lm_head的输入                │
```

**nanochat 的设计**：

| 对比项 | GPT-2 | nanochat |
|--------|-------|----------|
| wte 和 lm_head | 共享权重（tied weights） | 独立初始化（untied weights） |
| 优势 | 节省参数量 | 各自可以更专业地学习 |

```python
# nanochat 的独立初始化
transformer.wte   = nn.Embedding(vocab_size, n_embd)  # 词→向量
lm_head           = nn.Linear(n_embd, vocab_size, bias=False)  # 向量→词
```

---

## 3. Block 层：注意力机制

### 3.1 注意力在 Block 中的角色

每层 Block 包含两个核心组件：

```
Block
├── CausalSelfAttention  ← 注意力机制（核心）
└── MLP                 ← 前馈网络
```

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # 注意力
        x = x + self.mlp(norm(x))                                        # MLP
        return x
```

**注意力机制的作用**：
1. **计算词与词之间的关系**：让每个词知道其他词在说什么
2. **信息传递**：解决"他"指代谁、"它"是什么等问题
3. **上下文聚合**：把分散的相关信息聚集到每个词上

### 3.2 Q/K/V 概念

```
Query（查询）: "我想找关于机器学习的书"
Key（键）:     每本书的分类标签
Value（值）:   书的实际内容

注意力机制：用 Query 去查询所有 Key，找到最相关的书，然后读取 Value。
```

### 3.3 点积（Dot Product）

点积是向量之间的一种运算：`a·b = a1*b1 + a2*b2 + ... + an*bn`

```
点积 > 0：向量方向相近（相似）
点积 = 0：向量垂直（无关）
点积 < 0：向量方向相反（相反）
```

**为什么用点积衡量相似度？**
- 计算简单，GPU 高效并行
- 几何直觉：点积越大越"相似"
- 可学习：通过训练自动调整

### 3.4 注意力公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**四步拆解**：

```python
# 第一步：计算相似度
scores = Q @ K.T  # 每个词和其他所有词的关联程度

# 第二步：缩放（防止 softmax 饱和）
scores = scores / √d_k

# 第三步：Softmax 归一化
weights = softmax(scores, dim=-1)  # softmax(x_i) = exp(x_i) / Σ exp(x_j)

# 第四步：加权求和
output = weights @ V  # 每个词的表示 = 所有词的 Value 加权和
```

### 3.5 自注意力（Self-Attention）

自注意力让序列自己和自己比较：

```
输入: "我想学习机器学习"
        ↓
每个词都同时是 Query、Key、Value
        ↓
"学习" 作为 Query，去问 "机器"、"学习"、"我"、"想"
        ↓
找到最相关的词，加权求和
        ↓
输出: "学习"的新表示（融合了上下文）
```

**自注意力能解决什么问题？**

1. **长距离依赖**：直接计算"他"和"小明"的关联，解决指代问题
2. **上下文理解**：让"苹果"参考周围词确定具体含义
3. **并行计算**：所有位置可以同时计算
4. **信息聚合**：把分散的相关信息聚集到一起

### 3.6 注意力机制 vs 自注意力机制

| 对比项 | 注意力机制 | 自注意力机制 |
|--------|-----------|-------------|
| Q/K/V 来源 | 可以不同（跨语言、跨模态） | 必须相同（同一序列） |
| 应用场景 | 机器翻译、图像描述 | 语言模型内部、Transformer |
| 经典例子 | Seq2Seq 的 encoder-decoder attention | Transformer 的 multi-head self-attention |

```
注意力机制（Attention）
    ├── 跨序列注意力：Query 来自序列 A，Key/Value 来自序列 B
    └── 自注意力（Self-Attention）：Query、Key、Value 都来自序列 A
```

### 3.7 因果注意力（Causal Attention）

语言模型只能看前面的词，不能偷看后面的答案：

```
输入: "今天天气真好"
位置:  0   1   2   3   4

"天" (位置1) 只能看 "今" (位置0)     ✓
"气" (位置2) 只能看 "今、天"        ✓
"好" (位置4) 只能看 "今、天、气、"   ✓
              不能看位置5,6,7...       ✗
```

---

## 4. Block 层：Multi-Head 与 GQA

### 4.1 Multi-Head（多头注意力）

把 embedding 维度分成多个头：

```python
n_embd = 768      # embedding 总维度
n_head = 6        # 头的数量
head_dim = 128    # 每个头的维度 = n_embd / n_head = 768 / 6 = 128
```

**为什么需要多头？**

一个注意力头只能学到一种类型的模式。多头让不同头学习不同关系：

```
"学习" 被 6 个头处理：
Head 1: 学习语法关系
Head 2: 学习语义关系
Head 3: 学习位置关系
...
所有头的结果拼接，得到更丰富的表示
```

**头数选择**：n_embd 必须能被 n_head 整除

```python
n_embd = 768
768 / 6 = 128   ✓  # 常用选择
768 / 5 = 153.6 ✗  # 不是整数
```

**为什么 head_dim 通常是 128？**
- 足够大，能捕捉足够的模式
- 足够小，适合 GPU 并行计算
- 128 是 2 的幂次，GPU 计算效率高

### 4.2 MHA（Multi-Head Attention）

标准 MHA 中，每个 Q/K/V 头数相同：

```
n_head = 6 时：6 个 Query 头 + 6 个 Key 头 + 6 个 Value 头
推理时需要缓存 6+6 = 12 个向量序列
```

### 4.3 GQA（Group-Query Attention）

当 `n_kv_head < n_head` 时，多个 Q 头共享同一个 K/V 头：

```
GQA (n_head=6, n_kv_head=2):
  6个Q头（各自独立）
  2个K头（每3个Q头共享1个）
  2个V头（每3个Q头共享1个）
  推理时只需缓存 2+2 = 4 个向量序列
```

**显存节省 3 倍！**

### 4.4 如何启用

```python
# MHA（标准）
config = GPTConfig(n_head=12, n_kv_head=12, n_embd=768)

# GQA（分组查询注意力）
config = GPTConfig(n_head=12, n_kv_head=4, n_embd=768)
```

### 4.5 应用场景

| 模型 | n_head | n_kv_head | 类型 |
|------|--------|-----------|------|
| LLaMA-2 7B | 32 | 8 | GQA |
| Mistral 7B | 8 | 1 | GQA |
| Qwen 7B | 32 | 32 | MHA |
| nanochat (默认) | 6 | 6 | MHA |

---

## 5. Block 层：RoPE 旋转位置编码

### 5.1 为什么需要 RoPE？

注意力机制本身不包含位置信息。传统方法（绝对位置）只能学到"位置1"，学不到"在位置2旁边"。

### 5.2 RoPE 核心思想

不是把位置信息加到向量上，而是旋转向量。

```
对于维度 (x1, x2)，旋转角度 θ：

不同位置使用不同的旋转角度：
  位置0: θ=0°      → [x1, x2]
  位置1: θ=θ        → 旋转θ
  位置2: θ=2θ       → 旋转2θ
```

旋转后的点积只依赖于 (i-j)，即**相对位置**！模型能学到"前面2个位置"vs"前面1个位置"。

### 5.3 代码实现

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * (-sin)
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], 3)
```

### 5.4 RoPE 的优势

| 对比项 | 绝对位置编码 | RoPE |
|--------|-------------|------|
| 表达能力 | 只知道"在哪" | 同时知道"在哪"和"相对距离" |
| 外推能力 | 超出训练长度效果差 | 能处理超长序列 |

---

## 6. Block 层：MLP 与 Block 结构

### 6.1 MLP（ReLU² 激活）

使用 `(ReLU(x))²` 而非标准的 GELU 或 SwiGLU：

```python
class MLP(nn.Module):
    def forward(self, x):
        x = self.c_fc(x)           # n_embd → 4*n_embd
        x = F.relu(x).square()      # ReLU² 激活函数
        x = self.c_proj(x)         # 4*n_embd → n_embd
        return x
```

### 6.2 Block 结构

标准的 Pre-Norm 结构，使用 **RMSNorm**（无可学习参数）：

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # 注意力残差
        x = x + self.mlp(norm(x))                                        # MLP残差
        return x
```

### 6.3注意力 + MLP 的配合

```
输入 x → 注意力层（聚合上下文）→ MLP层（非线性变换）→ 输出
```

---

## 7. 前向传播流程

```python
def forward(self, idx, targets=None, kv_cache=None):
    # ① Token Embedding
    x = self.transformer.wte(idx)
    x = norm(x)

    # ② Smear：混合前一个token的embedding
    gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
    x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]])

    # ③ 通过12层Block
    x0 = x  # 保存初始embedding
    for i, block in enumerate(self.transformer.h):
        x = resid_lambdas[i] * x + x0_lambdas[i] * x0  # 可学习的残差混合
        ve = value_embeds[i](idx)
        x = block(x, ve, cos_sin, window_size, kv_cache)

    # ④ Backout：减去中间层残差
    x = x - backout_lambda * x_backout

    # ⑤ LM Head
    logits = lm_head(norm(x))

    # ⑥ Softcap + 交叉熵损失
    logits = softcap * tanh(logits / softcap)
    loss = cross_entropy(logits, targets)
```

---

## 8. nanochat 特殊设计

### 8.1 Smear 层

混合前一个 token 的 embedding（类似 bigram 信号）：

```python
gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]])
```

### 8.2 Backout 层

减去中间层残差，去除低级特征（拼写、格式）：

```python
x = x - backout_lambda * x_backout
```

### 8.3 Value Embeddings

ResFormer 风格：可学习的 value 表注入注意力：

```python
ve = value_embeds[i](idx)
```

### 8.4 可学习残差混合

```python
x = resid_lambdas[i] * x + x0_lambdas[i] * x0  # 可学习的残差混合
```

### 8.5 设计亮点汇总

| 特性 | 说明 |
|------|------|
| **GQA** | 减少 KV cache，推理更快 |
| **Sliding Window** | 浅层用滑动窗口，深层用全上下文 |
| **Value Embeddings** | 可学习的 value 表注入注意力 |
| **Smear** | 混合前 token embedding，bigram 信号 |
| **Backout** | 减中间层残差，去除低级特征 |
| **Pre-Norm + RMSNorm** | 无偏置，省显存，训练稳定 |
| **ReLU²** | MLP 激活函数 |
| **QK Norm + 1.2 缩放** | 让注意力更 sharp |
| **Untied Weights** | wte 和 lm_head 独立初始化 |
| **Softcap** | logits 裁剪到 [-15, 15]，防止数值爆炸 |

---

## 9. GPTConfig 配置

### 9.1 配置代码

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # 上下文窗口长度
    vocab_size: int = 32768       # 词表大小
    n_layer: int = 12             # Transformer 层数
    n_head: int = 6               # Query 头数
    n_kv_head: int = 6            # Key/Value 头数 (GQA)
    n_embd: int = 768             # 隐藏层维度
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

### 9.2 参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sequence_len` | 2048 | 上下文窗口长度。超过会报错，同时用于预计算 RoPE 表。 |
| `vocab_size` | 32768 | 词表大小，决定 `lm_head` 输出维度。 |
| `n_layer` | 12 | Transformer Block 层数，越多越深。 |
| `n_head` | 6 | Query 头数，`n_head * head_dim = n_embd`。 |
| `n_kv_head` | 6 | Key/Value 头数，支持 GQA。当 `< n_head` 时多个 Q 共享同一 KV。 |
| `n_embd` | 768 | 隐向量维度，决定模型宽度。 |
| `window_pattern` | `"SSSL"` | 滑动窗口模式：`S`=短窗口，`L`=全上下文。最后层强制为 `L`。 |

### 9.3 常见配置

| 模型 | n_embd | n_head | head_dim |
|------|--------|--------|----------|
| nanochat 小模型 | 256 | 2 | 128 |
| nanochat 默认 | 768 | 6 | 128 |
| LLaMA-2 7B | 4096 | 32 | 128 |
| GPT-3 | 12288 | 96 | 128 |

---

## 10. 权重初始化

```python
# Embedding: 高方差初始化
wte:    normal(std=0.8)
lm_head: normal(std=0.001)  # 很小，避免初期过度自信

# 注意力/MLP层: 均匀分布
attn.c_q/k/v:  uniform(std=1/√n_embd)
attn.c_proj:   zeros
mlp.c_fc:      uniform(0.4×std)
mlp.c_proj:    zeros

# 可学习标量: 退火初始化
resid_lambdas: 从1.15递减到1.05
x0_lambdas:    从0.20递减到0.05
```

---

## 11. 与 GPT-2 的对比

| 对比项 | 标准 GPT-2 | nanochat |
|--------|-----------|---------|
| 位置编码 | 学习式绝对位置 | RoPE 旋转编码 |
| 注意力 | MHA | **GQA** |
| MLP激活 | GELU | **ReLU²** |
| Norm | LayerNorm | **RMSNorm** (无参数) |
| 注意力头 Norm | 无 | **QK Norm** |
| 位置感知 | 仅注意力 | **Smear + RoPE** |
| Token/Head | 共享权重 | **独立权重** |
| 训练技巧 | 无 | Backout、Value Embeddings |

---

## 下一步学习

这是从零重写的现代 Transformer 实现，集成了 2023-2024 年的多项训练技巧。接下来可以看：

- `nanochat/engine.py` — 理解 KV Cache 推理机制
- `nanochat/common.py` — 工具函数