# nanochat/gpt.py Transformer 架构详解（小白版）

> 目标读者：刚接触大语言模型、能看懂一点 Python 和 PyTorch，但经常被论文里的术语砸晕的同学。
>
> 阅读建议：从头到尾按顺序读。每个新概念出现前，都会先解释“为什么需要它”。

本文档讲解 `nanochat/gpt.py` 的实现。它用 PyTorch 从头写了一个现代 GPT 模型，默认配置下约 **286M** 参数（含 Value Embeddings），并加入了一些 2023-2024 年的训练技巧。

---

## 目录

1. [先把问题说清楚：GPT 到底在做什么？](#1-先把问题说清楚gpt-到底在做什么)
2. [数据流总览：从一句话到下一个词](#2-数据流总览从一句话到下一个词)
3. [Step 0：Token、Token ID 与 Embedding——把文字变成向量](#3-step-0tokentoken-id-与-embedding把文字变成向量)
4. [Step 1：RMSNorm 归一化——让数值不要太“疯”](#4-step-1rmsnorm-归一化让数值不要太疯)
5. [Step 2：Smear——让模型偷偷看一眼前一个词](#5-step-2smear让模型偷偷看一眼前一个词)
6. [Step 3：保存 x0——防止模型“忘了初心”](#6-step-3保存-x0防止模型忘了初心)
7. [Step 4：Transformer Block——模型的“大脑”](#7-step-4transformer-block模型的-大脑)
   - 7.1 [为什么需要注意力？](#71-为什么需要注意力)
   - 7.2 [Query / Key / Value 是什么？](#72-query--key--value-是什么)
   - 7.3 [自注意力：一句话自己问自己](#73-自注意力一句话自己问自己)
   - 7.4 [因果注意力：不能偷看后面的答案](#74-因果注意力不能偷看后面的答案)
   - 7.5 [多头注意力：多双眼睛看不同关系](#75-多头注意力多双眼睛看不同关系)
   - 7.6 [GQA：让推理更省显存](#76-gqa让推理更省显存)
   - 7.7 [RoPE 旋转位置编码：让模型知道“第几个词”](#77-rope-旋转位置编码让模型知道第几个词)
   - 7.8 [QK Norm + 1.2 缩放：让注意力更聚焦](#78-qk-norm--12-缩放让注意力更聚焦)
   - 7.9 [Value Embeddings：给注意力加点“外部知识”](#79-value-embeddings给注意力加点外部知识)
   - 7.10 [MLP：非线性变换，提取更抽象特征](#710-mlp非线性变换提取更抽象特征)
8. [Step 5：Backout——把低级特征“减掉”](#8-step-5backout把低级特征减掉)
9. [Step 6：LM Head + Softcap——输出每个词的得分](#9-step-6lm-head--softcap输出每个词的得分)
10. [训练 vs 推理：KV Cache 是干嘛的？](#10-训练-vs-推理kv-cache-是干嘛的)
11. [Sliding Window Attention：不是所有层都看全部上下文](#11-sliding-window-attention不是所有层都看全部上下文)
12. [GPTConfig 配置参数解释](#12-gptconfig-配置参数解释)
13. [权重初始化：模型一开始怎么“长”出来的](#13-权重初始化模型一开始怎么长出来的)
14. [参数数量估算](#14-参数数量估算)
15. [nanochat 与原版 GPT-2 的主要区别](#15-nanochat-与原版-gpt-2-的主要区别)
16. [下一步看哪里](#16-下一步看哪里)

---

## 1. 先把问题说清楚：GPT 到底在做什么？

### 1.1 一句话定义

**GPT = 给定前面的文字，预测下一个最可能出现的词。**

比如：

```text
输入：今天天气真
模型输出：好（概率 0.7）、热（0.2）、糟（0.1）...
```

听起来很简单？是的。但就是因为这个任务简单，模型可以通过海量文本自学语法、常识、推理能力。

### 1.2 生成式 Pre-trained 是什么意思？

| 单词 | 含义 |
|------|------|
| **Generative（生成式）** | 模型能一个词一个词地“写”出文本 |
| **Pre-trained（预训练）** | 先在大量无标注文本上学习通用能力，再做具体任务 |
| **Transformer** | 一种基于注意力机制的神经网络结构 |

### 1.3 训练和推理的区别

| 阶段 | 做什么 | 是否需要前向传播 | 是否需要反向传播 |
|------|--------|------------------|------------------|
| **训练** | 看大量句子，学习预测下一个词 | ✅ | ✅ |
| **推理/生成** | 给定开头，让它续写 | ✅ | ❌ |

前向传播：数据从输入一层层流到输出，得到结果。
反向传播：根据预测结果和真实答案的差距，从后往前算每个参数该怎么调整。

---

## 2. 数据流总览：从一句话到下一个词

我们先把整个流程用一句话串起来，后面再拆开每一站。

```text
文字句子
  │
  ▼ 分词器（Tokenizer，不在 gpt.py 里）
Token ID 序列 [1024, 3345, 876, ...]
  │
  ▼ wte（Token Embedding）
向量序列 [batch, seq_len, n_embd]
  │
  ▼ RMSNorm 归一化
数值稳定的向量
  │
  ▼ Smear
混合前一个 token 的信息
  │
  ▼ 12 层 Transformer Block
逐层理解的语义向量
  │
  ▼ Backout
减去中间层的低级特征
  │
  ▼ RMSNorm + LM Head
每个词的得分 logits [batch, seq_len, vocab_size]
  │
  ▼ Softcap + Softmax
概率分布，采样得到下一个词
```

`gpt.py` 里最重要的函数就是 `GPT.forward()`（第 416 行开始），上面的每一步都在里面。

---

## 3. Step 0：Token、Token ID 与 Embedding——把文字变成向量

### 3.1 为什么计算机要先做“分词”？

模型不认识汉字，只认识数字。所以第一步要把句子切成小块，每块对应一个词表里的编号。

```text
句子：今天天气真好
  │
  ▼ 分词
["今天", "天气", "真", "好"]
  │
  ▼ 查词表（Vocab）
[1024, 3345, 876, 2190]
```

这些编号就叫 **Token ID**。词表大小 `vocab_size=32768`，表示模型一共认识 32768 种不同的 token。

> 注意：`gpt.py` 本身不管分词，它接收的是已经分好词的 `idx`（一个整数张量）。分词在 `nanochat/tokenizer.py` 里做。

### 3.2 Embedding：从整数到向量

一个整数本身没有语义。比如 1024 和 1025 在数值上差 1，但可能完全没关系。所以我们需要一个“翻译表”，把每个 token ID 翻译成一个稠密向量。

这个翻译表就是 `transformer.wte`：

```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),
    ...
})
```

输入形状：`[batch, seq_len]`（一批句子，每个句子由 token ID 组成）
输出形状：`[batch, seq_len, n_embd]`（每个 token 变成一个 n_embd 维向量）

`n_embd=768` 是默认配置，所以每个 token 被表示成 768 个浮点数。

```python
x = self.transformer.wte(idx)  # [B, T] -> [B, T, C]
```

### 3.3 为什么需要 Embedding 层，而不是直接把整数喂给神经网络？

神经网络对数字大小敏感。如果你把“猫=1000，狗=1001”直接当数值输入，模型可能会误以为狗比猫“大 1”。

Embedding 做的是把每个整数映射到一个向量空间里的点。训练过程中，语义相近的词（比如“国王”和“女王”）会靠得很近，这就是 Embedding 的魔力。

### 3.4 `padded_vocab_size` 是什么？

代码里不是直接用 `vocab_size`，而是先 padding 到 64 的倍数：

```python
padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
```

原因：GPU 的矩阵乘法对特定尺寸更高效（Tensor Core 喜欢对齐）。输出时会把多出来的 padding 切掉：

```python
logits = logits[..., :self.config.vocab_size]
```

所以这不影响模型效果，只是性能优化。

---

## 4. Step 1：RMSNorm 归一化——让数值不要太“疯”

### 4.1 什么是归一化？

想象一堆数字，有的很大，有的很小。神经网络一层一层乘矩阵，数值容易越来越大或越来越小，导致训练不稳定。

**归一化就是把一组数字按比例缩放到合适的范围，让它们大小差不多。**

### 4.2 LayerNorm vs RMSNorm

常见的 LayerNorm 会做两步：
1. 减均值
2. 除以标准差

RMSNorm 更简单，只做第二步：

```python
def norm(x):
    return F.rms_norm(x, (x.size(-1),))
```

它用“均方根”代替标准差，省去了减均值的操作。实践中效果差不多，但更快更省显存。

### 4.3 为什么放在每个 Block 前面？

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

这种结构叫 **Pre-Norm**：先归一化，再进注意力/MLP。它比 Post-Norm 更稳定，能让深层模型训练起来不那么容易炸。

---

## 5. Step 2：Smear——让模型偷偷看一眼前一个词

### 5.1 为什么需要 Smear？

注意力机制很强，但它需要多层叠加才能建立词与词之间的联系。Smear 是一个很便宜的“捷径”：直接把前一个 token 的 embedding 混进当前 token。

```text
位置 0：今
位置 1：天  ← 把“今”的信息混一点进来
位置 2：气  ← 把“天”的信息混一点进来
```

这有点像 bigram（二元语法）模型：看到“今”后面容易接“天”。

### 5.2 代码实现

```python
# 训练时
if kv_cache is None:
    gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
    x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
```

拆解：
- `x[:, :-1]`：整段序列往前挪一位，变成“前一个词的向量”
- `self.smear_gate(x[:, 1:, :24])`：只看前 24 维，学一个“混合比例”
- `sigmoid`：把比例压到 0~1 之间
- `self.smear_lambda`：整体缩放这个信号的大小
- 最后用 `torch.cat` 把第一个位置（没有前一个词）原样拼回去

### 5.3 为什么只看前 24 维？

这是一个设计选择。前 24 维作为“low-level”信号通道，专门用来传递相邻 token 的位置/局部信息，剩下的维度保留原始语义。这样 Smear 不会干扰深层语义。

### 5.4 推理时怎么办？

推理是一个词一个词生成的，没有“后一个位置”。所以用 KV Cache 里保存的 `prev_embedding`：

```python
x_pre_smear = kv_cache.prev_embedding
kv_cache.prev_embedding = x[:, -1:, :]
# ... 用 x_pre_smear 作为前一个词的向量
```

---

## 6. Step 3：保存 x0——防止模型“忘了初心”

### 6.1 深层网络的问题

Transformer 堆很多层后，每一层都在前一层的输出上做变换。经过 12 层后，最初的 token 信息可能被“稀释”或“扭曲”。

### 6.2 把最初的信息带回来

```python
x0 = x  # 保存经过 Smear 后的初始embedding
```

然后在每一层 Block 之前，都把当前状态 `x` 和初始状态 `x0` 按学习到的比例混合：

```python
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

| 系数 | 作用 |
|------|------|
| `resid_lambdas[i]` | 当前层状态保留多少 |
| `x0_lambdas[i]` | 最初embedding混回多少 |

初始化时：
- `resid_lambdas` 从 1.15 逐渐降到 1.05
- `x0_lambdas` 从 0.20 逐渐降到 0.05

含义：浅层多保留当前状态，但也多回顾初始信息；深层慢慢减少初始信息的混合。

---

## 7. Step 4：Transformer Block——模型的“大脑”

每一层 Block 是 GPT 的核心计算单元：

```python
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
```

每层做两件事：
1. **注意力（Attention）**：看看上下文其他词，更新自己的理解
2. **MLP**：做非线性变换，提取更抽象特征

两者都有 **残差连接**（`x + ...`）：保留原始输入，只叠加变化量。这让深层网络训练更稳定。

### 7.1 为什么需要注意力？

考虑这个句子：

```text
小明把书包放在桌子上，然后他开始写作业。
```

看到“他”时，模型需要知道“他”指的是“小明”。这个信息在很远的前面。

传统 RNN/LSTM 需要一步步传过去，容易丢失。注意力机制让“他”直接和“小明”建立联系，不管距离多远。

### 7.2 Query / Key / Value 是什么？

这是注意力机制的三个角色，可以用图书馆类比：

| 角色 | 类比 | 作用 |
|------|------|------|
| **Query** | 你问的问题 | “我想找和当前词相关的信息” |
| **Key** | 书的标签 | 每个词提供的“主题标签” |
| **Value** | 书的内容 | 每个词实际携带的信息 |

计算过程：
1. 用 Query 和所有 Key 算相似度
2. 相似度高的 Key 对应的 Value 就多拿一些
3. 最后把 Value 加权求和，作为当前词的新表示

### 7.3 自注意力：一句话自己问自己

在 GPT 里，Query、Key、Value 都来自同一个句子，所以叫 **Self-Attention（自注意力）**。

```python
q = self.c_q(x)  # 从 x 生成 Query
k = self.c_k(x)  # 从 x 生成 Key
v = self.c_v(x)  # 从 x 生成 Value
```

`c_q`、`c_k`、`c_v` 都是线性层（没有 bias），作用是把同一个 `x` 投影成三种不同的“视角”。

### 7.4 因果注意力：不能偷看后面的答案

语言模型生成下一个词时，只能看前面的词，不能看后面的词。否则就是作弊。

```text
句子：今天天气真好
预测“天”时，只能看“今”
预测“气”时，只能看“今天”
预测“好”时，只能看“今天天气真”
```

这叫 **Causal Attention（因果/因果掩码注意力）**，代码里由 Flash Attention 自动处理：

```python
y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
```

`causal=True` 表示只关注当前位置及之前的 token。

### 7.5 多头注意力：多双眼睛看不同关系

一个注意力头只能捕捉一种关系。多个头并行，可以捕捉不同层面的模式：

```python
n_embd = 768
n_head = 6
head_dim = n_embd // n_head  # 128
```

每个头独立计算注意力，最后把结果拼起来：

```python
q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
# ... 注意力计算 ...
y = y.contiguous().view(B, T, -1)  # 把头拼回一个向量
y = self.c_proj(y)
```

### 7.6 GQA：让推理更省显存

标准多头注意力里，Query、Key、Value 的头数一样多。推理时为了加速，我们会缓存之前的 Key 和 Value（KV Cache），这很占显存。

**Group-Query Attention（GQA）** 让多个 Query 头共享同一个 Key/Value 头：

```python
n_head = 6      # Query 头数
n_kv_head = 2   # Key/Value 头数
```

这样每 3 个 Query 共享 1 个 Key 和 1 个 Value。缓存量减少到 1/3。

代码里通过断言保证 `n_head % n_kv_head == 0`：

```python
assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
```

默认配置 `n_head=6, n_kv_head=6`，此时就是标准 MHA（每个 Query 一个 KV）。

### 7.7 RoPE 旋转位置编码：让模型知道“第几个词”

#### 7.7.1 为什么需要位置编码？

注意力机制本身对位置不敏感。你把句子顺序打乱，它算出来的注意力分数可能完全一样。但语言顺序很重要：

```text
"狗追猫" 和 "猫追狗" 意思完全不同
```

所以需要告诉模型每个词在句子中的位置。

#### 7.7.2 传统绝对位置编码的问题

早期做法是直接学一组位置向量（比如 position 0 对应向量 A，position 1 对应向量 B）。问题是：
- 只能表示“我在第几个位置”
- 很难泛化到比训练时更长的序列
- 两个位置的“相对距离”不好学

#### 7.7.3 RoPE 的思想

**RoPE（Rotary Position Embedding，旋转位置编码）** 不直接加位置向量，而是把 Query 和 Key 向量按维度两两分组，每组做一个二维旋转。旋转角度和位置有关。

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # 把最后一维分成两半
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

位置越远，旋转角度越大。神奇的是：旋转后的点积只和 **相对位置** 有关。

也就是说，模型不需要死记硬背“位置 5 的向量是什么”，它只要学“距离为 2 的两个词有什么关系”。这让它更容易处理没见过的长序列。

#### 7.7.4 预计算 cos/sin

旋转角度只和位置、维度有关，和输入无关，所以可以预计算：

```python
cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
self.register_buffer("cos", cos, persistent=False)
self.register_buffer("sin", sin, persistent=False)
```

`persistent=False` 表示不存进 checkpoint，因为每次初始化都能重新算。

`_precompute_rotary_embeddings` 里用 `base=100000` 控制旋转频率：

```python
channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
inv_freq = 1.0 / (base ** (channel_range / head_dim))
```

维度越高的分组，旋转越慢；维度越低的分组，旋转越快。高频维度捕捉短距离关系，低频维度捕捉长距离关系。

### 7.8 QK Norm + 1.2 缩放：让注意力更聚焦

#### 7.8.1 为什么要对 Q 和 K 做 Norm？

点积注意力里，如果 Q 和 K 的数值尺度不稳定，softmax 容易“饱和”——某个分数特别大，其他都接近 0，梯度就消失了。

对 Q 和 K 分别做 RMSNorm，可以让它们的尺度稳定：

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = norm(q), norm(k)  # QK norm
```

#### 7.8.2 1.2 缩放是什么？

```python
q = q * 1.2
k = k * 1.2
```

在 norm 之后再整体放大 1.2 倍，让点积结果更大，注意力分布更“尖锐”（sharp）。这样模型更敢于聚焦到真正重要的词上。

> 代码注释也写了这是 TODO，作者还没完全调优这个数值。

### 7.9 Value Embeddings：给注意力加点“外部知识”

#### 7.9.1 什么是 Value Embeddings？

标准注意力里，Value 是从输入 `x` 线性投影出来的。 nanochat 额外加了一个可学习的 Value Embedding 表：

```python
self.value_embeds = nn.ModuleDict({
    str(i): nn.Embedding(padded_vocab_size, kv_dim)
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
})
```

每个 token 除了普通 Embedding `wte`，还有一个专门给 Value 用的 Embedding。这个 Value Embedding 会通过一个门控机制混进注意力里的 `v`：

```python
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

#### 7.9.2 为什么这么做？

可以把它理解为“专家建议通道”。普通 Embedding 主要负责语义，Value Embedding 可以专门学习对注意力有用的一些额外信息。

#### 7.9.3 为什么是隔层使用？

```python
def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2
```

这个函数决定哪些层用 Value Embedding。默认 `n_layer=12` 时，第 1, 3, 5, 7, 9, 11 层用（从 0 开始数的话是 0, 2, 4, 6, 8, 10）。

隔层使用是为了控制参数量和计算量。最后一层总是包含 Value Embedding，因为那是最终输出前的重要层。

### 7.10 MLP：非线性变换，提取更抽象特征

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
```

结构：
1. `c_fc`：把 `n_embd` 映射到 `4*n_embd`
2. 激活函数：`relu(x).square()`，也就是 ReLU²
3. `c_proj`：把 `4*n_embd` 映射回 `n_embd`

#### 7.10.1 为什么需要 MLP？

注意力做的是“加权求和”，本质上是线性的。MLP 引入非线性，让模型能学到更复杂的函数。

可以粗略理解：
- **注意力**：收集上下文信息
- **MLP**：对收集到的信息做深加工

#### 7.10.2 为什么是 ReLU²？

常见选择有 GELU、SwiGLU 等。nanochat 用 `relu(x).square()`，因为它：
- 计算非常简单
- 在论文和实践中有不错的表现
- 平滑且非负

```python
x = F.relu(x).square()
```

---

## 8. Step 5：Backout——把低级特征“减掉”

### 8.1 什么是 Backout？

在 Transformer 中间某一层（默认是 `n_layer // 2`），保存当前的残差状态 `x_backout`。最后输出前，从最终表示里减去一部分 `x_backout`：

```python
backout_layer = n_layer // 2  # 第6层
x_backout = None
for i, block in enumerate(self.transformer.h):
    ...
    if i == backout_layer:
        x_backout = x

if x_backout is not None:
    x = x - self.backout_lambda.to(x.dtype) * x_backout
```

### 8.2 为什么减去中间层？

作者希望浅层学到的低级特征（拼写、局部模式等）不要过度影响最终的语义判断。通过减去中间层表示，可以让高层更专注于抽象语义。

### 8.3 `backout_lambda=0.2`

不是完全减掉，而是减 20%。保留大部分信息，只抑制一部分。

---

## 9. Step 6：LM Head + Softcap——输出每个词的得分

### 9.1 LM Head：把语义向量映射回词表

```python
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
```

输入：`[B, T, n_embd]`
输出：`[B, T, vocab_size]`

每个位置都会输出词表里每个词的“原始得分”，也叫 **logits**。

### 9.2 为什么 logits 不是概率？

logits 可以是任意实数，没有约束。这样模型更容易学习。真正用的时候再 softmax 转成概率。

```python
# 训练时直接算交叉熵，内部会做 softmax
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
```

### 9.3 Softcap：防止 logits 太大

```python
softcap = 15
logits = logits.float()
logits = softcap * torch.tanh(logits / softcap)
```

`tanh` 会把输入压到 `[-1, 1]` 之间，再乘以 15，所以 logits 被限制在 `[-15, 15]`。

好处：防止某些词得分特别高，让训练更稳定。

### 9.4 wte 和 lm_head 是独立的

nanochat 里 `wte` 和 `lm_head` 不共享权重（untied）：

```python
self.transformer = nn.ModuleDict({
    "wte": nn.Embedding(padded_vocab_size, config.n_embd),
    ...
})
self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
```

GPT-2 是共享权重的（tied），nanochat 选择独立，让输入嵌入和输出投影各自优化。

---

## 10. 训练 vs 推理：KV Cache 是干嘛的？

### 10.1 推理为什么慢？

生成文本时，每次只能预测一个词：

```text
输入：[今天]
输出：天
输入：[今天，天]
输出：气
输入：[今天，天，气]
输出：真
...
```

每次输入都变长了，如果每次都重新算所有位置的 Key 和 Value，会非常浪费。

### 10.2 KV Cache 的思想

把之前算好的 Key 和 Value 存起来，下次只算新 token 的 Q/K/V，然后把新的 K/V 拼到缓存里。

```python
if kv_cache is None:
    # 训练：一次性看完整序列
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
else:
    # 推理：复用缓存
    k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
    y = flash_attn.flash_attn_with_kvcache(
        q, k_cache, v_cache,
        k=k, v=v,
        cache_seqlens=kv_cache.cache_seqlens,
        causal=True,
        window_size=window_size,
    )
    if self.layer_idx == kv_cache.n_layers - 1:
        kv_cache.advance(T)
```

`gpt.py` 本身不实现 KV Cache 的细节，它调用 `flash_attn_with_kvcache`。具体缓存逻辑在 `nanochat/engine.py` 里。

### 10.3 训练时为什么不用 KV Cache？

训练时一次性输入整个序列，直接做因果注意力更高效。KV Cache 主要是为了推理加速。

---

## 11. Sliding Window Attention：不是所有层都看全部上下文

### 11.1 为什么有的层只看局部？

注意力计算量和序列长度的平方成正比。如果所有层都看全部 2048 个 token，计算量很大。

观察发现：浅层可能更多处理局部语法、词法，不需要看太远；深层需要理解全局语义，需要全上下文。

### 11.2 `window_pattern` 配置

```python
window_pattern: str = "SSSL"
```

- `S` = Short：只看附近（约 1/4 上下文）
- `L` = Long：看全部上下文

模式会循环应用到各层，但**最后一层强制为 L**。

```python
def _compute_window_sizes(self, config):
    pattern = config.window_pattern.upper()
    long_window = config.sequence_len          # 2048
    short_window = -(-long_window // 4 // 128) * 128  # 768
    char_to_window = {
        "L": (long_window, 0),
        "S": (short_window, 0),
    }
    window_sizes = []
    for layer_idx in range(config.n_layer):
        char = pattern[layer_idx % len(pattern)]
        window_sizes.append(char_to_window[char])
    window_sizes[-1] = (long_window, 0)  # 最后一层强制全上下文
    return window_sizes
```

默认 12 层 + "SSSL" 模式下，窗口大小为：

```text
Layer 0:  S (768)
Layer 1:  S (768)
Layer 2:  S (768)
Layer 3:  L (2048)
Layer 4:  S (768)
Layer 5:  S (768)
Layer 6:  S (768)
Layer 7:  L (2048)
Layer 8:  S (768)
Layer 9:  S (768)
Layer 10: S (768)
Layer 11: L (2048)  # 强制
```

---

## 12. GPTConfig 配置参数解释

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # 最大序列长度
    vocab_size: int = 32768       # 词表大小
    n_layer: int = 12             # Transformer 层数
    n_head: int = 6               # Query 头数
    n_kv_head: int = 6            # Key/Value 头数（=n_head 时为标准 MHA）
    n_embd: int = 768             # 隐藏层维度
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `sequence_len` | 2048 | 一次最多处理多少个 token，也是 RoPE 预计算长度 |
| `vocab_size` | 32768 | 词表大小，决定输出维度 |
| `n_layer` | 12 | 模型深度，层数越多越“深” |
| `n_head` | 6 | 注意力 Query 头数 |
| `n_kv_head` | 6 | 注意力 Key/Value 头数，小于 `n_head` 时启用 GQA |
| `n_embd` | 768 | 模型宽度，每个 token 的向量维度 |
| `window_pattern` | "SSSL" | 滑动窗口模式，S=短 L=长 |

---

## 13. 权重初始化：模型一开始怎么“长”出来的

神经网络训练前需要给参数赋初值。不好的初始化会让模型一开始就在瞎猜或梯度爆炸。

```python
@torch.no_grad()
def init_weights(self):
    # 输入/输出嵌入
    torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
    torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

    # 线性层用均匀分布，标准差约为 1/sqrt(n_embd)
    n_embd = self.config.n_embd
    s = 3**0.5 * n_embd**-0.5
    for block in self.transformer.h:
        torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
        torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
        torch.nn.init.zeros_(block.attn.c_proj.weight)  # 输出投影初始化为0
        torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
        torch.nn.init.zeros_(block.mlp.c_proj.weight)

    # 可学习标量
    for i in range(n_layer):
        self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

    torch.nn.init.zeros_(self.smear_lambda)
    torch.nn.init.constant_(self.backout_lambda, 0.2)
    torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

    # Value embeddings 和 gate
    for ve in self.value_embeds.values():
        torch.nn.init.uniform_(ve.weight, -s, s)
    for block in self.transformer.h:
        if block.attn.ve_gate is not None:
            torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
```

### 13.1 关键设计

| 参数 | 初始化方式 | 原因 |
|------|-----------|------|
| `wte` | normal(std=0.8) | Embedding 需要一定幅度 |
| `lm_head` | normal(std=0.001) | 初始输出很“软”，避免一开始过度自信 |
| `c_q/c_k/c_v` | uniform(±s) | 稳定注意力初值 |
| `c_proj` | zeros | 残差分支初始为 0，相当于先不走这条支路 |
| `c_fc` | uniform(±0.4s) | MLP 输入层缩小一点 |
| `resid_lambdas` | 1.15 → 1.05 | 浅层残差强一点，深层弱一点 |
| `x0_lambdas` | 0.20 → 0.05 | 浅层多回顾初始 embedding |

---

## 14. 参数数量估算

默认配置 `n_embd=768, n_layer=12, n_head=6, n_kv_head=6, vocab_size=32768`，实际用 `sum(p.numel())` 统计得到约 **286M** 参数。分布如下：

| 组件 | 计算 | 参数量 |
|------|------|--------|
| Token Embedding (wte) | 32768 × 768 | ~25.2M |
| LM Head | 768 × 32768 | ~25.2M |
| 每层 Attention | c_q + c_k + c_v + c_proj | ~2.4M |
| 每层 MLP | c_fc + c_proj | ~4.7M |
| 每层 Block 合计 | Attention + MLP | ~7.1M |
| 12 层 Block | 12 × 7.1M | ~85.0M |
| Value Embeddings | 6 层 × 32768 × 768 | ~151.0M |
| 可学习标量 + smear_gate | 可忽略 | ~0.01M |
| **总计** | 上面相加 | **~286M** |

### 14.1 为什么 Value Embeddings 占这么多？

```python
head_dim = config.n_embd // config.n_head  # 128
kv_dim = config.n_kv_head * head_dim        # 6 * 128 = 768
self.value_embeds = nn.ModuleDict({
    str(i): nn.Embedding(padded_vocab_size, kv_dim)
    for i in range(config.n_layer) if has_ve(i, config.n_layer)
})
```

每个 Value Embedding 表的大小和普通词嵌入 `wte` 一样（`vocab_size × n_embd`）。默认 12 层里有 6 层启用 Value Embedding，所以光这一项就是 `6 × 25.2M ≈ 151M`。

这就是为什么默认配置下参数量达到 286M，而不是原来文档里说的“85M~95M”——那个估计漏掉了 Value Embeddings。

### 14.2 想控制参数量怎么办？

- 减少 `n_kv_head`：KV 维度 `kv_dim = n_kv_head × head_dim` 会随之减小
- 减少 `n_layer`：层数少了，Value Embedding 层数也会少
- 禁用 Value Embeddings：需要改代码（把 `has_ve` 永远返回 False）
- 减小 `vocab_size` 或 `n_embd`

### 14.3 常用配置对比

| 模型 | n_embd | n_head | n_kv_head | head_dim | 备注 |
|------|--------|--------|-----------|----------|------|
| nanochat 默认 | 768 | 6 | 6 | 128 | 含 VE 约 286M |
| nanochat 小模型 | 256 | 2 | 2 | 128 | 规模大幅缩小 |
| LLaMA-2 7B | 4096 | 32 | 8 | 128 | 标准 GQA |
| GPT-3 | 12288 | 96 | 96 | 128 | 标准 MHA |

---

## 15. nanochat 与原版 GPT-2 的主要区别

| 特性 | GPT-2 | nanochat |
|------|-------|----------|
| 位置编码 | 可学习绝对位置 | RoPE 旋转位置编码 |
| 注意力 | 标准 MHA | 支持 GQA |
| MLP 激活 | GELU | ReLU² |
| 归一化 | LayerNorm | RMSNorm（无参数） |
| Q/K Norm | 无 | 有 |
| 残差连接 | 标准 | 可学习 `resid_lambdas` + `x0_lambdas` |
| 特殊层 | 无 | Smear、Backout、Value Embeddings |
| 输入/输出 Embedding | 共享权重 | 独立权重 |
| 滑动窗口 | 无 | 有 |

---

## 16. 下一步看哪里

- `nanochat/engine.py`：训练循环、KV Cache 推理、混合精度训练
- `nanochat/tokenizer.py`：分词器，如何把原始文本转成 token ID
- `nanochat/optim.py`：MuonAdamW 优化器，不同参数组用不同学习率
- `nanochat/flash_attention.py`：Flash Attention 封装，自动选 FA3 或 SDPA 回退

---

*本文档基于 nanochat/gpt.py 当前实现整理。如有更新，请以代码为准。*
