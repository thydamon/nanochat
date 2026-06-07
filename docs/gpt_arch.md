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

## 1. 模型整体结构

### 1.1 GPT 组件一览

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

### 1.2 各组件通俗解释

| 组件 | 通俗理解 | 类比 |
|------|----------|------|
| **transformer.wte** | 把文字转成数字向量 | 翻译员 |
| **transformer.h (Block×12)** | 12层"思考单元"，逐层理解语义 | 12个大脑区域 |
| **lm_head** | 把处理后的向量转成词表概率 | 决策者 |
| **resid_lambdas** | 控制每层的"思考深度" | 音量旋钮 |
| **x0_lambdas** | 把最初的信息混合回来 | 复习笔记 |
| **smear_gate + smear_lambda** | 混合前一个词的信息 | 上下文线索 |
| **backout_lambda** | 去掉低级特征 | 纠错 |
| **value_embeds** | 额外注入的"知识向量" | 专家建议 |

### 1.3 前向传播顺序（对应 nanochat/gpt.py 行428-472）

```
1. wte（行428）：x = transformer.wte(idx)  # 把 token 转成向量
2. norm（行430）：归一化
3. smear（行436-449）：混合前一个 token 的 embedding
4. x0 保存（行452）：保存初始 embedding
5. Block×12（行456-461）：
   - resid_lambdas + x0_lambdas 混合
   - value_embeds 注入
   - block 处理（注意力 + MLP）
6. backout（行463-464）：减去中间层残差
7. lm_head（行469）：预测下一个词
8. softcap（行472）：logits 裁剪
```

### 1.4 通俗例子：写小说

**场景**：让 GPT 续写故事 "小明走进森林，..."

| 组件 | 实际工作 |
|------|----------|
| **wte** | "小" → [0.1, 0.3, ...] |
| **smear** | "林" = "林" + 0.3 × "森" |
| **Block×12** | 12轮深度思考（每轮包含注意力） |
| **backout** | 去掉低级错误 |
| **lm_head** | → "发现"(0.7), "遇到"(0.2) |

### 1.5 参数计算（默认配置约 85M~95M）

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

### 2.1 Embedding 维度（n_embd）

Embedding 维度是模型表示每个 token 的**向量长度**：

```python
# 每个 token 被表示为一个 n_embd 维的向量
n_embd = 768  # "学习" → [0.1, 0.5, ..., 0.2]  # 768 个数字
```

| n_embd | 模型容量 | 表达能力 |
|--------|----------|----------|
| 256 | 小 | 较弱 |
| 768 | 中 | 中等 |
| 4096 | 大 | 强 |

### 2.2 wte（Token Embedding）

把 token 转成向量：

```python
x = transformer.wte(idx)  # [batch, seq_len] → [batch, seq_len, n_embd]
```

### 2.3 lm_head（输出层）

把向量转成词表概率：

```python
logits = lm_head(norm(x))  # [batch, seq_len, n_embd] → [batch, seq_len, vocab_size]
```

**注意**：nanochat 使用 untied weights（wte 和 lm_head 独立初始化），而非 GPT-2 的共享权重。

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