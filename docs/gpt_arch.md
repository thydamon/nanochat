# nanochat/gpt.py Transformer 架构详解

这是项目的核心模型文件，让我按模块逐步拆解。

---

## 1. 整体结构

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

---

## 2. 配置 GPTConfig

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048      # 支持的最大上下文长度
    vocab_size: int = 32768       # 词表大小
    n_layer: int = 12             # Transformer 层数
    n_head: int = 6               # Query 头数
    n_kv_head: int = 6            # Key/Value 头数 (GQA)
    n_embd: int = 768             # 隐藏层维度
    window_pattern: str = "SSSL"  # 滑动窗口模式
```

默认配置约 **85M 参数**的模型。

### GPTConfig 参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sequence_len` | 2048 | 上下文窗口长度。超过会报错，同时用于预计算 RoPE 表。 |
| `vocab_size` | 32768 | 词表大小，决定 `lm_head` 输出维度。 |
| `n_layer` | 12 | Transformer Block 层数，越多越深。 |
| `n_head` | 6 | Query 头数，`n_head * head_dim = n_embd`。 |
| `n_kv_head` | 6 | Key/Value 头数，支持 GQA。当 `< n_head` 时多个 Q 共享同一 KV，节省显存。 |
| `n_embd` | 768 | 隐向量维度，决定模型宽度。 |
| `window_pattern` | `"SSSL"` | 滑动窗口模式：`S`=短窗口(~768 tokens)，`L`=全上下文。最后层强制为 `L`。 |

### GQA 的意义

当 `n_kv_head = n_head` 时，每个 Q/K/V 投影维度相同。当 `n_kv_head < n_head` 时：

```
例如 n_head=8, n_kv_head=2：
  Q 有 8 个头，每个头独立
  K,V 只有 2 个头，被 8 个 Q 头共享
  KV cache 大幅减少（8→2），但表达能力略有下降
```

### Sliding Window 的意义

```python
window_pattern = "SSSL"  # 前3层用短窗口，最后层用全上下文
```

浅层用局部注意力减少计算量，深层用全局注意力捕捉长距离依赖。这是 [Mistral](https://arxiv.org/abs/2310.06825) 的设计思路。

### 参数量的粗估

```python
# 85M 模型参数粗估
n_embd=768, n_layer=12, n_head=6, head_dim=128
# wte:       vocab_size * n_embd = 32768 * 768 ≈ 25M
# 每层 Block:
#   attn:    3 * (n_embd * n_embd) = 3 * 768² ≈ 1.8M
#   mlp:     4 * n_embd * n_embd * 2 = 2.3M  (c_fc + c_proj)
# 12层合计约 50M
# lm_head:  n_embd * vocab_size = 25M
# 总计 ≈ 85M
```

---

## 3. 核心组件

### 3.1 CausalSelfAttention — 带 GQA 的自注意力

#### 第一步：什么是注意力（Attention）？

想象你在图书馆找书：

```
Query（查询）: "我想找关于机器学习的书"
Key（键）:     每本书的分类标签
Value（值）:   书的实际内容
```

注意力机制做的事情：**用 Query 去查询所有的 Key，找到最相关的书，然后读取 Value（内容）**。

在代码里，这就是一个加权求和：

```python
# 注意力本质：加权求和
output = Σ (similarity(query, key_i) / √d) * value_i
#                           ↑  除以维度开方，让方差稳定
#         ↓
#    softmax归一化，得到权重
```

---

#### 第二步：自注意力（Self-Attention）

之前是"书"和"查询"是**不同**的东西。自注意力是让序列自己和自己比较：

```
输入: "我想学习机器学习"
        ↓
每个词都同时是 Query、Key、Value
        ↓
"学习" 作为 Query，去问 "机器"、"学习"、"我"、"想"
        ↓
找到和"学习"最相关的词（比如"机器"），加权求和
        ↓
输出: "学习"的新表示（融合了上下文）
```

```python
# 自注意力：Q、K、V 都来自同一个输入
Q = x @ W_q   # x是输入，"我在找什么"
K = x @ W_k   # x是输入，"我是什么"
V = x @ W_v   # x是输入，"我的内容是什么"
```

---

#### 第三步：多头注意力（Multi-Head Attention）

一个"注意力头"只能学到一种关系。**多个头并行学习不同类型的关系**：

```
"我想学习机器学习"

Head 1 (语法关系): "学习" ←→ "想"    发现"想"修饰"学习"
Head 2 (语义关系): "机器" ←→ "学习"  发现"机器"修饰"学习"
Head 3 (位置关系): "机器" ←→ "学习"  发现相邻关系
```

```python
# 多头的实现：把维度拆分
# n_embd=768, n_head=6, head_dim=128

q = x @ W_q  # (B, T, 768)
q = q.view(B, T, 6, 128)  # 分成6个头，每头128维
# 每头独立做注意力，最后 concat 在一起
```

为什么拆开？
- 类比：一个人只能同时关注一件事
- 分头后，6个人可以关注6种不同的模式
- 最后把结果拼起来，得到更丰富的表示

---

#### 第四步：Causal Attention（因果注意力）

语言模型必须**只能看前面的词**，不能偷看后面的答案（这是训练目标决定的）。

```
输入: "今天天气真好"
位置:  0   1   2   3   4

"天" (位置1) 只能看 "今" (位置0)     ✓
"气" (位置2) 只能看 "今、天"        ✓
"好" (位置4) 只能看 "今、天、气、"   ✓
              不能看位置5,6,7...       ✗ 这是未来
```

```python
# causal attention 本质：加一个 mask
attention_scores = q @ k.T  # (T, T) 矩阵

# 方法1: 手动 mask
attention_scores = attention_scores.masked_fill(
    mask,  # 上三角为True（不能看的部分）
    float('-inf')
)

# 方法2: Flash Attention 的 causal 参数
flash_attn.flash_attn_func(q, k, v, causal=True)
# Flash Attention 在内部自动处理，效率更高
```

---

#### 第五步：Group-Query Attention（GQA）

标准 MHA 的问题：**KV Cache 太大了**。

推理时，每次生成一个新 token，都需要把之前的 K/V 存起来（KV Cache）。

```
MHA (n_head=6):
  6个Q头 + 6个K头 + 6个V头
  推理时要缓存 6+6 个向量

GQA (n_head=6, n_kv_head=2):
  6个Q头（各自独立）
  2个K头（每3个Q头共享1个）
  2个V头（每3个Q头共享1个）
  推理时只需缓存 2+2 个向量
```

**显存节省 3 倍！** 代价是 Q 头之间的 KV 信息不能完全独立交互，但在实践中影响很小。

```python
# 代码里的体现
self.c_q = Linear(768, 6 * 128)   # Q: 6个头，每个头128维
self.c_k = Linear(768, 2 * 128)   # K: 2个头，每个头128维（更少！）
self.c_v = Linear(768, 2 * 128)   # V: 2个头，每个头128维（更少！）
```

---

#### 第六步：nanochat 的实现

现在对照代码看整个流程：

```python
def forward(self, x, ve, cos_sin, window_size, kv_cache):
    B, T, C = x.size()  # (Batch, 序列长度, 768)

    # ① 投影成 Q、K、V
    q = self.c_q(x).view(B, T, 6, 128)   # Q: 6个头
    k = self.c_k(x).view(B, T, 2, 128)   # K: 2个头（GQA）
    v = self.c_v(x).view(B, T, 2, 128)   # V: 2个头（GQA）

    # ② 旋转位置编码（RoPE）
    # 让"学习"知道自己在位置2，而不是位置5
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)

    # ③ QK 归一化 + 缩放
    q, k = norm(q), norm(k)   # 稳定训练
    q, k = q * 1.2, k * 1.2   # 让注意力更 sharp

    # ④ Flash Attention 计算
    # FA3 自动处理 causal mask，效率高
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # ⑤ 合并多头 + 输出投影
    y = y.view(B, T, -1)      # 合并6个头
    y = self.c_proj(y)        # 投影回 768 维
    return y
```

---

#### 总结：为什么这样设计？

| 组件 | 作用 | 为什么用 |
|------|------|---------|
| **Q/K/V 投影** | 提取不同角度的信息 | 同一输入，三种视角 |
| **多头** | 并行学习多种关系 | 一个头学一种模式 |
| **Causal** | 防止看到未来 | 语言模型必须自左向右生成 |
| **GQA** | 减少 KV Cache | 推理省显存 |
| **RoPE** | 编码位置信息 | 旋转比加法更优雅，能处理超长序列 |
| **QK Norm** | 稳定注意力分数 | 防止 softmax 退化 |
| **Flash Attention** | 高效计算 | 训练快，省显存 |

### 3.2 RoPE 旋转位置编码

```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * (-sin)
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], 3)
```

对 Q/K 的每对维度做旋转，让模型学会相对位置关系，不需要可学习的位置参数。

### 3.3 MLP — 使用 ReLU² 激活

```python
class MLP(nn.Module):
    def forward(self, x):
        x = self.c_fc(x)           # n_embd → 4*n_embd
        x = F.relu(x).square()      # ReLU² 激活函数
        x = self.c_proj(x)         # 4*n_embd → n_embd
        return x
```

使用 `(ReLU(x))²` 而非标准的 GELU 或 SwiGLU，是这个项目的特殊设计。

---

## 4. Block — 标准的 Pre-Norm 结构

```python
class Block(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)  # 注意力残差
        x = x + self.mlp(norm(x))                                        # MLP残差
        return x
```

使用 **RMSNorm**（无可学习参数），计算轻量。

---

## 5. GPT 前向传播流程

```python
def forward(self, idx, targets=None, kv_cache=None):
    # ① Token Embedding
    x = self.transformer.wte(idx)
    x = norm(x)

    # ② Smear：混合前一个token的embedding（类似bigram信息）
    gate = smear_lambda * sigmoid(smear_gate(x[:, 1:, :24]))
    x = cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]])

    # ③ 通过12层Block
    x0 = x  # 保存初始embedding
    for i, block in enumerate(self.transformer.h):
        x = resid_lambdas[i] * x + x0_lambdas[i] * x0  # 可学习的残差混合
        ve = value_embeds[i](idx)                       # Value Embedding
        x = block(x, ve, cos_sin, window_size, kv_cache)

    # ④ Backout：减去中间层残差（去低级特征）
    x = x - backout_lambda * x_backout

    # ⑤ LM Head
    logits = lm_head(norm(x))

    # ⑥ Softcap + 交叉熵损失
    logits = softcap * tanh(logits / softcap)
    loss = cross_entropy(logits, targets)
```

---

## 6. 特殊设计亮点

| 特性 | 说明 |
|------|------|
| **GQA** | 减少 KV cache，推理更快 |
| **Sliding Window** | 浅层用滑动窗口（768 tokens），深层用全上下文 |
| **Value Embeddings** | ResFormer 风格：可学习的 value 表注入注意力 |
| **Smear** | 混合前 token embedding，廉价但有效的 bigram 信号 |
| **Backout** | 减中间层残差，去除低级特征（拼写、格式） |
| **Pre-Norm + RMSNorm** | 无偏置，省显存，训练稳定 |
| **ReLU²** | MLP 激活函数 |
| **QK Norm + 1.2 缩放** | 让注意力更 sharp |
| **Untied Weights** | wte 和 lm_head 独立初始化 |
| **Softcap** | logits 裁剪到 [-15, 15]，防止数值爆炸 |

---

## 7. 权重初始化

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

## 8. 与标准 GPT-2 的主要区别

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

## 9. 下一步学习

这是从零重写的现代 Transformer 实现，集成了 2023-2024 年的多项训练技巧（RoPE、GQA、ReLU²、Sliding Window）。接下来可以看：

- `nanochat/engine.py` — 理解 KV Cache 推理机制
- `nanochat/common.py` — 工具函数