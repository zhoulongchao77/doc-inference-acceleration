# 03 Transformer 主干计算：KV Cache、FlashAttention、量化

向量序列进入模型主体——$L$ 个结构相同的 Transformer Block，这是整条路径上算力、显存带宽与优化手段最密集的一环。本章先用 3.1 建立背景、诊断 Decode 为什么慢；再沿三条优化主线展开：3.2 KV Cache 家族（少算、放得下、装得少，能复用见 6.2）、3.3 FlashAttention、3.4 量化。

> **图 7** · 03 Transformer 主干计算结构地图
> **03 章结构地图**：3.1 背景与瓶颈诊断（两节拍 + 显存账 + Roofline）；3.2 KV Cache 家族三层手段（少算 / 放得下 / 装得少；跨请求前缀复用属于并发请求一章，不在本图）；3.3 FlashAttention；3.4 量化。

## 3.1 背景：推理只有前向，Decode 为什么慢

### 推理只有前向：权重固定，不断预测下一个 token

**前向传播（Forward）**是数据从 Embedding 一路算到 logits 的过程，训练和推理都有；**反向传播（Backward）**从损失 Loss 倒推梯度、更新权重，只在训练中存在。**推理阶段权重固定、只有前向、不断预测下一个 token**，不更新任何参数——这也是为什么推理优化的抓手集中在“怎么搬数据、怎么调度”，而不需要保存反向用的中间状态。

### 30 秒回顾 Transformer 与 QKV

向量序列流过 $L$ 个结构相同的 **Transformer Block**（多头自注意力 MHA + 前馈网络 MLP/FFN，配 LayerNorm 与残差），最后由输出 Linear 层投影成词表维度的 logits。注意力做的事，是让当前 token 的 Query 与所有历史 token 的 Key 打分、再按权重取 Value：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

其中 $Q$ 是当前 token 在“问什么”，$K$ 是每个历史 token 的“标签”，$V$ 是每个历史 token 的“内容”。多头（Multi-Head）只是把 Q/K/V 切成多组并行计算后拼回。

容易被忽略的是**另一半参数：MLP/FFN**。每个 Block 里注意力只占约 1/3 参数，剩下约 2/3 在两层（或 SwiGLU 三层）的前馈网络里——Decode 每步“搬权重”，大头其实是搬 MLP。现代模型的主流做法是把 MLP 换成 **MoE（Mixture of Experts，混合专家）**：把一个大 FFN 拆成数十到数百个“专家”，每个 token 由路由器只激活其中少数几个（如 Qwen3-30B-A3B 总参 30.5B、每 token 仅激活 3.3B）。MoE 对推理的含义是**显存按总参数装、带宽与算力按激活参数算**：模型必须完整常驻显存，但每个 token 只需读取被激活专家的权重——Decode 访存大减、TPS 天花板抬高（6.1 会按此修正口径），代价是专家路由引入 All-to-All 通信与负载均衡问题（见 7.3 趋势）。

> **图 8** · MoE 混合专家路由示意
> MoE 路由：每个 token 经 Router 打分只激活少数专家（橙），其余专家（灰）本轮不读取；模型按总参数占显存、按激活参数耗带宽与算力——Qwen3-30B-A3B 为 128 专家选 8，DeepSeek-V3 为 256 选 8。

### 两个节拍：Prefill 与 Decode

> **图 9** · Prefill 与 Decode 两阶段流程
> 推理两阶段：**Prefill** 一次性并行处理整个 prompt 并生成 KV Cache（算力受限）；**Decode** 逐 token 自回归生成、反复读取 KV（带宽受限），每步把新 K/V 追加进缓存。

**为什么 Decode 每次只输入 1 个 token？**不是硬件强制，而是**自回归逻辑 + KV Cache 设计**的自然结果：Prefill 已把全部历史 K/V 存好，Decode 时“新信息”只有刚生成的那 1 个 token，所以本轮的 Q 只有 1 个；它用这 1 个 Q 去和缓存里全部历史 K/V 打分，采样出下一个 token，再把这 1 个 token 的 K/V 追加进缓存。第 $N$ 个 token 无法提前知道第 $N+1$ 个，只能逐步进行。

| 对比项 | Prefill（提示填充） | Decode（逐 token 生成） |
| --- | --- | --- |
| 输入 | 整个 prompt（$n$ 个 token）一次进 | 每次仅 1 个新 token |
| 并行度 | 高，所有 token 并行矩阵乘 | 低，序列维度并行度为 1 |
| 瓶颈类型 | Compute-Bound 算力受限，利用率可达 ~95% | Memory-Bound 带宽受限，利用率可低至 ~12% |
| 主要开销 | 大量 GEMM 矩阵运算 | 反复读取全部权重 + 全部 KV Cache |
| 对应指标 | 决定 TTFT（首 token 时延） | 决定 ITL（token 间时延） |
| 优化抓手 | FlashAttention、张量并行、量化 | PagedAttention、减 KV 访存、PD 分离、投机解码 |

> **关键认知**
> Prefill 和 Decode 的瓶颈完全不同——**一个拼算力、一个拼带宽**。这是后面所有优化（包括 PD 分离）成立的根本原因。

### 显存占用：模型权重、KV Cache 与激活

$$\text{GPU 显存总占用}\ \approx\ \underbrace{P\times b}_{\text{模型权重}}+\underbrace{\mathrm{KV}_{\text{bytes}}}_{\text{KV Cache}}+\underbrace{A}_{\text{激活/中间张量}}+\text{碎片}$$

- 模型权重 ：参数量 $P$ × 每参数字节。BF16/FP16 = 2B/参数，FP8 = 1B，INT4 ≈ 0.5B。如 7B BF16 ≈ 14 GiB ，70B BF16 ≈ 140 GiB ——这部分常驻、不随上下文变化。
- KV Cache ：随并发数与序列长度 线性增长 ，是长上下文、高并发场景下最容易爆显存的部分，也是调度要重点管理的对象。
- 激活 $A$ ：推理时主要是 Prefill 的中间张量，相对权重小得多，Decode 阶段几乎可忽略。

### 先看清 GPU 的存储层级

GPU 不是一块“均匀的大内存”，而是一个**越靠近计算核心越快、但容量越小**的层级结构。硬件细节不是本文重点，只需记住结论：数据在不同层级之间搬运的速度，往往才是推理的真正瓶颈。

> **图 10** · GPU 存储层级金字塔
> GPU 存储层级：片上寄存器/SRAM 带宽极高但以 KB 计；HBM / GDDR6 是模型权重与 KV Cache 的主阵地（H100 HBM3 3.35TB/s、80GB，A10 GDDR6 600GB/s、24GB）；再往下到主机 DRAM、SSD，带宽断崖式下降。

### 显存墙：算得快，搬得慢

当一次计算的**算术强度（Arithmetic Intensity，每搬运 1 字节能做多少次浮点运算，FLOPs/Byte）很低**时，计算单元会大量空转等数据，这就是 Memory-Bound。Decode 阶段每个 token 要把**全部权重 + 全部 KV Cache 从 HBM 读一遍，却只做约 2 倍参数量的运算**，算术强度极低，因此被显存带宽死死卡住。

### Roofline：一张图判断瓶颈在算力还是带宽

> **图 11** · Roofline 模型与 Prefill Decode 落点
> Roofline 模型（Williams et al., 2009）：斜屋顶由显存带宽决定（性能 = β·I），平屋顶由峰值算力决定，二者交点为“脊点”。H100 脊点 ≈ 989/3.35 ≈ 295 FLOPs/B。Decode 落在斜坡（带宽受限，约 5 TF/s 量级），Prefill 贴近平顶（算力受限）。

*A10 口径：FP16 Tensor Core 稠密峰值约 125 TFLOPS、GDDR6 带宽 600 GB/s，脊点 ≈ 125 / 0.6 ≈ 208 FLOPs/B——Decode 同样落在斜坡，结论不变；A10 无 FP8 硬件，FP8 屋顶不适用。*

> **用 Roofline 指导优化**
> Decode 想提速，要么**减少搬运字节**（量化、压缩/驱逐 KV、减 KV 头）；要么**提高算术强度**（多请求拼 batch，让一次搬运服务更多 token，第 6 章）；Prefill 想提速则靠 FlashAttention、张量并行与低精度。

## 3.2 KV Cache 家族：少算、分页、压缩与逐出

如果只记住一个推理优化，那就是 KV Cache。本节把所有围绕 KV Cache 的手段收为一个家族，按四个问题展开：**① 少算**——历史 K/V 只算一次；**② 放得下**——PagedAttention 分页管理；**③ 装得少**——GQA/MQA、MLA 与运行时逐出（KIVI 位宽量化见 3.4）；**④ 能复用**——RadixAttention 前缀树（见 6.2）。

### ① 少算：把历史 K/V 存起来，注意力计算 O(n²) → O(n)

它的思想极其朴素：**把已经算过的历史 token 的 K、V 存起来，新 token 只算自己的 Q，直接复用旧 K/V**，从而把每步“重算全部历史”的开销砍掉。

> **图 12** · 有无 KV Cache 的计算量对比
> KV Cache 把单请求解码的注意力计算量从 $O(n^2)$ 降到 $O(n)$：左图每步计算量随序列线性增长，右图每步恒定。代价是要用显存存放历史 K/V，序列越长占用越高。

### KV Cache 的数据流动：HBM、SRAM 与一次写入

把 KV Cache 放到硬件层面看，它改变的是「数据在 HBM 与片上 SRAM 之间怎么流动」。没有缓存时，**每生成一个新 token，都要把它之前所有 token 的 K/V 重新算一遍**（生成第 k+1 个 token 时，重算的是之前的 1~k 个）：权重反复从 HBM 读进 SRAM、算出的 K/V 用完即弃，下一步全部重来。有了缓存，Prefill 阶段权重只需流式读一遍，算出的 K/V 作为结果写回 HBM 的 KV Cache 区、只写一次；Decode 每生成一个 token，只算自己的 Q/K/V，把之前已缓存 token 的 KV 块从 HBM 读进 SRAM，在片上完成 $QK^\top$、softmax、乘 V，再把新 token 的 K/V 作为增量追加写回。另外要严格区分两层含义：**「缓存随生成变长」说的是已缓存的有效内容逐步增多；而物理显存是按块管理的**——旧式方案按最大生成长度连续预留一整段，PagedAttention（②）则切成固定页、按需分配、写满再取新页，空间总是先备好、内容再逐步写入。

> **图 13** · KV Cache 写入与复用：以「今天 北京 天气 晴朗」为例
> 以「今天、北京、天气、晴朗」为例看 KV Cache 的数据流：无缓存时，每生成一个新 token 都要重算它之前所有 token 的 K/V（生成第 k+1 个时，重算之前的 1~k 个）；有缓存时，Prefill 把已有 token 的 K/V 一次写入 HBM 的 KV Cache 区，Decode 每步只读缓存、并把新 token 的 K/V 追加写入。注意「缓存随生成变长」指有效内容逐步增多；物理显存按块管理——旧式方案按最大长度连续预留，PagedAttention 按需分页。它与 FlashAttention 正交：一个让历史「不重算」，一个让中间矩阵「不落地」。

注意区分两件正交的事：**KV Cache 让历史 K/V「不重算」**（省计算，代价是占用 HBM）；**FlashAttention（3.3）让注意力的中间矩阵「不落地」**——读进 SRAM 的 KV 块在片上算完、结果只写回一次（省访存）。二者叠加，才构成 Decode 的完整数据流。

### 为什么只缓存 K、V，从不缓存 Q？

这来自注意力结构的**根本不对称**：

- $K$、$V$ 是 历史 token 的表示 ，一旦算好就不再改变，后续每一步都要反复使用——天然值得缓存。
- $Q$ 是 “当前这一步”的查询 ，只在当前位置、当前这一次注意力里使用，用完即弃；下一步的新 token 会产生全新的 Q。缓存 Q 没有复用价值，反而白占显存。

一句话：**K/V 面向过去、可复用；Q 面向当下、一次性**。所以 KV Cache 里永远只有 K 和 V。

### KV Cache 显存，一个公式算清

$$\mathrm{KV}_{\text{bytes}}=2\times n_{\text{kv}}\times d_{\text{head}}\times L\times s\times b_{\text{size}}\times \underbrace{b}_{\text{每元素字节}}$$

- $2$ ：K 和 V 两份，最容易漏；
- $n_{\text{kv}}$ ： KV 头数 （GQA/MQA 用它，不是 Query 头数！Q 头再多也不影响 KV 大小）；
- $d_{\text{head}}$ ：单头维度，Llama/Qwen 系列一般为 $128$；
- $L$ ：模型层数； $s$ ：序列总长度（prompt + 生成）； $b_{\text{size}}$ ：并发 batch；
- $b$ ：每元素字节，BF16/FP16 = 2，FP8/INT8 = 1。

### ② 放得下：PagedAttention，像操作系统管内存一样管 KV

缓存本身该怎么在显存里摆放？传统做法给每个请求**按最大生成长度连续预留一整段显存**。但请求实际生成长度未知，预留多了形成**内部碎片**、预留少了直接 OOM；再加上预留的显存无法在请求间灵活共享，浪费惊人（vLLM 论文测得浪费 60–80%）。

**PagedAttention 借鉴 OS 虚拟内存分页**：把 KV Cache 切成固定大小的 **Block（页，如每块 16 个 token）**；逻辑上连续的 KV，物理上可以分散存放，由**页表**记录映射；按需分配、用完即回收。更进一步，**相同前缀（系统提示、公共 few-shot、beam/采样分叉）的 KV 块可以在多个请求间共享（写时复制）**，几乎消灭碎片。

> **图 14** · PagedAttention 分页与页表
> 上：连续预留产生大量内部碎片；下：PagedAttention 把 KV 切成固定块、页表映射，物理块按需分散分配，相同前缀块（橙）跨请求共享。显存浪费从 60–80% 降到个位数。

- **vLLM** [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（Kwon et al. · UC Berkeley · SOSP 2023 · PagedAttention 与 vLLM，吞吐较 HF/传统方案数倍提升）

### ③ 装得少（结构侧）：减头 GQA / MQA

先从模型结构入手，减少每个 token 要缓存的 K/V。

KV Cache 大小正比于 KV 头数，于是模型侧最直接的演进就是**减少 KV 头、让多个 Q 头共享同一组 K/V**：

> **图 15** · MHA GQA MQA 头结构对比
> MHA 中每个 Q 头有专属 K/V；GQA 将多个 Q 头分组共享 K/V（Llama2/3、Qwen 主流）；MQA 所有 Q 头全局共用 1 组 K/V。KV Cache 只与 KV 头数有关。

- **MQA** [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)（Noam Shazeer · 2019 · 首次提出多查询注意力，所有 Q 共享一组 K/V）
- **GQA** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)（Ainslie et al. · Google · EMNLP 2023 · 分组查询注意力，质量与 MHA 接近、KV 按组数缩减，可从 MHA 检查点低成本转换，现为 Llama/Qwen 等主流标配）

> **易错三连**
> ① GQA 必须用 **KV 头数**而非 Q 头数；② 不要漏 K+V 的系数 **2**；③ KV Cache 显存 ≠ 模型权重显存，二者要分开估算再相加。

### ③ 装得少（维度侧）：MLA 多头潜在注意力

GQA/MQA 减的是 KV **头的数量**；DeepSeek 提出的 **MLA（Multi-head Latent Attention）**走另一条路——**把 K、V 向量本身通过低秩投影压缩成更小的潜向量（latent），缓存里只存这个潜向量，Decode 时再投影还原**。KV 头数量不变，但每个 token 缓存体积大幅下降。

> **图 16** · MLA 压缩与还原
> MLA：Prefill 用可学习投影把高维 K/V 编码为低维潜向量，缓存只存潜向量；Decode 再投影还原。压缩/还原矩阵随模型一起训练，注意力精度近无损（DeepSeek-V2/V3）。

- **MLA** [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)（DeepSeek-AI · 2024 · 首次提出 MLA 多头潜在注意力，配合 MLA 大幅压缩 KV Cache）
同属“结构侧装得少”的前沿方向还有**稀疏注意力**：DeepSeek 的 NSA 与月之暗面的 MoBA 让注意力只计算被选中的 KV 块、而非全部历史，把长上下文的注意力开销从 $O(n^2)$ 压到近线性，且设计为**可训练、近无损**——可以理解为“把 KV 驱逐（3.2）直接做进模型结构”的版本，二者均已进入新一代长上下文模型（趋势见 7.3，论文见参考文献）。工程上还有一个更温和的中间档：**滑窗注意力（SWA）**及其与全局注意力的层间混合——大部分层只看最近 W 个 token、每隔几层插一层全局注意力（Mistral 首开，Gemma 2/3 的 5:1 局部/全局交替即此路线；Qwen3-Next 等则把部分层换成线性注意力），KV 占用随窗口封顶、实现简单，是中长上下文模型的实用默认。

### 三个主流模型，代入公式实算

| 模型（BF16） | 结构：层数 / Q头·KV头 / d | 单 token KV | 典型上下文 KV 占用 |
| --- | --- | --- | --- |
| Qwen3-30B-A3B | 48 层 / 32 · 4 / 128（GQA） | $2{\times}48{\times}4{\times}128{\times}2$ = **96 KiB** | 16K ≈ **1.5 GiB**；128K ≈ 12 GiB |
| Llama3-70B | 80 层 / 64 · 8 / 128（GQA） | $2{\times}80{\times}8{\times}128{\times}2$ = **320 KiB** | 32K ≈ 10 GiB；128K ≈ **40 GiB** |
| DeepSeek-V2（MLA） | 60 层 / kv_lora_rank 512（潜变量） | 只存 512 维潜向量 + RoPE 部分 | 128K ≈ **8.4 GiB**，较 MHA 压缩 93.3% |

*口径：KiB/GiB 按 1024 换算；KV 为单请求、单并发，线上还要乘以并发 batch。Qwen3-30B 为 MoE（总参 30.5B、单 token 激活 3.3B），但 KV Cache 只与注意力结构有关，与 MoE 无关。*

### ③ 装得少（运行时）：KV 逐出 H2O 与 SnapKV

结构改动需要重新训练模型；不训练模型，也能在运行时主动丢 KV——这就是逐出（eviction）。

KV Cache 随序列线性膨胀，长上下文下甚至超过权重。但注意力本身是**高度稀疏**的：实证表明，生成时 **95% 以上的历史 token 几乎从未被高权重关注**，真正反复被“看”的只有少数“关键 token”（Heavy Hitters）加上最近的一小段。这给了我们**主动丢弃低价值 KV**的空间（属有损压缩，需控制比例以保质量）。

> **图 17** · 注意力稀疏性与 H2O SnapKV 时机
> 上：注意力稀疏、只有少数关键 token 被反复关注；左下 H2O 在生成中持续保留 Heavy Hitter + 最近窗口、KV 总量恒定；右下 SnapKV 用 prefill 末尾观察窗口投票、在生成前一次性压缩。

**H2O：Heavy-Hitter Oracle · 边生成边驱逐。**

H2O 观察到一个规律：**历史贡献（累计注意力得分）高的 token，未来也更可能被需要**。它在每步维护一个固定大小的 KV 池，保留“累计注意力最高的 Heavy Hitter”加“最近窗口”，其余直接驱逐，从而把 KV Cache 变成**有上界**的滚动窗口，训练免费、可直接嵌入任意引擎。

- **H2O** [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)（Zhang, Zhang, Mirzadeh et al. · NeurIPS 2023 · 边生成边驱逐，KV 显存大幅下降、质量可控）
**SnapKV：生成前「一次看对眼」。**

SnapKV 利用 prefill 末尾若干 query 位置（观察窗口）对各历史 KV 的注意力作为**投票**，选出对当前任务真正重要的 KV 位置，在 Decode 开始前**一次性批量压缩**。它特别契合长 prompt / RAG 场景——大量无关文档块可以在生成前就被丢掉，且因为依据的是真实任务相关的投票，压缩近无损。

- **SnapKV** [SnapKV: LLM Knows What You Are Looking for Before Generation](https://arxiv.org/abs/2404.14469)（Li et al. · NeurIPS 2024 · 观察窗口加权投票 + 生成前一次压缩）
| 对比 | H2O | SnapKV | KIVI（见 3.4） |
| --- | --- | --- | --- |
| 压缩手段 | 整段丢弃低价值 KV | 整段丢弃（投票选块） | 保留全部、降低每元素位宽 |
| 时机 | Decode 中持续驱逐 | Prefill 末、Decode 前一次 | 全程量化 |
| 判据 | 累计注意力 Heavy Hitter + 最近 | 末尾观察窗口注意力投票 | key 按通道 / value 按 token |
| 最适合 | 超长生成、流式对话 | 长 prompt / RAG | 通用、与其他方法叠加 |

**Attention Sink：为什么开头几个 token 永远丢不得。**驱逐类方法有一个共同的实证基础：softmax 要求注意力权重归一，模型会把大量“无处安放”的注意力**倾倒给序列最开始的几个 token**（Attention Sink）——它们被关注的强度远超其语义重要性，一旦被驱逐，输出立刻崩坏。StreamingLLM 利用这一点给出极简方案：**只保留最前 4 个 sink token + 最近滑动窗口**，KV 有界、无需微调，即可在远超训练长度的流式输入上稳定生成。它既是所有驱逐策略“特殊照顾开头”的物理直觉，也是无限流式对话（直播字幕、持续监控等场景）的经典解法。

- **StrLLM** [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)（Xiao et al. · MIT · ICLR 2024 · Attention Sink 现象 + 滑动窗口，KV 有界的无限流式生成）

> **KV Cache 家族一览**
> 少算（缓存复用）→ 放得下（PagedAttention）→ 装得少（GQA/MQA、MLA、H2O/SnapKV；位宽量化 KIVI 见 3.4）→ 能复用（RadixAttention 见 6.2）。现代引擎里这几件事通常同时开启，且与 3.3/3.4 正交可叠加。

## 3.3 FlashAttention：用分块把 N×N 矩阵“焊”在片上

先把标准注意力的完整计算写全（与 3.1 的公式一致）：

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

拆成三步看：$S=\dfrac{QK^{\top}}{\sqrt{d_k}}$（打分，**别忘了除以 $\sqrt{d_k}$ 做缩放**，防止分数随维度变大把 softmax 推进饱和区）、$P=\mathrm{softmax}(S)$（归一化成注意力权重）、$O=PV$（按权重加权求和 Value）。标准实现要把 $S$、$P$ 这些 **$N\times N$ 中间矩阵反复写回 HBM 再读出**，访存量是 $O(N^2)$。FlashAttention 的做法是**算子融合 + 分块（Tiling）+ 在线 Softmax（Online Softmax）**：把 Q/K/V 分块搬进片上 SRAM，在片上连续完成 $QK^\top$、softmax、乘 V，用流式方式逐块更新 softmax 归一化因子，最终输出 $O$ 只写回 HBM 一次，HBM 访存降到 $O(N)$。

> **图 18** · FlashAttention 分块与在线 softmax
> FlashAttention 的 Tiling：Q/K/V 分块载入 SRAM，片上融合完成注意力与在线 softmax，避免 N×N 矩阵落地 HBM。它与 KV Cache 正交——一个优化 Prefill 的算子访存，一个优化 Decode 的重复计算。

- **FA1** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（Dao et al. · NeurIPS 2022 · IO 感知、分块 + 在线 softmax）
- **FA2** [FlashAttention-2: Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)（Tri Dao · 2023 · 约 2× FA1，减少非 matmul 开销）
- **FA3** [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)（Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao · NeurIPS 2024 · Hopper TMA 异步、Warp 特化、FP8）

### 同层的另一件武器：CUDA Graph 消除启动开销

FlashAttention 优化的是单个算子**内部**的访存；Decode 还有一层更“机械”的开销——每步前向要发起成百上千次 kernel launch，每次 launch 本身有数微秒的 CPU→GPU 开销，而 Decode 的每个 kernel 只算几十微秒，**小 batch 下 launch 开销能占单步耗时的两三成**（这正是 6.1 实测低于理论天花板的原因之一）。**CUDA Graph 把 Decode 的整条计算图一次性“录制”下来，之后每步整体回放**：CPU 只发一次启动指令，launch 开销几乎归零，vLLM、SGLang、TensorRT-LLM 均已默认对 Decode 开启。它与 FlashAttention、量化正交，是“不改算法、只改执行方式”的纯系统优化——代价是计算图形状（batch、序列长度档位）必须固定，引擎需要按档位预录多份。

## 3.4 量化：直接减少要搬运的字节

Decode 是 Memory-Bound，**把权重/KV 从 BF16 降到 FP8/INT8/INT4，直接把搬运字节减半甚至减到 1/4**，提速立竿见影；代价是引入微小精度损失（属**有损**优化，需校准或少量微调）。

| 技术 | 作用对象 | 方法要点 | 典型精度 |
| --- | --- | --- | --- |
| SmoothQuant | 权重 + 激活 | 把激活的离群值平滑迁移到权重，实现 W8A8 训练后量化 | INT8，近无损 |
| GPTQ | 仅权重 | 利用二阶 Hessian 信息逐层一键量化，误差补偿 | W4 / W8 |
| AWQ | 仅权重 | 识别并保护约 1% 显著权重通道（缩放而非直接裁剪） | W4，部署最广 |
| KIVI | KV Cache | Key 按通道、Value 按 token 分块非对称量化，无需调参 | 2bit 近无损 |
| FP8 | 权重 + 激活 | Hopper / Blackwell 原生支持，需校准或 QAT | FP8（E4M3） |

- **AWQ** [AWQ: Activation-aware Weight Quantization for On-Device LLM Compression](https://arxiv.org/abs/2306.00978)（Lin et al. · MLSys 2024）
- **KIVI** [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)（Zhang et al. · ICML 2024）
*A10 等 Ampere 卡**没有 FP8 硬件**，优先选择 INT8（SmoothQuant，A10 INT8 Tensor Core 可直接加速）或 W4（AWQ / GPTQ）；FP8 仅在 Hopper / Blackwell 上可用。*

> **量化的两条边界**
> ① **W4A16 只省搬运、不省计算**：权重以 4bit 存储、计算前反量化回 FP16，只有配套专用 kernel（Marlin、ExLlamaV2 等）才能把“省带宽”兑现为真实加速，且 batch 调大、负载转入 Compute-Bound 后收益递减；② **只有激活也量化（W8A8 / FP8）才能吃到低精度 Tensor Core 的计算加速**——A10 上的 SmoothQuant INT8 即属此类，而纯 W4 方案的价值主要在省显存与带宽。先用 Roofline（3.1）判瓶颈，再决定量化谁。
