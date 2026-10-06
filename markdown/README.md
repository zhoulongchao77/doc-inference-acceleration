# 《大模型推理与加速技术全景》分章 Markdown 索引

给 AI / 自己阅读用：**需要哪章就只打开哪个文件**，不必把整个 HTML 塞进上下文。

| 文件 | 章节 | 小节 |
| --- | --- | --- |
| [00-开篇.md](00-开篇.md) | 00 开篇：全景地图与推理加速为什么值得做 | 全景地图：一次请求的完整流水线与 AI 基建八层；为什么推理加速值得做：两大根本矛盾 |
| [01-输入与 Tokenize.md](01-输入与 Tokenize.md) | 01 输入与 Tokenizer：文字和图片变成 ID | Tokenizer：文本怎样切成序号；图片怎样进来：视觉编码器与图像 token |
| [02-Embedding.md](02-Embedding.md) | 02 Embedding：ID 转向量，位置编码就位 | Embedding：一次近乎免费的查表；位置编码：给向量装上“坐标” |
| [03-推理计算.md](03-推理计算.md) | 03 推理计算：KV Cache、显存墙、FlashAttn、量化与压缩 | 推理只有前向：权重固定，不断预测下一个 token；30 秒回顾 Transformer 与 QKV；两个节拍：Prefill 与 Decode；KV Cache：推理加速第一性原理；为什么只缓存 K、V，从不缓存 Q？；KV Cache 显存，一个公式算清；从头数减少：MHA → GQA → MQA；从维度压缩：MLA 多头潜在注意力；三个主流模型，代入公式实算；显存占用：模型权重、KV Cache 与激活；先看清 GPU 的存储层级；显存墙：算得快，搬得慢；Roofline：一张图判断瓶颈在算力还是带宽；FlashAttention：用分块把 N×N 矩阵“焊”在片上；量化：直接减少要搬运的字节；KV 驱逐：注意力稀疏，95% 历史可以丢；H2O：Heavy-Hitter Oracle；SnapKV：生成前“一次看对眼” |
| [04-并发横切面.md](04-并发横切面.md) | 04 并发横切面：批处理、分页、前缀复用与 PD 分离 | Static Batching 的三个痛点；PagedAttention：像操作系统管内存一样管 KV；Chunked Prefill：别让长 prompt 一次堵死所有人；RadixAttention：把公共前缀缓存成一棵前缀树；从显存带宽反推：7B 模型的理论 TPS 极限；服务指标体系：先定义“好”，再谈优化；架构级：PD 分离 |
| [05-采样.md](05-采样.md) | 05 采样：从 logits 到下一个 ID | 采样旋钮：温度、Top-K、Top-P；约束解码：把非法 ID 提前屏蔽；主流工具链；投机解码：一次前向，确认多个 ID |
| [06-Detokenizer.md](06-Detokenizer.md) | 06 Detokenizer：把 ID 拼回文字，送到用户眼前 | 三个工程细节 |
| [07-总结、落地清单与前沿趋势.md](07-总结、落地清单与前沿趋势.md) | 07 总结、落地清单与前沿趋势 | 一句话串起全文；工程落地 Checklist（按流水线环节）；前沿趋势（2025–2026）；建议的读论文顺序；参考文献 |

## 使用建议

1. **局部修改/提问**：只把对应章节的 .md 发给 AI（单章约 2–16 KB）。
2. **整体校对/串讲**：用 `大模型推理与加速技术全景-全文.md`（约 60 KB）。
3. **看图**：图中信息已转写为「图号 · 主题 + 图注」；需要原图时打开同名 HTML。
4. HTML 改动后重新生成：`python tools/html_split_md.py`。
