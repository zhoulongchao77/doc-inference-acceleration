# -*- coding: utf-8 -*-
"""
一次性重组第 3/4/5 章：
- 03 改名「推理加速：KV Cache、FlashAttention、量化」；3.1 背景与瓶颈诊断（合并原 3.3）；
  3.2 KV Cache 家族（少算 / 放得下 PagedAttention / 装得少 GQA·MLA·逐出 / 能复用见 4.2）；
  3.3 FlashAttention；3.4 量化。新增 KV 数据流图，重绘结构地图。
- 04 改名「并发横切面：批处理、前缀复用与 PD 分离」；4.1 背景（痛点+天花板+指标），
  4.2 机内优化（Continuous Batching / Chunked Prefill / RadixAttention），4.3 PD 分离。
- 05：5.1 背景（采样旋钮），5.2 约束解码，5.3 投机解码家族。
图号顺延重排（图7..图30），静态目录同步。
用法：python tools/rebuild_ch345.py
"""
import os, re
from bs4 import BeautifulSoup, Tag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "大模型推理与加速技术全景.html")

def E(soup, html):
    """解析单个元素。"""
    return BeautifulSoup(html, "html.parser").find()

def blocks(sec):
    """section 的直接元素子节点。"""
    return [ch for ch in sec.children if isinstance(ch, Tag)]

def bidx(sec, name, contains=None, nth=0):
    """按标签名+文本片段定位直接子节点下标。"""
    hit = 0
    for i, ch in enumerate(blocks(sec)):
        if ch.name == name and (contains is None or contains in ch.get_text()):
            if hit == nth:
                return i
            hit += 1
    raise ValueError("not found: %s %s #%d" % (name, contains, nth))

def bget(sec, name, contains=None, nth=0):
    return blocks(sec)[bidx(sec, name, contains, nth)]

def rebuild(soup):
    sec3, sec4, sec5 = (soup.find("section", id=x) for x in ("ch3", "ch4", "ch5"))
    b3, b4, b5 = blocks(sec3), blocks(sec4), blocks(sec5)

    # ---------- 抓取 ch3 原块 ----------
    h3_cog = b3[bidx(sec3, "h3", "三个整体认知")]
    h4_fwd = b3[bidx(sec3, "h4", "推理只有前向")]
    p_fwd = b3[bidx(sec3, "p", "前向传播（Forward）")]
    h4_qkv = b3[bidx(sec3, "h4", "30 秒回顾")]
    p_qkv1 = b3[bidx(sec3, "p", "向量序列流过")]
    fb_attn = b3[bidx(sec3, "div", "mathrm{Attention}")]
    p_qkv2 = b3[bidx(sec3, "p", "其中 $Q$")]
    h4_beat = b3[bidx(sec3, "h4", "两个节拍")]
    fig8 = b3[bidx(sec3, "figure", "推理两阶段")]
    p_why1 = b3[bidx(sec3, "p", "为什么 Decode 每次只输入")]
    tbl_pd = b3[bidx(sec3, "div", "对比项")]
    cal_pd = b3[bidx(sec3, "div", "关键认知")]

    h3_kv = b3[bidx(sec3, "h3", "KV Cache：推理加速")]
    h4_save = b3[bidx(sec3, "h4", "把历史 K/V 存起来")]
    p_save = b3[bidx(sec3, "p", "如果只记住一个推理优化")]
    fig9 = b3[bidx(sec3, "figure", "把单请求解码的注意力")]
    h4_why = b3[bidx(sec3, "h4", "为什么只缓存")]
    p_why2 = b3[bidx(sec3, "p", "这来自注意力结构")]
    ul_why = b3[bidx(sec3, "ul", "历史 token 的表示")]
    p_why3 = b3[bidx(sec3, "p", "K/V 面向过去")]
    h4_formula = b3[bidx(sec3, "h4", "KV Cache 显存")]
    fb_kv = b3[bidx(sec3, "div", "KV}_{")]
    ul_form = b3[bidx(sec3, "ul", "K 和 V 两份")]
    cal_err = b3[bidx(sec3, "div", "易错三连")]

    h3_wall = b3[bidx(sec3, "h3", "显存墙与 Roofline")]
    p_wall_intro = b3[bidx(sec3, "p", "看清 KV Cache")]
    h4_occ = b3[bidx(sec3, "h4", "显存占用：模型权重")]
    fb_occ = b3[bidx(sec3, "div", "GPU 显存总占用")]
    ul_occ = b3[bidx(sec3, "ul", "参数量")]
    h4_lvl = b3[bidx(sec3, "h4", "先看清 GPU")]
    p_lvl = b3[bidx(sec3, "p", "GPU 不是一块")]
    fig12 = b3[bidx(sec3, "figure", "片上寄存器/SRAM")]
    h4_wall = b3[bidx(sec3, "h4", "显存墙：算得快")]
    p_wall = b3[bidx(sec3, "p", "当一次计算的")]
    h4_roof = b3[bidx(sec3, "h4", "Roofline：一张图")]
    fig13 = b3[bidx(sec3, "figure", "Williams")]
    note_a10 = b3[bidx(sec3, "small", "A10 口径")]
    cal_roof = b3[bidx(sec3, "div", "用 Roofline")]

    h3_flash = b3[bidx(sec3, "h3", "FlashAttention：用分块")]
    p_flash = b3[bidx(sec3, "p", "标准注意力要把")]
    fig14 = b3[bidx(sec3, "figure", "Tiling")]
    grid_fa = b3[bidx(sec3, "div", "FA1")]
    paper_fa3 = b3[bidx(sec3, "div", "FA3")]

    h3_quant = b3[bidx(sec3, "h3", "量化：直接减少")]
    p_quant = b3[bidx(sec3, "p", "Decode 是 Memory-Bound")]
    tbl_quant = b3[bidx(sec3, "div", "SmoothQuant")]
    grid_awq = b3[bidx(sec3, "div", "AWQ: Activation-aware")]
    note_fp8 = b3[bidx(sec3, "small", "A10 等 Ampere")]

    h3_comp = b3[bidx(sec3, "h3", "KV 压缩")]
    p_comp_intro = b3[bidx(sec3, "p", "模型侧缩小 KV Cache")]
    h4_head = b3[bidx(sec3, "h4", "从头数减少")]
    p_head = b3[bidx(sec3, "p", "KV Cache 大小正比")]
    fig10 = b3[bidx(sec3, "figure", "MHA 中每个 Q 头")]
    paper_mqa = b3[bidx(sec3, "div", "MQA")]
    paper_gqa = b3[bidx(sec3, "div", "GQA")]
    h4_mla = b3[bidx(sec3, "h4", "从维度压缩")]
    p_mla = b3[bidx(sec3, "p", "GQA/MQA 减的是")]
    fig11 = b3[bidx(sec3, "figure", "MLA：Prefill")]
    paper_mla = b3[bidx(sec3, "div", "MLA")]
    h4_calc = b3[bidx(sec3, "h4", "三个主流模型")]
    tbl_calc = b3[bidx(sec3, "div", "Qwen3-30B-A3B")]
    note_calc = b3[bidx(sec3, "small", "口径：KiB/GiB")]

    h3_evict = b3[bidx(sec3, "h3", "KV 逐出")]
    p_evict = b3[bidx(sec3, "p", "KV Cache 随序列线性膨胀")]
    fig15 = b3[bidx(sec3, "figure", "注意力稀疏、只有少数")]
    h4_h2o = b3[bidx(sec3, "h4", "H2O：Heavy-Hitter")]
    p_h2o = b3[bidx(sec3, "p", "H2O 观察到")]
    paper_h2o = b3[bidx(sec3, "div", "H2O")]
    h4_snap = b3[bidx(sec3, "h4", "SnapKV：生成前")]
    p_snap = b3[bidx(sec3, "p", "SnapKV 利用")]
    paper_snap = b3[bidx(sec3, "div", "SnapKV")]
    tbl_cmp = b3[bidx(sec3, "div", "压缩手段")]

    # ---------- 抓取 ch4 原块 ----------
    h4_gap = b4[bidx(sec4, "h4", "Prefill 和 Decode 的资源诉求")]
    p_gap1 = b4[bidx(sec4, "p", "同一个模型、同一块卡")]
    ul_gap = b4[bidx(sec4, "ul", "强算力")]
    p_gap2 = b4[bidx(sec4, "p", "把两阶段")]
    fig21 = b4[bidx(sec4, "figure", "吞吐差约两个数量级")]
    h4_how = b4[bidx(sec4, "h4", "PD 分离怎么做")]
    p_how = b4[bidx(sec4, "p", "核心是把两阶段")]
    fig22 = b4[bidx(sec4, "figure", "全局调度器把请求导向")]
    h4_papers = b4[bidx(sec4, "h4", "三篇奠基论文")]
    paper_dist = b4[bidx(sec4, "div", "DistServe")]
    paper_moon = b4[bidx(sec4, "div", "Mooncake")]
    paper_split = b4[bidx(sec4, "div", "Splitwise")]
    tbl_three = b4[bidx(sec4, "div", "核心主张")]
    h4_xfer = b4[bidx(sec4, "h4", "关键工程问题")]
    tbl_xfer = b4[bidx(sec4, "div", "NVLink")]
    p_xfer = b4[bidx(sec4, "p", "工程上还会做")]
    h4_front = b4[bidx(sec4, "h4", "前沿：PD 分离")]
    cal_dynamo = b4[bidx(sec4, "div", "NVIDIA Dynamo")]
    cal_three = b4[bidx(sec4, "div", "显存三件套")]

    h3_static = b4[bidx(sec4, "h3", "Static Batching")]
    p_static = b4[bidx(sec4, "p", "最简单的批处理")]
    ul_static = b4[bidx(sec4, "ul", "木桶效应")]
    fig17 = b4[bidx(sec4, "figure", "整批同进同出")]
    paper_orca = b4[bidx(sec4, "div", "ORCA")]
    h3_page = b4[bidx(sec4, "h3", "PagedAttention：像操作系统")]
    p_page1 = b4[bidx(sec4, "p", "传统做法给每个请求")]
    p_page2 = b4[bidx(sec4, "p", "PagedAttention 借鉴")]
    fig18 = b4[bidx(sec4, "figure", "连续预留产生大量内部碎片")]
    paper_vllm = b4[bidx(sec4, "div", "vLLM")]
    h3_chunk = b4[bidx(sec4, "h3", "Chunked Prefill")]
    p_chunk = b4[bidx(sec4, "p", "Prefill 是 Compute-Bound")]
    fig19 = b4[bidx(sec4, "figure", "长 Prefill 被切成 chunk")]
    grid_sar = b4[bidx(sec4, "div", "SARATHI")]
    h3_radix = b4[bidx(sec4, "h3", "RadixAttention")]
    p_radix = b4[bidx(sec4, "p", "多轮对话、Agent")]
    fig20 = b4[bidx(sec4, "figure", "基数树：")]
    paper_sglang = b4[bidx(sec4, "div", "SGLANG")]
    h3_tps = b4[bidx(sec4, "h3", "从显存带宽反推")]
    style_tps = b4[bidx(sec4, "style", "flow4")]
    flow_tps = b4[bidx(sec4, "div", "STEP 1")]
    cal_tps = b4[bidx(sec4, "div", "理论 vs 现实")]
    h3_metrics = b4[bidx(sec4, "h3", "服务指标体系")]
    tbl_metrics1 = b4[bidx(sec4, "div", "Time To First Token")]
    tbl_metrics2 = b4[bidx(sec4, "div", "现象 / 问题")]

    # ---------- 抓取 ch5 原块 ----------
    h3_knob = b5[bidx(sec5, "h3", "采样旋钮")]
    p_knob = b5[bidx(sec5, "p", "确定性方法")]
    fig23 = b5[bidx(sec5, "figure", "温度对 softmax 的影响")]
    tbl_knob = b5[bidx(sec5, "div", "贪心 Greedy")]
    note_knob = b5[bidx(sec5, "small", "惩罚项在 softmax")]
    h3_con = b5[bidx(sec5, "h3", "约束解码：把非法")]
    p_con = b5[bidx(sec5, "p", "做后端最常需要")]
    fig24 = b5[bidx(sec5, "figure", "文法先编译成状态机")]
    h3_tools = b5[bidx(sec5, "h3", "主流工具链")]
    tbl_tools = b5[bidx(sec5, "div", "Outlines")]
    grid_papers = b5[bidx(sec5, "div", "OUTLINES")]
    paper_gcd = b5[bidx(sec5, "div", "Grammar-Constrained Decoding for Structured")]
    cal_remind = b5[bidx(sec5, "div", "两个工程提醒")]
    h3_spec = b5[bidx(sec5, "h3", "投机解码：一次前向")]
    p_spec = b5[bidx(sec5, "p", "前面的优化都在")]
    h4_draft = b5[bidx(sec5, "h4", "两阶段：Draft")]
    fig25 = b5[bidx(sec5, "figure", "猜测者快速连续写出")]
    h4_lossless = b5[bidx(sec5, "h4", "为什么能")]
    p_loss1 = b5[bidx(sec5, "p", "设目标模型分布")]
    fb_loss = b5[bidx(sec5, "div", "P(\\text{接受}")]
    p_loss2 = b5[bidx(sec5, "p", "决定接受")]
    paper_spec1 = b5[bidx(sec5, "div", "SPEC-DEC")]
    paper_spec2 = b5[bidx(sec5, "div", "SPEC-SAMP")]
    h4_self = b5[bidx(sec5, "h4", "自投机")]
    p_self = b5[bidx(sec5, "p", "不想维护一个独立")]
    paper_self = b5[bidx(sec5, "div", "D&V")]
    h4_medusa = b5[bidx(sec5, "h4", "Medusa：给大模型")]
    p_medusa = b5[bidx(sec5, "p", "Medusa 在冻结")]
    fig26 = b5[bidx(sec5, "figure", "冻结主干 + 多个新增候选头")]
    paper_medusa = b5[bidx(sec5, "div", "MEDUSA")]
    h4_eagle = b5[bidx(sec5, "h4", "EAGLE：在")]
    p_eagle = b5[bidx(sec5, "p", "Medusa 直接猜 token")]
    ul_eagle = b5[bidx(sec5, "ul", "EAGLE（ICML")]
    grid_eagle = b5[bidx(sec5, "div", "EAGLE")]
    paper_eagle3 = b5[bidx(sec5, "div", "EAGLE-3")]
    h4_mtp = b5[bidx(sec5, "h4", "MTP：把")]
    p_mtp = b5[bidx(sec5, "p", "DeepSeek-V3 的")]
    fig27 = b5[bidx(sec5, "figure", "主模型与 MTP 模块串联")]
    paper_mtp = b5[bidx(sec5, "div", "DSv3")]
    h4_look = b5[bidx(sec5, "h4", "Lookahead Decoding")]
    p_look = b5[bidx(sec5, "p", "Lookahead Decoding 完全")]
    fig28 = b5[bidx(sec5, "figure", "2D 窗口内 Guess")]
    paper_look = b5[bidx(sec5, "div", "LOOKAHEAD")]
    h4_cmpfam = b5[bidx(sec5, "h4", "家族全对比")]
    tbl_cmpfam = b5[bidx(sec5, "div", "猜测者来源")]
    note_cmpfam = b5[bidx(sec5, "small", "加速比为各论文")]
    fig29 = b5[bidx(sec5, "figure", "先看“能不能改模型”")]

    # ================= 新 SVG / 文案 =================
    NEW_MAP3 = '''<figure class="fig fig-wide">
<svg aria-label="03 推理加速结构地图" role="img" viewBox="0 0 960 606">
<defs><clipPath id="nclip3"><rect height="586" rx="14" width="940" x="10" y="10"></rect></clipPath></defs>
<g font-family="inherit">
<rect fill="#f7f9fc" height="586" rx="14" stroke="#1e3a5f" stroke-width="2" width="940" x="10" y="10"></rect>
<g clip-path="url(#nclip3)"><rect fill="#1e3a5f" height="46" width="940" x="10" y="10"></rect></g>
<text fill="#fff" font-size="15" font-weight="800" x="30" y="39">03 推理加速 · 先诊断瓶颈，再沿三条优化主线开方</text>
<!-- 3.1 -->
<rect fill="#eef2f7" height="118" rx="10" stroke="#94a3b8" width="908" x="26" y="68"></rect>
<rect fill="#23456f" height="118" rx="10" width="172" x="26" y="68"></rect>
<text fill="#fff" font-size="14" font-weight="800" text-anchor="middle" x="112" y="112">3.1 背景与</text>
<text fill="#fff" font-size="14" font-weight="800" text-anchor="middle" x="112" y="134">瓶颈诊断</text>
<g font-size="11.5">
<rect fill="#fff" height="26" rx="7" stroke="#c9d6e4" width="118" x="212" y="80"></rect><text fill="#16233a" font-weight="700" text-anchor="middle" x="271" y="98">推理只有前向</text>
<rect fill="#fff" height="26" rx="7" stroke="#c9d6e4" width="118" x="340" y="80"></rect><text fill="#16233a" font-weight="700" text-anchor="middle" x="399" y="98">QKV 结构</text>
<rect fill="#fff" height="26" rx="7" stroke="#c9d6e4" width="132" x="468" y="80"></rect><text fill="#16233a" font-weight="700" text-anchor="middle" x="534" y="98">Prefill/Decode 两节拍</text>
<rect fill="#fff" height="26" rx="7" stroke="#c9d6e4" width="104" x="610" y="80"></rect><text fill="#16233a" font-weight="700" text-anchor="middle" x="662" y="98">显存账</text>
<rect fill="#fff" height="26" rx="7" stroke="#c9d6e4" width="118" x="724" y="80"></rect><text fill="#16233a" font-weight="700" text-anchor="middle" x="783" y="98">GPU 存储层级</text>
<rect fill="#fff" height="26" rx="7" stroke="#c9d6e4" width="96" x="850" y="80"></rect><text fill="#16233a" font-weight="700" text-anchor="middle" x="898" y="98">显存墙</text>
<rect fill="#fdeee2" height="26" rx="7" stroke="#f0b28e" width="120" x="212" y="116"></rect><text fill="#d7541a" font-weight="700" text-anchor="middle" x="272" y="134">Roofline 诊断</text>
<text fill="#d7541a" font-weight="800" x="352" y="134">→ 结论：Decode 落在带宽斜坡（Memory-Bound），提速不能靠堆算力</text>
<text fill="#697a92" font-size="11" x="212" y="172">诊断顺序：前向数据流 → 两节拍 → 显存账 / 存储层级 → Roofline 落点</text>
</g>
<!-- 3.2 -->
<rect fill="#eef4f3" height="196" rx="10" stroke="#7fc4bd" width="908" x="26" y="198"></rect>
<rect fill="#0d9488" height="196" rx="10" width="172" x="26" y="198"></rect>
<text fill="#fff" font-size="14" font-weight="800" text-anchor="middle" x="112" y="252">3.2 KV Cache</text>
<text fill="#fff" font-size="14" font-weight="800" text-anchor="middle" x="112" y="274">家族</text>
<text fill="#d4ece9" font-size="10.5" text-anchor="middle" x="112" y="300">围绕「历史 K/V」</text>
<text fill="#d4ece9" font-size="10.5" text-anchor="middle" x="112" y="316">的四层手段</text>
<g text-anchor="middle">
<rect fill="#fff" height="120" rx="9" stroke="#bfe0db" width="166" x="212" y="222"></rect>
<text fill="#0f766e" font-size="12.5" font-weight="800" x="295" y="248">① 少算</text>
<text fill="#16233a" font-size="11" x="295" y="274">历史 K/V 只算一次</text>
<text fill="#697a92" font-size="10.5" x="295" y="296">注意力 O(n²)→O(n)</text>
<text fill="#697a92" font-size="10.5" x="295" y="316">KV Cache 本体</text>
<rect fill="#fff" height="120" rx="9" stroke="#bfe0db" width="166" x="392" y="222"></rect>
<text fill="#0f766e" font-size="12.5" font-weight="800" x="475" y="248">② 放得下</text>
<text fill="#16233a" font-size="11" x="475" y="274">PagedAttention</text>
<text fill="#697a92" font-size="10.5" x="475" y="296">切 block + 页表映射</text>
<text fill="#697a92" font-size="10.5" x="475" y="316">碎片 60–80%→个位数</text>
<rect fill="#fff" height="120" rx="9" stroke="#bfe0db" width="186" x="572" y="222"></rect>
<text fill="#0f766e" font-size="12.5" font-weight="800" x="665" y="248">③ 装得少</text>
<text fill="#16233a" font-size="10.5" x="665" y="272">GQA/MQA 减头 · MLA 降维</text>
<text fill="#16233a" font-size="10.5" x="665" y="292">H2O/SnapKV 逐出</text>
<text fill="#697a92" font-size="10.5" x="665" y="314">KIVI 位宽量化见 3.4</text>
<rect fill="#fff" height="120" rx="9" stroke="#bfe0db" width="150" x="772" y="222"></rect>
<text fill="#0f766e" font-size="12.5" font-weight="800" x="847" y="248">④ 能复用</text>
<text fill="#16233a" font-size="10.5" x="847" y="274">RadixAttention</text>
<text fill="#16233a" font-size="10.5" x="847" y="294">前缀树</text>
<text fill="#697a92" font-size="10.5" x="847" y="316">详见 4.2</text>
<text fill="#0f766e" font-size="11" font-weight="700" x="212" y="368">家族关系：①是第一性原理，②管摆放，③管体积，④管跨请求复用；现代引擎通常同时开启</text>
</g>
<!-- 3.3 / 3.4 -->
<rect fill="#eef2f7" height="78" rx="10" stroke="#94a3b8" width="446" x="26" y="408"></rect>
<rect fill="#23456f" height="78" rx="10" width="120" x="26" y="408"></rect>
<text fill="#fff" font-size="13.5" font-weight="800" text-anchor="middle" x="86" y="440">3.3 Flash</text>
<text fill="#fff" font-size="13.5" font-weight="800" text-anchor="middle" x="86" y="460">Attention</text>
<text fill="#16233a" font-size="11.5" font-weight="700" x="160" y="438">分块 HBM→SRAM，片上融合</text>
<text fill="#697a92" font-size="11" x="160" y="462">N×N 中间矩阵不落地，HBM 访存 O(N²)→O(N)</text>
<rect fill="#fdf3ec" height="78" rx="10" stroke="#f0b28e" width="446" x="488" y="408"></rect>
<rect fill="#e8590c" height="78" rx="10" width="120" x="488" y="408"></rect>
<text fill="#fff" font-size="14" font-weight="800" text-anchor="middle" x="548" y="452">3.4 量化</text>
<text fill="#16233a" font-size="11.5" font-weight="700" x="622" y="438">BF16→FP8 / INT8 / INT4</text>
<text fill="#697a92" font-size="11" x="622" y="462">搬运字节减半 ~ 1/4（有损，需校准）</text>
<rect fill="#e9edf3" height="44" rx="9" width="908" x="26" y="498"></rect>
<text fill="#16233a" font-size="11.5" font-weight="700" x="42" y="518">主线分工：3.2 让历史 K/V 不重算、好摆放、体积小；3.3 让注意力中间矩阵不落地；3.4 让每个字节更便宜——彼此正交、可叠加。</text>
<text fill="#697a92" font-size="11.5" x="42" y="536">另一条思路是减少前向「次数」（一次前向出多个 token）：投机解码家族，见第 05 章。</text>
</g>
</svg>
<figcaption><b>图 7</b><span><b>03 章结构地图</b>：3.1 背景与瓶颈诊断（两节拍 + 显存账 + Roofline）；3.2 KV Cache 家族四层手段（少算 / 放得下 / 装得少 / 能复用）；3.3 FlashAttention；3.4 量化。</span></figcaption>
</figure>'''

    NEW_FIG10 = '''<figure class="fig">
<svg aria-label="KV Cache 在 HBM 与 SRAM 间的数据流动" role="img" viewBox="0 0 900 430">
<defs><marker id="ar10" markerHeight="10" markerWidth="10" orient="auto" refX="8" refY="5"><path d="M0,0 L10,5 L0,10 Z" fill="#3c4d66"></path></marker></defs>
<g font-family="inherit">
<line stroke="#d7dee8" stroke-dasharray="5 5" x1="450" x2="450" y1="18" y2="400"></line>
<!-- left: no cache -->
<text fill="#16233a" font-size="13.5" font-weight="800" text-anchor="middle" x="225" y="34">无 KV Cache：每步重算全部历史</text>
<rect fill="#f3f6fa" height="96" rx="12" stroke="#d7dee8" width="380" x="35" y="50"></rect>
<text fill="#697a92" font-family="JetBrains Mono" font-size="11" x="50" y="72">HBM · 慢、容量大</text>
<rect fill="#23456f" height="50" rx="8" width="230" x="110" y="82"></rect><text fill="#fff" font-size="12.5" font-weight="700" text-anchor="middle" x="225" y="113">模型权重</text>
<rect fill="#e2f1ef" height="110" rx="12" stroke="#bfe0db" width="380" x="35" y="236"></rect>
<text fill="#0d9488" font-family="JetBrains Mono" font-size="11" x="50" y="258">片上 SRAM · 快、容量小</text>
<rect fill="#0d9488" height="56" rx="8" width="230" x="110" y="270"></rect><text fill="#fff" font-size="11.5" font-weight="700" text-anchor="middle" x="225" y="296">QKV 投影 + 注意力</text>
<g stroke="#d7541a" stroke-width="2.4" marker-end="url(#ar10)">
<line x1="225" x2="225" y1="132" y2="266"></line>
</g>
<text fill="#d7541a" font-size="11" font-weight="700" text-anchor="start" x="238" y="180">第 k 步：为全部 k 个</text>
<text fill="#d7541a" font-size="11" font-weight="700" text-anchor="start" x="238" y="198">历史 token 重算 K/V</text>
<path d="M150,330 A78,60 0 0,0 300,330" fill="none" stroke="#d7541a" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ar10)"></path>
<text fill="#d7541a" font-size="11" text-anchor="middle" x="225" y="372">每步重复，K/V 用完即弃 → 总计算 O(n²)</text>
<!-- right: cache -->
<text fill="#16233a" font-size="13.5" font-weight="800" text-anchor="middle" x="675" y="34">有 KV Cache：一次写入，反复读取</text>
<rect fill="#f3f6fa" height="120" rx="12" stroke="#d7dee8" width="380" x="485" y="50"></rect>
<text fill="#697a92" font-family="JetBrains Mono" font-size="11" x="500" y="72">HBM</text>
<rect fill="#23456f" height="50" rx="8" width="120" x="500" y="82"></rect><text fill="#fff" font-size="12" font-weight="700" text-anchor="middle" x="560" y="113">权重</text>
<rect fill="#0d9488" height="50" rx="8" width="150" x="640" y="82"></rect><text fill="#fff" font-size="11.5" font-weight="700" text-anchor="middle" x="715" y="104">KV Cache 区</text><text fill="#d4ece9" font-size="9.5" text-anchor="middle" x="715" y="120">随生成变长</text>
<rect fill="#e2f1ef" height="120" rx="12" stroke="#bfe0db" width="380" x="485" y="236"></rect>
<text fill="#0d9488" font-family="JetBrains Mono" font-size="11" x="500" y="258">SRAM</text>
<rect fill="#0d9488" height="62" rx="8" width="250" x="550" y="270"></rect><text fill="#fff" font-size="11" font-weight="700" text-anchor="middle" x="675" y="296">片上分块算注意力</text><text fill="#d4ece9" font-size="9.5" text-anchor="middle" x="675" y="316">QKᵀ·softmax··V（FlashAttn）</text>
<!-- arrows -->
<g stroke="#3c4d66" stroke-width="2" marker-end="url(#ar10)">
<line x1="560" x2="600" y1="132" y2="266"></line>
<line x1="700" x2="690" y1="132" y2="266"></line>
</g>
<text fill="#3c4d66" font-size="10.5" text-anchor="start" x="600" y="180">Prefill：权重流式读一遍</text>
<text fill="#3c4d66" font-size="10.5" text-anchor="start" x="600" y="198">Decode：每步只读 KV 块</text>
<path d="M760,266 L800,200" fill="none" stroke="#0f766e" stroke-width="2.2" marker-end="url(#ar10)"></path>
<text fill="#0f766e" font-size="10.5" font-weight="700" text-anchor="start" x="772" y="240">K/V 写一次</text>
<path d="M620,332 A70,46 0 0,1 730,332" fill="none" stroke="#b9832b" stroke-width="2" stroke-dasharray="6 4" marker-end="url(#ar10)"></path>
<text fill="#b9832b" font-size="10.5" text-anchor="middle" x="675" y="372">新 K/V 增量追加；历史不重算 → O(n)</text>
<text fill="#697a92" font-size="11" text-anchor="middle" x="450" y="412">左：算力浪费在重复投影；右：用 HBM 空间换计算，再靠片上融合把读取成本压到最低</text>
</g>
</svg>
<figcaption><b>图 10</b><span>KV Cache 的硬件数据流：无缓存时每步把权重读进 SRAM、为全部历史重算 K/V；有缓存时 K/V 在 Prefill 算好后只向 HBM 写一次，Decode 每步只把 KV 块读进 SRAM、片上完成注意力，新 K/V 增量追加。它与 FlashAttention 正交：一个让历史「不重算」，一个让中间矩阵「不落地」。</span></figcaption>
</figure>'''

    NEW_MAP4 = '''<figure class="fig fig-wide">
<svg aria-label="04 并发横切面结构地图" role="img" viewBox="0 0 960 522">
<defs><clipPath id="nclip4"><rect height="502" rx="14" width="940" x="10" y="10"></rect></clipPath>
<marker id="m4a" markerHeight="9" markerWidth="9" orient="auto" refX="7" refY="4.5"><path d="M0,0 L9,4.5 L0,9 Z" fill="#33415c"></path></marker></defs>
<g font-family="inherit">
<rect fill="#f7f9fc" height="502" rx="14" stroke="#33415c" stroke-width="2" width="940" x="10" y="10"></rect>
<g clip-path="url(#nclip4)"><rect fill="#33415c" height="46" width="940" x="10" y="10"></rect></g>
<text fill="#fff" font-size="15" font-weight="800" x="30" y="39">04 并发横切面 · 多请求同时处在流水线的不同环节</text>
<!-- scheduler row -->
<rect fill="#eef2f7" height="46" rx="10" stroke="#94a3b8" stroke-dasharray="5 4" width="180" x="26" y="66"></rect>
<text fill="#16233a" font-size="12" font-weight="700" text-anchor="middle" x="116" y="94">多请求持续到达</text>
<line stroke="#33415c" stroke-width="2" marker-end="url(#m4a)" x1="212" x2="276" y1="89" y2="89"></line>
<rect fill="#16233a" height="46" rx="10" width="280" x="282" y="66"></rect>
<text fill="#fff" font-size="12.5" font-weight="800" text-anchor="middle" x="422" y="94">全局调度器（每 iteration 重排）</text>
<line stroke="#33415c" stroke-width="2" marker-end="url(#m4a)" x1="568" x2="632" y1="89" y2="89"></line>
<rect fill="#eef4f3" height="46" rx="10" stroke="#7fc4bd" width="180" x="638" y="66"></rect>
<text fill="#0f766e" font-size="12" font-weight="700" text-anchor="middle" x="728" y="94">GPU 资源池</text>
<!-- 4.1 -->
<rect fill="#eef2f7" height="104" rx="10" stroke="#94a3b8" width="908" x="26" y="130"></rect>
<rect fill="#23456f" height="104" rx="10" width="150" x="26" y="130"></rect>
<text fill="#fff" font-size="14" font-weight="800" text-anchor="middle" x="101" y="176">4.1 背景</text>
<text fill="#cfe0f2" font-size="10.5" text-anchor="middle" x="101" y="200">痛点·天花板·指标</text>
<g font-size="11">
<rect fill="#fff" height="62" rx="8" stroke="#c9d3df" width="226" x="190" y="150"></rect>
<text fill="#16233a" font-weight="700" x="204" y="172">Static Batching 三痛点</text>
<text fill="#697a92" x="204" y="192">木桶效应 · 无法动态加入</text>
<text fill="#697a92" x="204" y="210">按最大长度预留→碎片</text>
<rect fill="#fff" height="62" rx="8" stroke="#c9d3df" width="226" x="430" y="150"></rect>
<text fill="#16233a" font-weight="700" x="444" y="172">带宽反推 TPS 天花板</text>
<text fill="#697a92" x="444" y="192">7B/A10 ≈ 43 TPS（理论）</text>
<text fill="#697a92" x="444" y="210">实测 ~30，差距即优化空间</text>
<rect fill="#fff" height="62" rx="8" stroke="#c9d3df" width="252" x="670" y="150"></rect>
<text fill="#16233a" font-weight="700" x="684" y="172">服务指标体系</text>
<text fill="#697a92" x="684" y="192">TTFT · ITL · TPS</text>
<text fill="#697a92" x="684" y="210">Goodput（SLO 达标吞吐）</text>
</g>
<!-- 4.2 -->
<rect fill="#eef4f3" height="118" rx="10" stroke="#7fc4bd" width="908" x="26" y="246"></rect>
<rect fill="#0d9488" height="118" rx="10" width="150" x="26" y="246"></rect>
<text fill="#fff" font-size="13.5" font-weight="800" text-anchor="middle" x="101" y="292">4.2 机内优化</text>
<text fill="#d4ece9" font-size="10.5" text-anchor="middle" x="101" y="316">调度器统一驱动</text>
<g text-anchor="middle">
<rect fill="#fff" height="76" rx="9" stroke="#bfe0db" width="226" x="190" y="266"></rect>
<text fill="#0f766e" font-size="12" font-weight="800" x="303" y="292">Continuous Batching</text>
<text fill="#697a92" font-size="10.5" x="303" y="314">iteration 级进出</text>
<text fill="#697a92" font-size="10.5" x="303" y="332">GPU 持续满载（ORCA）</text>
<rect fill="#fff" height="76" rx="9" stroke="#bfe0db" width="226" x="430" y="266"></rect>
<text fill="#0f766e" font-size="12" font-weight="800" x="543" y="292">Chunked Prefill</text>
<text fill="#697a92" font-size="10.5" x="543" y="314">长 prompt 切块</text>
<text fill="#697a92" font-size="10.5" x="543" y="332">与 Decode 交错护 ITL</text>
<rect fill="#fff" height="76" rx="9" stroke="#bfe0db" width="252" x="670" y="266"></rect>
<text fill="#0f766e" font-size="12" font-weight="800" x="796" y="292">RadixAttention</text>
<text fill="#697a92" font-size="10.5" x="796" y="314">基数树 + 最长前缀匹配</text>
<text fill="#697a92" font-size="10.5" x="796" y="332">公共前缀只算一次</text>
</g>
<!-- 4.3 -->
<rect fill="#f6f0e6" height="96" rx="10" stroke="#d9b878" width="908" x="26" y="376"></rect>
<rect fill="#b9832b" height="96" rx="10" width="150" x="26" y="376"></rect>
<text fill="#fff" font-size="13.5" font-weight="800" text-anchor="middle" x="101" y="414">4.3 架构解耦</text>
<text fill="#f6ead2" font-size="10.5" text-anchor="middle" x="101" y="438">PD 分离</text>
<text fill="#16233a" font-size="11.5" font-weight="700" x="190" y="406">Prefill 池 → KV 高速迁移（NVLink / RDMA / NIXL）→ Decode 池</text>
<text fill="#697a92" font-size="11" x="190" y="430">两池硬件、卡数、并行策略独立扩缩容；P/D 吞吐差约 140 倍，混部双输</text>
<text fill="#697a92" font-size="11" x="190" y="454">DistServe · Mooncake · Splitwise · NVIDIA Dynamo</text>
<rect fill="#e9edf3" height="30" rx="8" width="908" x="26" y="482"></rect>
<text fill="#33415c" font-size="11" font-weight="700" x="42" y="502">脉络：机内手段把单 GPU 打满；P/D 矛盾无法调和时走向架构解耦——目标：Goodput 最大化、每 token 成本最小化</text>
</g>
</svg>
<figcaption><b>图 18</b><span><b>04 章结构地图</b>：4.1 背景（Static Batching 痛点、TPS 天花板、指标体系）；4.2 机内优化（Continuous Batching / Chunked Prefill / RadixAttention）；4.3 架构解耦（PD 分离）。</span></figcaption>
</figure>'''

    # ================= 组装 ch3 =================
    sec3.find("h2").string = "推理加速：KV Cache、FlashAttention、量化"
    intro3 = sec3.find("p")
    intro3.string = ("向量序列进入模型主体——$L$ 个结构相同的 Transformer Block，这是整条路径上算力、显存带宽与优化手段最密集的一环。"
        "本章先用 3.1 建立背景、诊断 Decode 为什么慢；再沿三条优化主线展开：3.2 KV Cache 家族（少算、放得下、装得少，能复用见 4.2）、"
        "3.3 FlashAttention、3.4 量化。")

    new3 = [b3[0], intro3, b3[bidx(sec3, "figure", "流水线位置导航")], E(soup, NEW_MAP3)]
    # 3.1
    new3 += [E(soup, '<h3>背景与瓶颈：推理只有前向，Decode 为什么慢</h3>'),
             h4_fwd, p_fwd, h4_qkv, p_qkv1, fb_attn, p_qkv2,
             h4_beat, fig8, p_why1, tbl_pd, cal_pd,
             E(soup, '<h4>显存占用：模型权重、KV Cache 与激活</h4>'), fb_occ, ul_occ,
             h4_lvl, p_lvl, fig12, h4_wall, p_wall,
             h4_roof, fig13, note_a10, cal_roof]
    # 3.2
    p_kv_intro = E(soup, "<p>如果只记住一个推理优化，那就是 KV Cache。本节把所有围绕 KV Cache 的手段收为一个家族，按四个问题展开："
        "<strong>① 少算</strong>——历史 K/V 只算一次；<strong>② 放得下</strong>——PagedAttention 分页管理；"
        "<strong>③ 装得少</strong>——GQA/MQA、MLA 与运行时逐出（KIVI 位宽量化见 3.4）；"
        "<strong>④ 能复用</strong>——RadixAttention 前缀树（见 4.2）。</p>")
    p_flow1 = E(soup, "<p>把 KV Cache 放到硬件层面看，它改变的是「数据在 HBM 与片上 SRAM 之间怎么流动」。没有缓存时，第 $k$ 步要为全部 $k$ 个历史 token 重新做 QKV 投影："
        "权重反复从 HBM 读进 SRAM、算出的 K/V 用完即弃，下一步全部重来。有了缓存，Prefill 阶段权重只需流式读一遍，算出的 K/V 作为结果写回 HBM 的 KV Cache 区、只写一次；"
        "Decode 每步把需要的 KV 块从 HBM 读进 SRAM，在片上完成 $QK^\\top$、softmax、乘 V，只有新 token 的 K/V 作为增量追加写回。</p>")
    p_flow2 = E(soup, "<p>注意区分两件正交的事：<strong>KV Cache 让历史 K/V「不重算」</strong>（省计算，代价是占用 HBM）；"
        "<strong>FlashAttention（3.3）让注意力的中间矩阵「不落地」</strong>——读进 SRAM 的 KV 块在片上算完、结果只写回一次（省访存）。二者叠加，才构成 Decode 的完整数据流。</p>")
    p_page_lead = E(soup, "<p><strong>② 放得下。</strong>缓存本身该怎么在显存里摆放？传统做法给每个请求按最大生成长度连续预留一整段显存。</p>")
    p_head_lead = E(soup, "<p><strong>③ 装得少（结构侧）。</strong>先从模型结构入手，减少每个 token 要缓存的 K/V。</p>")
    p_evict_lead = E(soup, "<p><strong>③ 装得少（运行时）。</strong>结构改动需要重新训练模型；不训练模型，也能在运行时主动丢 KV——这就是逐出（eviction）。</p>")
    lead_h2o = E(soup, '<p><strong>H2O：Heavy-Hitter Oracle · 边生成边驱逐。</strong></p>')
    lead_snap = E(soup, '<p><strong>SnapKV：生成前「一次看对眼」。</strong></p>')
    cal_family = E(soup, '<div class="callout c-teal"><span class="ct">KV Cache 家族一览</span>'
        '<p>少算（缓存复用）→ 放得下（PagedAttention）→ 装得少（GQA/MQA、MLA、H2O/SnapKV；位宽量化 KIVI 见 3.4）→ 能复用（RadixAttention 见 4.2）。'
        '现代引擎里这几件事通常同时开启，且与 3.3/3.4 正交可叠加。</p></div>')

    new3 += [E(soup, '<h3>KV Cache 家族：少算、分页、压缩与逐出</h3>'), p_kv_intro,
             h4_save, p_save, fig9,
             E(soup, '<h4>KV Cache 的数据流动：HBM、SRAM 与一次写入</h4>'), p_flow1, E(soup, NEW_FIG10), p_flow2,
             h4_why, p_why2, ul_why, p_why3,
             h4_formula, fb_kv, ul_form, cal_err,
             E(soup, '<h4>放得下：PagedAttention，像操作系统管内存一样管 KV</h4>'), p_page_lead, p_page1, p_page2, fig18, paper_vllm,
             E(soup, '<h4>装得少（结构侧）：减头 GQA / MQA</h4>'), p_head_lead, p_head, fig10, paper_mqa, paper_gqa,
             E(soup, '<h4>装得少（维度侧）：MLA 多头潜在注意力</h4>'), p_mla, fig11, paper_mla,
             h4_calc, tbl_calc, note_calc,
             E(soup, '<h4>装得少（运行时）：KV 逐出 H2O 与 SnapKV</h4>'), p_evict_lead, p_evict, fig15,
             lead_h2o, p_h2o, paper_h2o, lead_snap, p_snap, paper_snap, tbl_cmp, cal_family]
    # 3.3 / 3.4
    new3 += [h3_flash, p_flash, fig14, grid_fa, paper_fa3,
             h3_quant, p_quant, tbl_quant, grid_awq, note_fp8]

    sec3.clear()
    for el in new3:
        sec3.append(el)

    # ================= 组装 ch4 =================
    sec4.find("h2").string = "并发横切面：批处理、前缀复用与 PD 分离"
    intro4 = sec4.find("p")
    intro4.string = ("前面三章跟踪的是一个请求的旅程；服务端真正面对的是成千上万个请求同时处在流水线的不同环节：有的在 Prefill、有的在 Decode、有的刚结束。"
        "这一层横切所有环节，核心命题是调度粒度、显存组织与架构解耦，目标是把 GPU 吞吐打满。本章先看静态做法的痛点与天花板（4.1），"
        "再讲机内调度手段（4.2），最后是架构级的 PD 分离（4.3）。")

    # static bullet: fragmentation now explained in 3.2
    for li in ul_static.find_all("li"):
        if "内部碎片" in li.get_text():
            li.clear()
            li.append(E(soup, "<strong>显存按最大长度预留</strong>：内部碎片严重（3.2 节 PagedAttention 展开）。"))

    p_cont = E(soup, "<p><strong>Continuous Batching</strong>（ORCA 首创）把调度单位从「整批」细化到每个 iteration（一次前向）："
        "完成的请求立刻离开、空槽插入新请求，GPU 持续满载（时序对比见上图）。</p>")
    p_ceiling = E(soup, "<p>浪费之外，再算清单卡的理论天花板，才知道优化空间有多大。</p>")
    p_metrics_lead = E(soup, "<p>谈优化前还要先定义「好」：下面这套指标贯穿本章与第 3 章。</p>")
    p_lead_gap = E(soup, '<p><strong>为什么必须分离：两阶段资源诉求相差约 140 倍。</strong></p>')
    p_lead_how = E(soup, '<p><strong>怎么做：两池拆分，KV 迁移。</strong></p>')
    p_lead_papers = E(soup, '<p><strong>三篇奠基论文，三种侧重。</strong></p>')
    p_lead_xfer = E(soup, '<p><strong>关键工程问题：KV Cache 怎么搬。</strong></p>')
    p_lead_front = E(soup, '<p><strong>前沿：PD 分离正在「产品化、标配化」。</strong></p>')
    cal_end4 = E(soup, '<div class="callout c-teal"><span class="ct">两章手段如何叠加</span>'
        '<p>机内优化（4.2）把单块 GPU 打满，架构解耦（4.3）让两类资源独立扩缩；它们与第 3 章的 KV Cache 家族、FlashAttention、量化全部可叠加——'
        '最终目标都是 SLO 达标前提下最大化 Goodput、最小化每 token 成本。</p></div>')

    new4 = [b4[0], intro4, b4[bidx(sec4, "figure", "流水线位置导航")], E(soup, NEW_MAP4)]
    new4 += [E(soup, "<h3>背景：一块 GPU 的算力是怎样被浪费的</h3>"),
             p_static, ul_static, fig17,
             p_ceiling, style_tps, flow_tps, cal_tps,
             p_metrics_lead, tbl_metrics1, tbl_metrics2]
    new4 += [E(soup, "<h3>机内优化：动态拼批、切块调度与前缀复用</h3>"),
             E(soup, "<p>4.1 的痛点对应三类机内手段：调度粒度细化到 iteration（Continuous Batching）、长 Prefill 切块交错（Chunked Prefill）、"
                 "公共前缀缓存复用（RadixAttention）；显存摆放问题（PagedAttention）已在 3.2 讲过。</p>"),
             p_cont, paper_orca,
             p_chunk, fig19, grid_sar,
             p_radix, fig20, paper_sglang]
    new4 += [E(soup, "<h3>架构级优化：PD 分离</h3>"),
             E(soup, "<p>当 Prefill 与 Decode 的资源矛盾在机内无法靠调度调和时，必然走向架构解耦。</p>"),
             p_lead_gap, p_gap1, ul_gap, p_gap2, fig21,
             p_lead_how, p_how, fig22,
             p_lead_papers, paper_dist, paper_moon, paper_split, tbl_three,
             p_lead_xfer, tbl_xfer, p_xfer,
             p_lead_front, cal_dynamo, cal_end4]

    sec4.clear()
    for el in new4:
        sec4.append(el)

    # ================= 组装 ch5 =================
    intro5 = sec5.find("p")
    intro5.string = ("Transformer 最后一层输出的隐藏向量，先经 lm_head 输出投影变成词表维度的一批浮点数（logits），采样环节要回答：这批浮点数对应哪个 token ID。"
        "本章先在 5.1 建立 logits→ID 的完整背景（采样旋钮），再讲两个优化方向：限制候选的约束解码（5.2），"
        "以及把「一次前向只出一个 ID」变成「一次前向出多个 ID」的投机解码家族（5.3）。")

    new5 = [b5[0], intro5, b5[bidx(sec5, "figure", "流水线位置导航")]]
    new5 += [E(soup, "<h3>背景：logits 怎样变成一个 ID</h3>"),
             p_knob, fig23, tbl_knob, note_knob]
    new5 += [E(soup, "<h3>优化方向一：约束解码——把非法 ID 提前屏蔽</h3>"),
             p_con, fig24, tbl_tools, grid_papers, paper_gcd, cal_remind]
    new5 += [E(soup, "<h3>优化方向二：投机解码家族——一次前向，确认多个 ID</h3>"), p_spec,
             h4_draft, fig25,
             h4_lossless, p_loss1, fb_loss, p_loss2, paper_spec1, paper_spec2,
             h4_self, p_self, paper_self,
             h4_medusa, p_medusa, fig26, paper_medusa,
             h4_eagle, p_eagle, ul_eagle, grid_eagle, paper_eagle3,
             h4_mtp, p_mtp, fig27, paper_mtp,
             h4_look, p_look, fig28, paper_look,
             h4_cmpfam, tbl_cmpfam, note_cmpfam, fig29]

    sec5.clear()
    for el in new5:
        sec5.append(el)

    # ================= 图号重排 =================
    FIG_MAP = {
        "图 12": "图 11", "图 13": "图 12", "图 18": "图 13",
        "图 10": "图 14", "图 11": "图 15", "图 15": "图 16", "图 14": "图 17",
        "图 16": "图 18", "图 17": "图 19", "图 19": "图 20", "图 20": "图 21",
        "图 21": "图 22", "图 22": "图 23", "图 23": "图 24", "图 24": "图 25",
        "图 25": "图 26", "图 26": "图 27", "图 27": "图 28", "图 28": "图 29", "图 29": "图 30",
    }
    # 顺序敏感：用占位两趟替换，避免连锁覆盖
    for sec in (sec3, sec4, sec5):
        for b in sec.find_all("b"):
            t = b.get_text()
            if t in FIG_MAP:
                b.string = "@@" + FIG_MAP[t]
        for b in sec.find_all("b"):
            if b.get_text().startswith("@@图"):
                b.string = b.get_text()[2:]

    # ================= 静态目录 =================
    toc = soup.find("div", class_="toc")
    for a in toc.find_all("a"):
        if a.get("href") == "#ch3":
            a.find(string=re.compile("推理计算")).replace_with(" 推理加速：KV Cache、FlashAttention、量化")
        if a.get("href") == "#ch4":
            a.find(string=re.compile("并发横切面")).replace_with(" 并发横切面：批处理、前缀复用与 PD 分离")

def main():
    html = open(SRC, encoding="utf-8").read()
    soup = BeautifulSoup(html, "html.parser")
    rebuild(soup)
    out = str(soup)
    open(SRC, "w", encoding="utf-8").write(out)
    print("rebuild done")

if __name__ == "__main__":
    main()
