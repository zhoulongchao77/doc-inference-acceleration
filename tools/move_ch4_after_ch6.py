# -*- coding: utf-8 -*-
"""
一次性结构调整（对应审阅意见第 5 条）：
- 04 章「并发横切面」改名「并发请求」，整章移到 Detokenizer 之后，成为第 06 章；
- 采样 05→04、Detokenizer 06→05，章节号、目录、顶部导航、交叉引用、图号全部顺延。
用法：python tools/move_ch4_after_ch6.py
"""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "大模型推理与加速技术全景.html")

src = open(SRC, encoding="utf-8").read()

def rep(s, old, new, n=1, scope=""):
    c = s.count(old)
    assert c == n, f"[{scope}] expect {n}x, got {c}x: {old[:60]!r}"
    return s.replace(old, new)

# ---------- 1. 抽出并发章 ----------
MARK4 = "<!-- ================= 第 4 章 · 并发横切面 ================= -->"
MARK5 = "<!-- ================= 第 5 章 · 采样 ================= -->"
MARK7 = "<!-- ================= 第 7 章 · 总结 ================= -->"
i4, i5, i7 = src.find(MARK4), src.find(MARK5), src.find(MARK7)
assert -1 < i4 < i5 < i7, "chapter markers not found / out of order"
block = src[i4:i5]
body = src[:i4] + src[i5:]

# ---------- 2. 并发章内部：图号引用先占位，再改章号 ----------
block = rep(block, "（图 25）", "（图 @@@25@@@）", 1, "block")
block = rep(block, "图 24 里", "图 @@@24@@@ 里", 1, "block")

ren_block = [
    ("第 4 章 · 并发横切面", "第 6 章 · 并发请求"),
    ('<section class="wrap" id="ch4">', '<section class="wrap" id="ch6">'),
    ('<span class="sec-no">04</span><h2>并发横切面：', '<span class="sec-no">06</span><h2>并发请求：'),
    ("前面三章跟踪的是一个请求的旅程", "前面五章跟踪了一个请求从输入到输出的完整旅程"),
    ("痛点与天花板（4.1），再讲机内调度手段（4.2），最后是架构级的 PD 分离（4.3）",
     "痛点与天花板（6.1），再讲机内调度手段（6.2），最后是架构级的 PD 分离（6.3）"),
    ("并发横切面 · 覆盖流水线每一个环节", "并发请求 · 覆盖流水线每一个环节"),
    ('aria-label="04 并发横切面结构地图"', 'aria-label="06 并发请求结构地图"'),
    ("04 并发横切面 · 多请求同时处在流水线的不同环节", "06 并发请求 · 多请求同时处在流水线的不同环节"),
    ("<!-- 4.1 -->", "<!-- 6.1 -->"),
    ("<!-- 4.2 -->", "<!-- 6.2 -->"),
    ("<!-- 4.3 -->", "<!-- 6.3 -->"),
    (">4.1 背景<", ">6.1 背景<"),
    (">4.2 机内优化<", ">6.2 机内优化<"),
    (">4.3 架构解耦<", ">6.3 架构解耦<"),
    ("<b>04 章结构地图</b>：4.1 背景（Static Batching 痛点、TPS 天花板、指标体系）；4.2 机内优化（Continuous Batching / Chunked Prefill / RadixAttention）；4.3 架构解耦（PD 分离）",
     "<b>06 章结构地图</b>：6.1 背景（Static Batching 痛点、TPS 天花板、指标体系）；6.2 机内优化（Continuous Batching / Chunked Prefill / RadixAttention）；6.3 架构解耦（PD 分离）"),
    ("见 4.3）。</small>", "见 6.3）。</small>"),
    ("4.1 的痛点对应三类机内手段", "6.1 的痛点对应三类机内手段"),
    ("上 PD 分离（4.3）", "上 PD 分离（6.3）"),
    ("机内优化（4.2）把单块 GPU 打满，架构解耦（4.3）让两类资源独立扩缩",
     "机内优化（6.2）把单块 GPU 打满，架构解耦（6.3）让两类资源独立扩缩"),
]
for old, new in ren_block:
    block = rep(block, old, new, 1, "block")
assert not re.search(r"(?<![0-9.])4\.[123](?![0-9])", block), "block still has 4.x refs"

# 并发章图号：19..26 -> 27..34（按出现顺序）
fig_iter = re.finditer(r"<figcaption><b>图 (\d+)</b>", block)
figs = [int(m.group(1)) for m in fig_iter]
assert figs == list(range(19, 27)), f"block figs unexpected: {figs}"
cnt = [27]
def fig_sub_block(m):
    v = cnt[0]; cnt[0] += 1
    return f"<figcaption><b>图 {v}</b>"
block = re.sub(r"<figcaption><b>图 \d+</b>", fig_sub_block, block)
block = block.replace("@@@25@@@", "32").replace("@@@24@@@", "31")

# ---------- 3. 主体：采样章 05→04 ----------
ren_sample = [
    ("第 5 章 · 采样", "第 4 章 · 采样"),
    ('<section class="wrap" id="ch5">', '<section class="wrap" id="ch4">'),
    ('<span class="sec-no">05</span><h2>采样：', '<span class="sec-no">04</span><h2>采样：'),
    ("本章先在 5.1 建立", "本章先在 4.1 建立"),
]
for old, new in ren_sample:
    body = rep(body, old, new, 1, "sample")
body = rep(body, "（5.2）", "（4.2）", 2, "sample+detok")   # 采样章 intro + Detokenizer 章引用
body = rep(body, "（5.3）", "（4.3）", 2, "sample+ch7")     # 采样章 intro + 第 7 章反模式

# ---------- 4. 主体：Detokenizer 章 06→05 ----------
ren_detok = [
    ("第 6 章 · Detokenizer", "第 5 章 · Detokenizer"),
    ('<section class="wrap" id="ch6">', '<section class="wrap" id="ch5">'),
    ('<span class="sec-no">06</span><h2>Detokenizer：', '<span class="sec-no">05</span><h2>Detokenizer：'),
]
for old, new in ren_detok:
    body = rep(body, old, new, 1, "detok")

# ---------- 5. 主体：指向并发章的交叉引用 4.x→6.x、第 4 章→第 6 章 ----------
ren_refs = [
    ("为 RadixAttention 的前缀复用铺路（第 4 章）", "为 RadixAttention 的前缀复用铺路（第 6 章）"),
    ("前缀复用（4.2 RadixAttention）", "前缀复用（6.2 RadixAttention）"),
    ("能复用见 4.2", "能复用见 6.2"),
    ("（4.1 会按此修正口径）", "（6.1 会按此修正口径）"),
    ("让一次搬运服务更多 token，第 4 章）", "让一次搬运服务更多 token，第 6 章）"),
    ("RadixAttention 前缀树（见 4.2）", "RadixAttention 前缀树（见 6.2）"),
    ("能复用（RadixAttention 见 4.2）", "能复用（RadixAttention 见 6.2）"),
    ("这正是 4.1 实测", "这正是 6.1 实测"),
    ("Goodput 反而下降（4.1）", "Goodput 反而下降（6.1）"),
    ("还白占显存（4.2）", "还白占显存（6.2）"),
    ("避免在采样阶段被拆碎（第 5 章）", "避免在采样阶段被拆碎（第 4 章）"),
    ("是第 5 章要面对的对象", "是第 4 章要面对的对象"),
    ("见第 05 章。", "见第 04 章。"),
]
for old, new in ren_refs:
    body = rep(body, old, new, 1, "refs")
assert not re.search(r"(?<![0-9.])5\.[123](?![0-9])", body), "body still has 5.x refs"
leftover = [m.group(0) for m in re.finditer(r"(?<![0-9.])4\.[123](?![0-9])", body)]
assert all(v in ("4.1", "4.2", "4.3") for v in leftover), leftover
# 主体里剩下的 4.x 应只属于采样章新编号
for m in re.finditer(r"(?<![0-9.])4\.[123](?![0-9])", body):
    s = max(0, m.start()-20)
    print("  body 4.x ok:", body[s:m.end()+20].replace("\n", " "))

# ---------- 6. 顶部导航 + 头部静态目录 ----------
body = rep(body,
    '<a href="#ch3">推理加速</a><a href="#ch4">并发调度</a><a href="#ch5">采样</a>\n<a href="#ch6">输出</a><a href="#ch7">总结</a>',
    '<a href="#ch3">推理加速</a><a href="#ch4">采样</a><a href="#ch5">输出</a>\n<a href="#ch6">并发请求</a><a href="#ch7">总结</a>',
    1, "nav")
body = rep(body,
    '<a href="#ch4"><span class="n">04</span> 并发横切面：批处理、前缀复用与 PD 分离</a>\n<a href="#ch5"><span class="n">05</span> 采样：从 logits 到下一个 ID</a>\n<a href="#ch6"><span class="n">06</span> Detokenizer：ID 拼回文本，流式输出</a>',
    '<a href="#ch4"><span class="n">04</span> 采样：从 logits 到下一个 ID</a>\n<a href="#ch5"><span class="n">05</span> Detokenizer：ID 拼回文本，流式输出</a>\n<a href="#ch6"><span class="n">06</span> 并发请求：批处理、前缀复用与 PD 分离</a>',
    1, "toc")

# ---------- 7. 主体图号整体重排 1..26 ----------
cnt2 = [1]
def fig_sub_body(m):
    v = cnt2[0]; cnt2[0] += 1
    return f"<figcaption><b>图 {v}</b>"
body = re.sub(r"<figcaption><b>图 \d+</b>", fig_sub_body, body)
assert cnt2[0] == 27, f"body fig count {cnt2[0]-1} != 26"

# ---------- 8. 并发章插到 Detokenizer 之后、总结之前 ----------
i7b = body.find(MARK7)
assert i7b != -1
out = body[:i7b] + block + body[i7b:]

open(SRC, "w", encoding="utf-8").write(out)
print("\nOK: chapter moved, renumbered, refs fixed. total figs =", cnt2[0]-1 + 8)
