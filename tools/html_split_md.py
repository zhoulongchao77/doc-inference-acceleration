# -*- coding: utf-8 -*-
"""
将《大模型推理与加速技术全景.html》按章节拆分为 Markdown。
- 保留：标题层级、加粗/行内代码/链接、公式（$...$ / $$...$$）、表格、列表、图注
- SVG 矢量图不适合喂给 AI，统一转写为「图号 + aria-label + 图注」文字
- 跳过：CSS/JS、目录块、流水线位置导航小图（minimap）
- 输出：markdown/00..07 分章文件、README 索引、合并全文
用法：python tools/html_split_md.py
"""
import os, re
from bs4 import BeautifulSoup, NavigableString, Tag, Comment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "大模型推理与加速技术全景.html")
OUT = os.path.join(ROOT, "markdown")

SKIP_CLASSES = {"lead", "hero-stats", "meta-row", "kicker"}

def clean_ws(s):
    return re.sub(r"\s+", " ", s).strip()

def inline(node):
    """行内节点 -> Markdown 文本"""
    parts = []
    for ch in node.children:
        if isinstance(ch, Comment):
            continue
        if isinstance(ch, NavigableString):
            parts.append(str(ch))
        elif not isinstance(ch, Tag):
            continue
        elif ch.name in ("style", "script", "svg"):
            continue
        elif ch.name in ("strong", "b"):
            t = inline(ch)
            if t: parts.append("**" + t + "**")
        elif ch.name in ("em", "i"):
            t = inline(ch)
            if t: parts.append("*" + t + "*")
        elif ch.name == "code":
            parts.append("`" + clean_ws(ch.get_text()) + "`")
        elif ch.name == "a":
            t = inline(ch); href = ch.get("href", "")
            if t and href.startswith("http"):
                parts.append("[" + t + "](" + href + ")")
            elif t:
                parts.append(t)
        elif ch.name == "br":
            parts.append(" ")
        else:
            parts.append(inline(ch))
    return clean_ws("".join(parts))

def li_content(li):
    frag = []
    for ch in li.children:
        if isinstance(ch, Tag) and ch.name in ("ul", "ol"):
            continue
        if isinstance(ch, Comment):
            continue
        frag.append(inline(ch) if isinstance(ch, Tag) else str(ch))
    return clean_ws(" ".join(frag))

def render_list(lst, depth=0):
    lines = []
    is_ol = lst.name == "ol"
    for i, li in enumerate(lst.find_all("li", recursive=False), 1):
        mark = (str(i) + ".") if is_ol else "-"
        lines.append("  " * depth + mark + " " + li_content(li))
        for sub in li.find_all(["ul", "ol"], recursive=False):
            lines.extend(render_list(sub, depth + 1))
    return lines

def render_table(tbl):
    def row_cells(tr):
        return [inline(c) for c in tr.find_all(["th", "td"])]
    head, body = [], []
    thead = tbl.find("thead")
    all_tr = tbl.find_all("tr")
    if thead and thead.find("tr"):
        head = row_cells(thead.find("tr"))
        for tr in all_tr:
            if not tr.find_parent("thead"):
                body.append(row_cells(tr))
    elif all_tr:
        head, body = row_cells(all_tr[0]), [row_cells(t) for t in all_tr[1:]]
    if not head:
        return []
    ncol = max([len(head)] + [len(r) for r in body])
    head += [""] * (ncol - len(head))
    body = [r + [""] * (ncol - len(r)) for r in body]
    def esc(r):
        return [c.replace("|", "\\|").replace("\n", " ") for c in r[:ncol]]
    out = ["| " + " | ".join(esc(head)) + " |",
           "| " + " | ".join(["---"] * ncol) + " |"]
    for r in body:
        out.append("| " + " | ".join(esc(r)) + " |")
    return out

def render_figure(fig):
    lines = [""]
    svg = fig.find("svg")
    aria = clean_ws(svg.get("aria-label", "")) if svg else ""
    cap = fig.find("figcaption")
    num, captext = "", ""
    if cap:
        b = cap.find("b")
        num = inline(b) if b else ""
        sp = cap.find("span")
        captext = inline(sp) if sp else inline(cap)
    if num or aria:
        lines.append("> **" + (num or "图") + "**" + ((" · " + aria) if aria else ""))
    if captext and captext != num:
        lines.append("> " + captext)
    if not num and not aria and not captext:
        return []
    lines.append("")
    return lines

def render_blocks(node, h3_prefix, skip_hero=False, h3n=0):
    """递归渲染块级元素，返回 (lines, h3n)，h3n 在嵌套调用间连续编号。"""
    lines = []
    for ch in node.children:
        if isinstance(ch, Comment):
            continue
        if isinstance(ch, NavigableString):
            t = clean_ws(str(ch))
            if t: lines.append(t)
            continue
        if not isinstance(ch, Tag):
            continue
        cls = ch.get("class", [])
        name = ch.name
        if name in ("nav", "aside", "script", "style", "noscript"):
            continue
        if name == "header" and "hero" in cls and skip_hero:
            continue
        if set(cls) & SKIP_CLASSES:
            continue
        if name == "span" and set(cls) & {"sec-no", "sub", "n"}:
            continue
        if name == "figure" and "minimap" in cls:
            continue
        if name == "div" and "toc" in cls:
            continue
        if name in ("h1", "h2"):
            continue
        if name == "h3":
            h3n += 1
            lines += ["", "## " + h3_prefix + str(h3n) + " " + inline(ch), ""]
            continue
        if name == "h4":
            lines += ["", "### " + inline(ch), ""]
            continue
        if name == "p":
            t = inline(ch)
            if t: lines += [t, ""]
            continue
        if name in ("ul", "ol"):
            lines += render_list(ch)
            lines.append("")
            continue
        if name == "figure":
            lines += render_figure(ch)
            continue
        if name == "table":
            lines += render_table(ch)
            lines.append("")
            continue
        if name == "blockquote":
            body, h3n = render_blocks(ch, h3_prefix, skip_hero, h3n)
            lines += ["> " + b if b.strip() else ">" for b in body]
            lines.append("")
            continue
        if name == "small" or (name == "div" and "note" in cls):
            t = inline(ch)
            if t: lines += ["*" + t + "*", ""]
            continue
        if name == "div" and "formula-box" in cls:
            t = clean_ws(ch.get_text())
            if t: lines += [t, ""]
            continue
        if name == "div" and "callout" in cls:
            ct = ch.find(class_="ct")
            label = inline(ct) if ct else ""
            inner, h3n = render_blocks(ch, h3_prefix, skip_hero, h3n)
            inner = [b for b in inner if b.strip() and b.strip() not in ("**" + label + "**", label)]
            if label or inner:
                lines.append("")
                if label: lines.append("> **" + label + "**")
                for b in inner: lines.append("> " + b)
                lines.append("")
            continue
        if name == "div" and "paper" in cls:
            pid = ch.find(class_="pid")
            ptitle = ch.find(class_="ptitle")
            pinfo = ch.find(class_="pinfo")
            a = ch.find("a", href=True)
            seg = "- " + ("**" + inline(pid) + "** " if pid else "")
            if ptitle:
                t = inline(ptitle)
                seg += ("[" + t + "](" + a["href"] + ")") if a else t
            if pinfo:
                seg += "（" + inline(pinfo) + "）"
            if a and ptitle is None:
                seg += " [" + inline(a) + "](" + a["href"] + ")"
            lines.append(seg)
            continue
        if name == "div" and "chips" in cls:
            chips = [inline(c) for c in ch.find_all(class_="chip")]
            chips = [c for c in chips if c]
            if chips: lines += ["**标签**：" + "；".join(chips), ""]
            continue
        if name == "div" and "fstep" in cls:
            k, v, sm = ch.find(class_="k"), ch.find(class_="v"), ch.find("small")
            if k: lines.append("- **" + inline(k) + "**" + (("：" + inline(v)) if v else ""))
            if sm: lines.append("  *" + inline(sm) + "*")
            continue
        if name in ("div", "section", "header", "footer"):
            more, h3n = render_blocks(ch, h3_prefix, skip_hero, h3n)
            lines += more
            continue
        t = inline(ch)
        if t: lines += [t, ""]
    return lines, h3n

def chapter_meta(sec, si):
    no = sec.find(class_="sec-no")
    h2 = sec.find("h2")
    if si == 0:
        return "00", "开篇：全景地图与推理加速为什么值得做"
    return (inline(no) if no else "%02d" % si), (inline(h2) if h2 else "")

def main():
    html = open(SRC, encoding="utf-8").read()
    soup = BeautifulSoup(html, "html.parser")
    secs = soup.find_all("section", id=re.compile(r"^ch"))
    os.makedirs(OUT, exist_ok=True)
    chapters, combined = [], ['# 大模型推理与加速技术全景（全文 Markdown）', '',
                '> 由 tools/html_split_md.py 从同名 HTML 自动生成；图为文字转写。', '']
    for si, sec in enumerate(secs):
        no, title = chapter_meta(sec, si)
        prefix = str(si) + "."
        body, h3n = [], 0
        if si == 0:
            hero = sec.find("header", class_="hero")
            lead = hero.find(class_="lead") if hero else None
            if lead: body += [inline(lead), ""]
            stats = hero.find(class_="hero-stats") if hero else None
            if stats:
                for st in stats.find_all(class_="hstat"):
                    n, l = st.find(class_="num"), st.find(class_="lab")
                    if n and l: body.append("- **" + inline(n) + "** " + inline(l))
                body.append("")
            if hero:
                more, h3n = render_blocks(hero, prefix, skip_hero=False, h3n=0)
                body += more
        more, h3n = render_blocks(sec, prefix, skip_hero=(si == 0), h3n=h3n)
        body += more
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()
        h3_titles = re.findall(r"^## " + re.escape(prefix) + r"\d+ (.+)$", text, re.M)
        fname = "%02d-%s.md" % (si, re.sub(r'[\\/:*?"<>|]', "", title.split("：")[0])[:12])
        header = "# %s %s\n\n" % (no, title)
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
            f.write(header + text + "\n")
        chapters.append((fname, no, title, h3_titles))
        combined += [header.rstrip(), "", text, "", "---", ""]

    rd = ["# 《大模型推理与加速技术全景》分章 Markdown 索引", "",
          "给 AI / 自己阅读用：**需要哪章就只打开哪个文件**，不必把整个 HTML 塞进上下文。", "",
          "| 文件 | 章节 | 小节 |", "| --- | --- | --- |"]
    for fname, no, title, subs in chapters:
        rd.append("| [%s](%s) | %s %s | %s |" % (
            fname, fname, no, title, "；".join(subs) if subs else "—"))
    rd += ["", "## 使用建议", "",
           "1. **局部修改/提问**：只把对应章节的 .md 发给 AI（单章约 2–16 KB）。",
           "2. **整体校对/串讲**：用 `大模型推理与加速技术全景-全文.md`（约 60 KB）。",
           "3. **看图**：图中信息已转写为「图号 · 主题 + 图注」；需要原图时打开同名 HTML。",
           "4. HTML 改动后重新生成：`python tools/html_split_md.py`。", ""]
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rd))
    with open(os.path.join(OUT, "大模型推理与加速技术全景-全文.md"), "w", encoding="utf-8") as f:
        f.write(re.sub(r"\n{3,}", "\n\n", "\n".join(combined)))
    for c in chapters:
        print(c[0], "|", len(c[3]), "个小节")
    print("README.md / 全文.md done")

if __name__ == "__main__":
    main()
