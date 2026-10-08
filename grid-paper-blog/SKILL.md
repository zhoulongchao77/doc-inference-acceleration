---
name: grid-paper-blog
description: 生成"稿纸网格 · 工程杂志风"单文件 HTML 技术博客/分享长文。当用户要求写技术博客、分享材料、技术长文、白皮书式文章，或提到"稿纸网格""工程杂志风""用之前的模板/那种风格写博客"时使用。产出为单个自包含 HTML 文件（浅灰网格纸底、藏青+橙配色、Noto Sans SC + JetBrains Mono、左侧自动目录、统计数字卡、SVG 示意图、四色 callout、论文 chip、KaTeX 公式）。
---

# Grid-Paper Blog 写作流程

## 流程

1. 拿到主题/大纲后，将 `assets/template.html` 复制为工作区中的目标文件（如 `<主题>.html`），**不要改动其中的 CSS 与 JS**——目录、章节编号（N.M）、阅读进度条、图片缩放均由 JS 自动生成。
2. 替换所有 `{{占位符}}`，按 `<section class="wrap" id="chN">` + `<div class="sec-head">` 结构增删章节；h3 小节、`<h4 class="l3">` 三级小节会自动进左侧目录。
3. 按 `references/style-guide.md` 的写作规范填内容（先全景地图、每个技术点遵循「是什么→为什么慢→怎么优化→论文出处」、数据带出处）。
4. 示意图用内嵌原创 SVG（圆角矩形 + 箭头 marker + 等宽字体标注，配色只用调色板内的颜色），包在 `<figure class="fig">` 中，figcaption 以 `<b>图N</b>` 开头。
5. 完成后在浏览器打开自检：进度条、目录高亮、图片点击放大、公式渲染是否正常。

## 组件速查

| 用途 | 写法 |
| --- | --- |
| 提示框 | `<div class="callout c-key/c-teal/c-blue/c-gold"><span class="ct">标题</span><p>…</p></div>` |
| 论文 chip | `<div class="paper">`（变体 `.spec` 橙 / `.sys` 蓝 / `.gold` 金），含 `.pid` / `.ptitle` / `.pinfo` |
| 标签 | `<span class="tag t-orange/t-teal/t-blue/t-gold/t-gray">` |
| 表格 | `<div class="tbl-scroll"><table>…` |
| 勾选清单 | `<ul class="check">` |
| 公式 | `<div class="formula-box">$$…$$</div>`，行内 `$…$` |
| 双栏 | `<div class="grid2">` |

## 注意

- 占位符 `{{...}}` 必须全部替换，交付前全文搜索确认无残留。
- 章节锚点 id 用 `ch0, ch1, ch2…`（正则 `section[id^="ch"]` 依赖此前缀）。
- 统计卡数字必须能在正文中找到对应出处。
