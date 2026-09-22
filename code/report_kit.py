"""Shared page furniture for the generated exercise reports.

The stylesheet and the two HTML helpers are identical across exercises, so
they live here rather than being copied into each builder.
"""


CSS = """
:root {
  color-scheme: light;
  --bg: #f6f5f1;
  --surface: #fcfcfb;
  --surface-2: #f1f0ec;
  --line: #e3e2dd;
  --line-strong: #d3d2cc;
  --ink: #16150f;
  --ink-soft: #52514e;
  --ink-muted: #86857f;
  --accent: #2a78d6;
  --accent-soft: #eaf2fd;
  --warn: #eda100;
  --mono: ui-monospace, "SFMono-Regular", "Cascadia Mono", Menlo, Consolas, monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16.5px;
  line-height: 1.68;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 940px; margin: 0 auto; padding: 0 22px 96px; }

header.hero {
  border-bottom: 1px solid var(--line);
  background: var(--surface);
  padding: 52px 0 34px;
  margin-bottom: 38px;
}
header.hero .wrap { padding-bottom: 0; }
.eyebrow {
  font-size: 12.5px; letter-spacing: .12em; text-transform: uppercase;
  color: var(--accent); font-weight: 650; margin: 0 0 12px;
}
h1 { font-size: 2.35rem; line-height: 1.16; margin: 0 0 14px; letter-spacing: -.02em; font-weight: 700; }
.lede { font-size: 1.08rem; color: var(--ink-soft); margin: 0 0 22px; max-width: 68ch; }
.meta { display: flex; flex-wrap: wrap; gap: 8px 10px; font-size: 13.5px; }
.meta span, .meta a {
  background: var(--surface-2); border: 1px solid var(--line);
  padding: 4px 11px; border-radius: 999px; color: var(--ink-soft); text-decoration: none;
}
.meta a { color: var(--accent); border-color: #cfe0f7; background: var(--accent-soft); font-weight: 550; }
.meta a:hover { background: #dde9fb; }

nav.toc {
  background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 20px 24px; margin-bottom: 44px;
}
nav.toc h2 { font-size: 12.5px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ink-muted); margin: 0 0 12px; font-weight: 650; }
nav.toc ol { margin: 0; padding-left: 20px; columns: 2; column-gap: 34px; font-size: 14.5px; }
@media (max-width: 620px) { nav.toc ol { columns: 1; } }
nav.toc li { margin: 3px 0; break-inside: avoid; }
nav.toc a { color: var(--ink-soft); text-decoration: none; }
nav.toc a:hover { color: var(--accent); text-decoration: underline; }

section { margin-bottom: 60px; scroll-margin-top: 20px; }
h2 {
  font-size: 1.62rem; margin: 52px 0 6px; letter-spacing: -.015em; font-weight: 700;
  padding-bottom: 10px; border-bottom: 2px solid var(--ink); line-height: 1.25;
}
h2 .num { color: var(--accent); font-variant-numeric: tabular-nums; margin-right: 10px; }
h3 { font-size: 1.14rem; margin: 36px 0 8px; font-weight: 650; letter-spacing: -.01em; }
h4 { font-size: .98rem; margin: 26px 0 6px; font-weight: 650; color: var(--ink-soft); }
p { margin: 0 0 15px; }
ul, ol { margin: 0 0 15px; padding-left: 24px; }
li { margin: 5px 0; }
a { color: var(--accent); }
strong { font-weight: 650; }
code {
  font-family: var(--mono); font-size: .875em; background: var(--surface-2);
  border: 1px solid var(--line); border-radius: 5px; padding: 1px 5px;
}
pre {
  background: #1c1b18; color: #eceada; border-radius: 10px; padding: 16px 18px;
  overflow-x: auto; font-family: var(--mono); font-size: 13px; line-height: 1.6;
  margin: 0 0 18px; border: 1px solid #2c2b26;
}
pre code { background: none; border: none; padding: 0; color: inherit; font-size: 13px; }

figure {
  margin: 26px 0 30px; background: var(--surface); border: 1px solid var(--line);
  border-radius: 12px; padding: 14px 14px 4px; overflow: hidden;
}
figure img { width: 100%; height: auto; display: block; border-radius: 6px; }
figcaption {
  font-size: 13.5px; color: var(--ink-soft); padding: 12px 4px 12px;
  border-top: 1px solid var(--line); margin-top: 12px; line-height: 1.55;
}
figcaption b { color: var(--ink); font-weight: 650; }

.tablewrap { overflow-x: auto; margin: 0 0 22px; border: 1px solid var(--line);
  border-radius: 10px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: 14.2px; }
caption { caption-side: top; text-align: left; font-size: 13.5px; color: var(--ink-soft);
  padding: 12px 16px 10px; border-bottom: 1px solid var(--line); }
caption b { color: var(--ink); font-weight: 650; }
th, td { padding: 9px 16px; text-align: left; border-bottom: 1px solid var(--line); }
thead th { font-size: 12px; letter-spacing: .05em; text-transform: uppercase;
  color: var(--ink-muted); font-weight: 650; background: var(--surface-2); }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; font-family: var(--mono); font-size: 13.2px; }
tr.hl td { background: var(--accent-soft); font-weight: 600; }
tr.hl td.num { color: #1c5cab; }

.callout {
  border-left: 3px solid var(--accent); background: var(--surface);
  border-radius: 0 10px 10px 0; padding: 15px 20px; margin: 22px 0;
  border-top: 1px solid var(--line); border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
.callout p:last-child { margin-bottom: 0; }
.callout .tag { display: block; font-size: 11.5px; letter-spacing: .09em;
  text-transform: uppercase; color: var(--accent); font-weight: 650; margin-bottom: 5px; }
.callout.warn { border-left-color: var(--warn); }
.callout.warn .tag { color: #a97400; }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin: 24px 0 28px; }
.kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 15px 17px; }
.kpi .v { font-size: 1.5rem; font-weight: 700; letter-spacing: -.02em;
  font-variant-numeric: tabular-nums; line-height: 1.2; }
.kpi .k { font-size: 12.5px; color: var(--ink-muted); margin-top: 3px; line-height: 1.4; }

footer { border-top: 1px solid var(--line); padding-top: 24px; margin-top: 40px;
  font-size: 13.5px; color: var(--ink-muted); }
footer p { margin: 0 0 8px; }
"""


def table(caption, headers, rows, aligns=None, hl=None):
    aligns = aligns or ["left"] * len(headers)
    hl = hl or set()
    th = "".join(
        f'<th class="num">{h}</th>' if a == "num" else f"<th>{h}</th>"
        for h, a in zip(headers, aligns))
    body = []
    for i, r in enumerate(rows):
        cls = ' class="hl"' if i in hl else ""
        tds = "".join(
            f'<td class="num">{c}</td>' if a == "num" else f"<td>{c}</td>"
            for c, a in zip(r, aligns))
        body.append(f"<tr{cls}>{tds}</tr>")
    return (f'<div class="tablewrap"><table><caption>{caption}</caption>'
            f"<thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def figure(src, cap):
    return f'<figure><img src="{src}" alt=""><figcaption>{cap}</figcaption></figure>'
