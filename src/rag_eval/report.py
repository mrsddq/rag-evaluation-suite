from __future__ import annotations

import html


def render_html(report: dict[str, object]) -> str:
    summary = report["summary"]
    cards = "".join(
        f'<section><span>{html.escape(name.replace("_", " ").title())}</span><strong>{value:.3f}</strong></section>'
        for name, value in summary.items()
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in item.values()) + "</tr>"
        for item in report["cases"]
    )
    headers = "".join(f"<th>{html.escape(name.replace('_', ' ').title())}</th>" for name in report["cases"][0])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>RAG evaluation report</title><style>
body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;color:#172033}}
h1{{margin-bottom:8px}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:28px 0}}
section{{padding:18px;border:1px solid #dfe5ef;border-radius:12px;background:#f8fafc}} section span{{display:block;font-size:12px;color:#536176}} section strong{{font-size:28px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{padding:10px;border-bottom:1px solid #dfe5ef;text-align:left}} th{{background:#172033;color:white}}
</style></head><body><h1>RAG evaluation</h1><p>{report['count']} cases evaluated.</p>
<div class="cards">{cards}</div><table><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></body></html>"""

