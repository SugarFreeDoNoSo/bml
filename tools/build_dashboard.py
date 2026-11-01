"""Build a dependency-free GitHub Pages dashboard from OpenSpec task lists."""

import argparse
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/SugarFreeDoNoSo/bml"


def build(output, ref="master"):
    cards = []
    totals = {"done": 0, "pending": 0, "deferred": 0}
    for path in sorted((ROOT / "openspec/changes").glob("*/tasks.md")):
        tasks = []
        counts = dict.fromkeys(totals, 0)
        section = ""
        for line in path.read_text().splitlines():
            if line.startswith("## "):
                section = line[3:]
            match = re.match(r"\s*- \[([ xX])\]\s+(.+)", line)
            if not match:
                continue
            checked, label = match.groups()
            state = "done" if checked.lower() == "x" else (
                "deferred" if "deferred" in label.lower() else "pending"
            )
            counts[state] += 1
            title = {"done": "Completada", "pending": "Pendiente", "deferred": "Diferida"}[state]
            tasks.append(f'<li data-state="{state}"><span class="badge {state}">{title}</span>'
                         f'<span>{escape(label)}<small>{escape(section)}</small></span></li>')
        total = sum(counts.values())
        for key in totals:
            totals[key] += counts[key]
        percent = round(100 * counts["done"] / total) if total else 0
        name = path.parent.name
        link = f"{REPO}/blob/{ref}/{path.relative_to(ROOT).as_posix()}"
        cards.append(f'''<article class="card" data-name="{escape(name)}">
<div class="card-heading"><h2>{escape(name.replace('-', ' '))}</h2><strong>{percent}%</strong></div>
<progress max="{total or 1}" value="{counts['done']}" aria-label="Avance de {escape(name)}"></progress>
<p class="muted">{counts['done']} de {total} completadas · {counts['pending']} pendientes · {counts['deferred']} diferidas</p>
<a href="{escape(link)}">Ver plan en GitHub ↗</a>
<details><summary>Consultar tareas ({total})</summary><ul>{''.join(tasks)}</ul><p class="empty" hidden>Sin tareas para este filtro.</p></details></article>''')
    template = (ROOT / "dashboard/index.html").read_text()
    substitutions = {"CARDS": "\n".join(cards), "DONE": str(totals["done"]),
                     "PENDING": str(totals["pending"]), "DEFERRED": str(totals["deferred"]),
                     "PLANS": str(len(cards)), "UPDATED": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                     "REF": escape(ref)}
    for key, value in substitutions.items():
        template = template.replace("{{" + key + "}}", value)
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(template)
    for name in ("style.css", "app.js"):
        shutil.copyfile(ROOT / "dashboard" / name, output / name)
    (output / ".nojekyll").write_text("")
    print(f"Built {len(cards)} plans in {output}: {totals}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "target/dashboard")
    parser.add_argument("--ref", default="master")
    args = parser.parse_args()
    build(args.output, args.ref)
