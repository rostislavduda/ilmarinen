"""Regenerate STUDY_LOG.md -- a single chronological compilation of every study/test write-up.

The log is a reference-for-posterity document that inlines the FULL TEXT of every write-up in
tests/*.md, in chronological order (by file mtime, which tracks when each write-up was authored),
under a per-entry header carrying its date, file name, and title, with a navigable table of contents
at the top. Run this whenever new .md write-ups are added so the log stays current:

    python studies/regenerate_study_log.py

The script is deterministic and idempotent: running it again with no new files reproduces the same
STUDY_LOG.md. It reads tests/*.md and writes STUDY_LOG.md at the package root.
"""

import datetime
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TESTS = os.path.join(ROOT, "tests")
OUT = os.path.join(ROOT, "STUDY_LOG.md")


def first_title(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip()
    return os.path.basename(path)


def read_body(path):
    """Full text of the write-up, with a leading H1 title line stripped (it is re-emitted in the
    entry header) and demoted headings so the compiled doc has a clean single hierarchy."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.splitlines()
    # drop the first H1 (the title) if present; keep everything else verbatim
    out, dropped_title = [], False
    for ln in lines:
        if not dropped_title and ln.strip().startswith("# "):
            dropped_title = True
            continue
        out.append(ln)
    body = "\n".join(out).strip("\n")

    # demote every ATX heading by two levels so per-file "##"/"###" sit under the entry's "###"
    # (entries are H3; a file's top-level "##" becomes "#####"). This keeps the global TOC clean
    # while preserving the write-up's internal structure.
    def demote(m):
        hashes = m.group(1)
        return "#" * min(len(hashes) + 2, 6) + " "

    body = re.sub(r"^(#{1,4})\s", demote, body, flags=re.MULTILINE)
    return body


def slugify(name, n):
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{n:03d}-{base}"


def build_rows():
    rows = []
    for p in glob.glob(os.path.join(TESTS, "*.md")):
        dt = datetime.datetime.fromtimestamp(os.path.getmtime(p))
        rows.append((dt, os.path.basename(p), first_title(p), p))
    rows.sort(key=lambda r: r[0])
    return rows


def render(rows):
    lines = []
    lines.append("# ilmarinen study log -- full compilation")
    lines.append("")
    lines.append("The complete text of every study/test write-up (`tests/*.md`) produced over the life of")
    lines.append("the project, inlined in chronological order (by authoring time) into a single document")
    lines.append("for posterity. Each entry below reproduces one write-up in full, under a header giving its")
    lines.append("date, file name, and title. A table of contents follows; the full entries come after it.")
    lines.append("")
    lines.append("**Maintenance.** This file is generated. After adding or editing a `tests/*.md` write-up,")
    lines.append("regenerate it with `python studies/regenerate_study_log.py` (deterministic and idempotent).")
    lines.append("Do not hand-edit -- edits are overwritten on regeneration.")
    lines.append("")
    span0 = rows[0][0].strftime("%Y-%m-%d") if rows else "-"
    span1 = rows[-1][0].strftime("%Y-%m-%d") if rows else "-"
    lines.append(
        f"_Total write-ups: {len(rows)} &nbsp;|&nbsp; span: {span0} to {span1} "
        f"&nbsp;|&nbsp; generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # table of contents
    lines.append("## Table of contents")
    lines.append("")
    cur_day = None
    for i, (dt, name, title, _p) in enumerate(rows, 1):
        day = dt.strftime("%Y-%m-%d")
        if day != cur_day:
            if cur_day is not None:
                lines.append("")
            cur_day = day
            lines.append(f"**{day}**")
            lines.append("")
        anchor = slugify(name, i)
        lines.append(f"{i}. [{name} — {title}](#{anchor})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # full entries
    for i, (dt, name, title, p) in enumerate(rows, 1):
        anchor = slugify(name, i)
        # an explicit anchor span so the TOC links resolve regardless of how headings render
        lines.append(f'<a id="{anchor}"></a>')
        lines.append("")
        lines.append(f"### {i}. {title}")
        lines.append("")
        lines.append(f"`{name}` · authored {dt.strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append(read_body(p))
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    if not os.path.isdir(TESTS):
        print(
            f"no tests/ directory at {TESTS} -- nothing to index (STUDY_LOG.md left unchanged). "
            f"The write-ups live in tests/, which is excluded from the distribution tarball; "
            f"run this from the working tree where tests/*.md are present."
        )
        return
    rows = build_rows()
    if not rows:
        print("tests/ has no .md write-ups -- STUDY_LOG.md left unchanged.")
        return
    text = render(rows)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    kb = os.path.getsize(OUT) / 1024
    print(
        f"wrote {OUT}: {len(rows)} entries inlined in full, "
        f"{rows[0][0].strftime('%Y-%m-%d')}..{rows[-1][0].strftime('%Y-%m-%d')}, {kb:.0f} KB"
    )


if __name__ == "__main__":
    main()
