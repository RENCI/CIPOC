"""Run every pending manual compile listed in config/variable_groups.json.

The grouping file is the target list: each note-extracted variable carries a
``rule_source`` naming the manual and chapter that governs it. This script turns
every entry still marked ``status: pending`` into a compile_manual run, so the
work queue is derived from the grouping rather than kept in a parallel list.

Each compile is a serial LLM tagging pass over one chapter's sections, so a full
run is long and costs tokens. Accordingly:

- nothing runs without ``--run`` (bare invocation prints the plan);
- a chapter whose output file already exists is skipped, making the run
  resumable after an interruption (use ``--force`` to recompile anyway);
- a failing chapter does not abort the others; failures are collected and
  summarized at the end, and the exit code is non-zero if any failed;
- on success the variable's ``status`` flips ``pending`` -> ``compiled`` in the
  grouping file, so a re-run only picks up what is genuinely left.

Every compile still writes a ``<site_group>.review.txt`` that must be eyeballed
before the units are trusted; this script reports fidelity and quarantine counts
but does not judge them.

    python -m scripts.rule_compilation.compile_pending              # show the plan
    python -m scripts.rule_compilation.compile_pending --run        # compile all pending
    python -m scripts.rule_compilation.compile_pending --run --item 410
    python -m scripts.rule_compilation.compile_pending --run --dry-run   # segment only, no LLM
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

GROUPS_PATH = Path("config/variable_groups.json")
MANIFEST_PATH = Path("documents/rules/manifest.json")
RULES_DIR = Path("documents/rules")

# SPCSM item chapters are level-2 headings. Pinning the level stops a deeper
# heading that quotes the chapter title from matching first (see select_subtree).
CHAPTER_ROOT_LEVEL = {"spcsm_2024": 2}


def slugify(text: str) -> str:
    """Chapter title -> output file stem, e.g. 'Mets at Diagnosis--Bone' -> 'mets_bone'."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    return re.sub(r"^mets_at_diagnosis_", "mets_", slug)


def pending_targets(groups: dict, manual: str, item: int | None) -> list[dict]:
    """Collect variables whose rule_source is an uncompiled chapter of ``manual``.

    Deduplicated by chapter: two items documented by one chapter compile once.
    """
    out: list[dict] = []
    by_chapter: dict[str, dict] = {}

    def visit(v: dict, group_id: str) -> None:
        rs = v.get("rule_source") or {}
        if rs.get("manual") != manual or rs.get("status") != "pending":
            return
        if item is not None and v["item_id"] != item:
            return
        chapter = rs.get("chapter")
        if not chapter:
            return
        if chapter in by_chapter:
            by_chapter[chapter]["item_ids"].append(v["item_id"])
            return
        target = {
            "item_ids": [v["item_id"]],
            "name": v["name"],
            "group": group_id,
            "chapter": chapter,
            "line": rs.get("chapter_line"),
            "site_group": slugify(chapter),
        }
        by_chapter[chapter] = target
        out.append(target)

    for grp in groups["groups"]:
        for v in grp.get("variables", []):
            visit(v, grp["id"])
        for sub in grp.get("subgroups", []):
            for v in sub.get("variables", []):
                visit(v, grp["id"])
    return out


def build_command(target: dict, manual: str, source: str, extra: list[str]) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.rule_compilation.compile_manual",
        "--manual", manual,
        "--site-group", target["site_group"],
        "--root-heading", target["chapter"],
        "--source", source,
    ]
    root_level = CHAPTER_ROOT_LEVEL.get(manual)
    if root_level is not None:
        cmd += ["--root-level", str(root_level)]
    return cmd + extra


def mark_compiled(manual: str, item_ids: set[int]) -> None:
    """Flip status pending -> compiled for the given items, re-reading the file.

    Re-read rather than reuse the in-memory copy so a long run does not clobber
    edits made to the grouping file while it was going.
    """
    groups = json.loads(GROUPS_PATH.read_text())

    def visit(v: dict) -> None:
        rs = v.get("rule_source") or {}
        if v["item_id"] in item_ids and rs.get("manual") == manual and rs.get("status") == "pending":
            rs["status"] = "compiled"

    for grp in groups["groups"]:
        for v in grp.get("variables", []):
            visit(v)
        for sub in grp.get("subgroups", []):
            for v in sub.get("variables", []):
                visit(v)
    GROUPS_PATH.write_text(json.dumps(groups, indent=2) + "\n")


def read_usage(manual: str, site_group: str) -> dict | None:
    """Token-count record a compile wrote, or None if absent."""
    usage_path = RULES_DIR / manual / f"{site_group}.usage.json"
    if not usage_path.exists():
        return None
    return json.loads(usage_path.read_text())


def summarize_output(manual: str, site_group: str) -> str:
    """One-line result for a finished chapter, read back from what it wrote."""
    out_path = RULES_DIR / manual / f"{site_group}.json"
    if not out_path.exists():
        return "no output file"
    units = json.loads(out_path.read_text())
    items = sorted({i for u in units for i in u.get("item_ids", [])})
    kinds: dict[str, int] = {}
    for u in units:
        kinds[u["kind"]] = kinds.get(u["kind"], 0) + 1
    kind_str = ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
    quarantined = ""
    report_path = RULES_DIR / manual / f"{site_group}.review.txt"
    if report_path.exists():
        m = re.search(r"quarantined[:\s]+(\d+)", report_path.read_text(), re.IGNORECASE)
        if m:
            quarantined = f", {m.group(1)} quarantined"
    usage = read_usage(manual, site_group)
    tokens = ""
    if usage and usage.get("usage_reported"):
        tokens = f" | {usage['total_tokens']:,} tok"
    return f"{len(units)} units{quarantined} | items {items or '[]'} | {kind_str or 'none'}{tokens}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manual", default="spcsm_2024", help="Manifest key to compile pending chapters for.")
    parser.add_argument("--item", type=int, default=None, help="Compile only this NAACCR item's chapter.")
    parser.add_argument("--run", action="store_true", help="Execute the compiles instead of printing them.")
    parser.add_argument("--dry-run", action="store_true", help="Pass --dry-run to each compile (segment only, no LLM).")
    parser.add_argument("--force", action="store_true", help="Recompile chapters whose output file already exists.")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort at the first failing chapter.")
    args = parser.parse_args()

    groups = json.loads(GROUPS_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    entry = manifest.get(args.manual)
    if entry is None:
        raise SystemExit(f"{args.manual!r} is not in {MANIFEST_PATH}; add a manifest entry first.")
    source = entry.get("source_markdown")
    if not source:
        raise SystemExit(f"Manifest entry {args.manual!r} has no source_markdown.")
    if not Path(source).exists():
        raise SystemExit(f"Source markdown {source!r} for {args.manual!r} does not exist.")

    targets = pending_targets(groups, args.manual, args.item)
    if not targets:
        print(f"No pending {args.manual} chapters in {GROUPS_PATH}.")
        return 0

    print(f"{len(targets)} pending chapter(s) for {args.manual}:\n")
    for t in targets:
        ids = ",".join(str(i) for i in t["item_ids"])
        print(f"  {ids:<12s} {t['name']:38s} L{str(t['line']):<6s} {t['chapter']}  [{t['group']}]")
    print()

    extra = ["--dry-run"] if args.dry_run else []

    if not args.run:
        for t in targets:
            cmd = build_command(t, args.manual, source, extra)
            print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        print("\nRe-run with --run to execute, or --run --dry-run to segment without calling the LLM.")
        return 0

    done: list[tuple[dict, str]] = []
    skipped: list[dict] = []
    failed: list[tuple[dict, str]] = []
    started = time.monotonic()

    for n, t in enumerate(targets, 1):
        out_path = RULES_DIR / args.manual / f"{t['site_group']}.json"
        if out_path.exists() and not args.force and not args.dry_run:
            print(f"[{n}/{len(targets)}] skip {t['chapter']} -- {out_path} exists (--force to recompile)")
            skipped.append(t)
            continue

        print(f"\n{'=' * 78}\n[{n}/{len(targets)}] {t['chapter']}  (items {t['item_ids']})\n{'=' * 78}")
        result = subprocess.run(build_command(t, args.manual, source, extra))
        if result.returncode != 0:
            failed.append((t, f"exit {result.returncode}"))
            print(f"!! FAILED: {t['chapter']} (exit {result.returncode})")
            if args.stop_on_error:
                break
            continue
        if args.dry_run:
            done.append((t, "segmented"))
            continue
        done.append((t, summarize_output(args.manual, t["site_group"])))
        mark_compiled(args.manual, set(t["item_ids"]))

    elapsed = time.monotonic() - started
    print(f"\n{'=' * 78}\nSummary -- {len(done)} compiled, {len(skipped)} skipped, "
          f"{len(failed)} failed in {elapsed / 60:.1f} min\n{'=' * 78}")
    for t, info in done:
        print(f"  ok    {t['chapter'][:44]:44s} {info}")
    for t in skipped:
        print(f"  skip  {t['chapter'][:44]:44s} already compiled")
    for t, why in failed:
        print(f"  FAIL  {t['chapter'][:44]:44s} {why}")

    if not args.dry_run:
        total_in = total_out = 0
        missing = False
        for t in done:
            usage = read_usage(args.manual, t[0]["site_group"])
            if usage and usage.get("usage_reported"):
                total_in += usage["input_tokens"]
                total_out += usage["output_tokens"]
            elif usage is not None:
                missing = True
        if total_in or total_out:
            note = "  (some chapters reported no usage)" if missing else ""
            print(f"\n  Total tokens: {total_in + total_out:,} "
                  f"(in {total_in:,} / out {total_out:,}){note}")
        elif done:
            print("\n  Total tokens: endpoint returned no usage metadata for these compiles.")

    if done and not args.dry_run:
        print(f"\nStatus flipped to 'compiled' in {GROUPS_PATH} for the chapters above.")
        print(f"Review each {RULES_DIR / args.manual}/<site_group>.review.txt before trusting the units.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
