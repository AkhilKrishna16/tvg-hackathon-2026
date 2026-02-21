"""
feasibility.py — Claude-powered site feasibility agent.

Reads top_candidates.json, calls Claude for each candidate,
enriches with feasibility verdict + reasoning, saves back to disk.

Usage:
    python src/agent/feasibility.py
    python src/agent/feasibility.py \
        --candidates ../../graph/results/top_candidates.json \
        --output     ../../graph/results/top_candidates.json \
        --city       "Houston, Texas, USA"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import anthropic

from .prompts import build_site_prompt

# Defaults (used when no CLI args given)
_HERE           = Path(__file__).resolve()
_DEFAULT_RESULTS = _HERE.parents[2] / "results"

MODEL       = "claude-sonnet-4-6"
CONCURRENCY = 3

VERDICT_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _ensure_id_name(candidate: dict, index: int) -> dict:
    """Add id/name fields if missing (GPU optimizer output doesn't have them)."""
    c = dict(candidate)
    if "id" not in c:
        c["id"] = c.get("rank", index + 1)
    if "name" not in c:
        c["name"] = f"Site #{c['id']}"
    return c


# ── Core async analysis ───────────────────────────────────────────────────────

async def analyze_candidate(
    client: anthropic.AsyncAnthropic,
    candidate: dict,
    semaphore: asyncio.Semaphore,
    city: str,
) -> dict:
    cid  = candidate.get("id", "?")
    name = candidate.get("name", f"Candidate {cid}")

    async with semaphore:
        prompt     = build_site_prompt(candidate, city=city)
        feasibility = None

        for attempt in range(2):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                feasibility = _parse_json(response.content[0].text)
                if feasibility is not None:
                    break
                # Retry with stricter instruction
                prompt = (
                    "Return ONLY a raw JSON object — no markdown, no explanation — "
                    "with keys: land_use, zoning_assessment, environmental_flags, "
                    "community_sensitivity, grid_proximity, feasibility, reasoning.\n\n"
                    f"Original prompt:\n{build_site_prompt(candidate, city=city)}"
                )
            except Exception as exc:
                if attempt == 0:
                    await asyncio.sleep(1)
                    continue
                print(f"  ERROR candidate {cid} ({name}): {exc}", file=sys.stderr)

        if feasibility is None:
            feasibility = {
                "land_use": "UNKNOWN",
                "zoning_assessment": "UNKNOWN",
                "environmental_flags": [],
                "community_sensitivity": "UNKNOWN",
                "grid_proximity": "UNKNOWN",
                "feasibility": "UNKNOWN",
                "reasoning": "Could not parse a valid response from the model.",
            }

        candidate = dict(candidate)
        candidate["feasibility"] = feasibility
        verdict = feasibility.get("feasibility", "UNKNOWN")
        print(f"  [OK] Candidate {cid} ({name}) -> {verdict}")
        return candidate


# ── Report generation ─────────────────────────────────────────────────────────

def generate_markdown_report(candidates: list, city: str) -> str:
    sorted_c = sorted(
        candidates,
        key=lambda c: (
            VERDICT_ORDER.get(
                (c.get("feasibility") or {}).get("feasibility", "UNKNOWN")
                if isinstance(c.get("feasibility"), dict)
                else "UNKNOWN",
                3,
            ),
            -float(c.get("composite_score", 0)),
        ),
    )

    lines = [
        f"# Site Feasibility Report — {city}",
        "",
        "Candidates ranked by feasibility verdict (HIGH -> MEDIUM -> LOW -> UNKNOWN), "
        "then by composite score.",
        "",
    ]

    for rank, c in enumerate(sorted_c, 1):
        f = c.get("feasibility") or {}
        if not isinstance(f, dict):
            f = {"feasibility": str(f), "reasoning": ""}
        verdict   = f.get("feasibility", "UNKNOWN")
        env_flags = f.get("environmental_flags", [])
        env_str   = ", ".join(env_flags) if env_flags else "None"

        lines += [
            "---",
            "",
            f"## #{rank} — {c.get('name', 'Unknown')}  (rank {c.get('rank', '?')})",
            "",
            f"**Verdict:** {verdict}  ",
            f"**Composite Score:** {c.get('composite_score', 'N/A')}  ",
            f"**Coordinates:** {c.get('lat')}, {c.get('lon')}",
            "",
            "| Objective | Score |",
            "|-----------|-------|",
            f"| Load Relief     | {c.get('load_relief_score',    'N/A')} |",
            f"| Loss Reduction  | {c.get('loss_reduction_score', 'N/A')} |",
            f"| Sustainability  | {c.get('sustainability_score', 'N/A')} |",
            f"| Redundancy      | {c.get('redundancy_score',     'N/A')} |",
            "",
            f"**Land Use:** {f.get('land_use', 'N/A')}  ",
            f"**Zoning:** {f.get('zoning_assessment', 'N/A')}  ",
            f"**Environmental Flags:** {env_str}  ",
            f"**Community Sensitivity:** {f.get('community_sensitivity', 'N/A')}  ",
            f"**Grid Proximity:** {f.get('grid_proximity', 'N/A')}",
            "",
            f"**Reasoning:** {f.get('reasoning', 'N/A')}",
            "",
        ]

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_feasibility(
    candidates_path: Path,
    output_path: Path,
    city: str,
    report_path: Path | None = None,
) -> list:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if not candidates_path.exists():
        print(f"ERROR: {candidates_path} not found.", file=sys.stderr)
        sys.exit(1)

    raw        = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = [_ensure_id_name(c, i) for i, c in enumerate(raw)]

    print(f"\n[feasibility] {len(candidates)} candidates  |  city: {city}  |  model: {MODEL}")

    client    = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    enriched = list(await asyncio.gather(
        *[analyze_candidate(client, c, semaphore, city) for c in candidates]
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    print(f"  Saved enriched candidates -> {output_path}")

    if report_path is None:
        report_path = output_path.parent / "feasibility_report.md"
    report = generate_markdown_report(enriched, city)
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved feasibility report  -> {report_path}")

    counts: dict[str, int] = {}
    for c in enriched:
        f = c.get("feasibility") or {}
        v = f.get("feasibility", "UNKNOWN") if isinstance(f, dict) else str(f)
        counts[v] = counts.get(v, 0) + 1
    print(f"  Verdicts: { {k: counts[k] for k in ['HIGH','MEDIUM','LOW','UNKNOWN'] if k in counts} }")

    return enriched


def main():
    parser = argparse.ArgumentParser(description="Run Claude feasibility analysis on candidates.")
    parser.add_argument("--candidates", default=None,
                        help="Path to top_candidates.json (default: sitereliability/results/)")
    parser.add_argument("--output",     default=None,
                        help="Output path (default: same as --candidates)")
    parser.add_argument("--report",     default=None,
                        help="Path for feasibility_report.md")
    parser.add_argument("--city",       default="Austin, Texas, USA",
                        help='City name used in the prompt (default: "Austin, Texas, USA")')
    args = parser.parse_args()

    candidates_path = Path(args.candidates) if args.candidates else _DEFAULT_RESULTS / "top_candidates.json"
    output_path     = Path(args.output)     if args.output     else candidates_path
    report_path     = Path(args.report)     if args.report     else None

    asyncio.run(run_feasibility(candidates_path, output_path, args.city, report_path))


if __name__ == "__main__":
    main()
