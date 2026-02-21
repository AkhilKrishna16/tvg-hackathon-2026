"""
Feasibility agent: runs Claude-powered site analysis on each candidate location
and enriches results/top_candidates.json with a feasibility assessment.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import anthropic

from .prompts import build_site_prompt

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
CANDIDATES_FILE = RESULTS_DIR / "top_candidates.json"
REPORT_FILE = RESULTS_DIR / "feasibility_report.md"

MODEL = "claude-sonnet-4-6"
CONCURRENCY = 3

VERDICT_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}


def parse_feasibility_response(text: str) -> dict | None:
    """Try to parse a JSON object out of the model's response text."""
    text = text.strip()
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


async def analyze_candidate(
    client: anthropic.AsyncAnthropic,
    candidate: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    candidate = dict(candidate)
    cid = candidate.get("id", "?")
    name = candidate.get("name", f"Candidate {cid}")

    async with semaphore:
        prompt = build_site_prompt(candidate)
        feasibility = None

        for attempt in range(2):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text
                feasibility = parse_feasibility_response(text)
                if feasibility is not None:
                    break
                # Retry with a stricter prompt
                prompt = (
                    "Return ONLY a raw JSON object — no markdown, no explanation — "
                    "with these exact keys: land_use, zoning_assessment, "
                    "environmental_flags, community_sensitivity, grid_proximity, "
                    f"feasibility, reasoning.\n\nOriginal prompt:\n{build_site_prompt(candidate)}"
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

        candidate["feasibility"] = feasibility
        verdict = feasibility.get("feasibility", "UNKNOWN")
        print(f"  ✓ Candidate {cid}: {name} → {verdict}")
        return candidate


def generate_markdown_report(candidates: list) -> str:
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            VERDICT_ORDER.get(
                c.get("feasibility", {}).get("feasibility", "UNKNOWN"), 3
            ),
            -c.get("composite_score", 0),
        ),
    )

    lines = [
        "# Site Feasibility Report",
        "",
        "Candidates ranked by feasibility verdict (HIGH → MEDIUM → LOW → UNKNOWN), "
        "then by composite optimization score.",
        "",
    ]

    for rank, c in enumerate(sorted_candidates, 1):
        f = c.get("feasibility", {})
        verdict = f.get("feasibility", "UNKNOWN")
        env_flags = f.get("environmental_flags", [])
        env_str = ", ".join(env_flags) if env_flags else "None"

        lines += [
            f"---",
            f"",
            f"## #{rank} — {c.get('name', 'Unknown')} (ID {c.get('id', '?')})",
            f"",
            f"**Verdict:** {verdict}  ",
            f"**Composite Score:** {c.get('composite_score', 'N/A')}  ",
            f"**Coordinates:** {c.get('lat')}, {c.get('lon')}",
            f"",
            f"| Score | Value |",
            f"|-------|-------|",
            f"| Load Relief | {c.get('load_relief_score')} |",
            f"| Loss Reduction | {c.get('loss_reduction_score')} |",
            f"| Sustainability | {c.get('sustainability_score')} |",
            f"| Redundancy | {c.get('redundancy_score')} |",
            f"",
            f"**Land Use:** {f.get('land_use', 'N/A')}  ",
            f"**Zoning:** {f.get('zoning_assessment', 'N/A')}  ",
            f"**Environmental Flags:** {env_str}  ",
            f"**Community Sensitivity:** {f.get('community_sensitivity', 'N/A')}  ",
            f"**Grid Proximity:** {f.get('grid_proximity', 'N/A')}",
            f"",
            f"**Reasoning:** {f.get('reasoning', 'N/A')}",
            f"",
        ]

    return "\n".join(lines)


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    if not CANDIDATES_FILE.exists():
        print(f"ERROR: {CANDIDATES_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    candidates = json.loads(CANDIDATES_FILE.read_text())
    print(f"Loaded {len(candidates)} candidates from {CANDIDATES_FILE}")
    print(f"Running feasibility analysis with model {MODEL} (concurrency={CONCURRENCY})...\n")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(CONCURRENCY)

    enriched = await asyncio.gather(
        *[analyze_candidate(client, c, semaphore) for c in candidates]
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_FILE.write_text(json.dumps(list(enriched), indent=2))
    print(f"\nSaved enriched candidates → {CANDIDATES_FILE}")

    report = generate_markdown_report(list(enriched))
    REPORT_FILE.write_text(report)
    print(f"Saved feasibility report  → {REPORT_FILE}")

    counts: dict[str, int] = {}
    for c in enriched:
        v = c.get("feasibility", {}).get("feasibility", "UNKNOWN")
        counts[v] = counts.get(v, 0) + 1

    print("\nSummary:")
    for verdict in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        if verdict in counts:
            print(f"  {verdict}: {counts[verdict]}")


if __name__ == "__main__":
    asyncio.run(main())
