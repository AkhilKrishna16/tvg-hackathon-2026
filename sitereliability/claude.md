
Person 3 — Claude AI Agent (Site Feasibility)
Branch: feature/agent

Your job
Take the top 10 candidates from the optimizer and have a Claude agent research each one, assessing real-world feasibility. This is where AI earns its place — the math told us the best locations geometrically, the agent tells us what's actually there.

Deliverables
src/agent/feasibility.py — Claude agent that researches each candidate location
src/agent/prompts.py — all prompt templates
Output: enriches results/top_candidates.json with a feasibility object per candidate
One-shot prompt

You are building the AI feasibility agent for a power grid placement optimizer.

Input: results/top_candidates.json (list of top candidate locations with lat/lon and scores)

Write src/agent/feasibility.py that runs a Claude-powered feasibility analysis on each candidate location.

For each candidate, the agent must assess:
1. Land use — what is actually at this location? (use reverse geocoding + web search)
2. Zoning — is this location zoned for industrial/utility use?
3. Environmental flags — flood zone? Protected land? Contaminated site?
4. Community sensitivity — near schools, hospitals, residential areas?
5. Grid proximity — how far is it from existing transmission infrastructure?
6. Overall feasibility verdict: HIGH / MEDIUM / LOW with one paragraph of reasoning

Implementation:

### src/agent/prompts.py
Define a function build_site_prompt(candidate: dict) -> str that returns a prompt like:

"You are a power infrastructure site selection expert. Analyze this candidate location for a new electrical substation:

Location: {lat}, {lon}
Optimization scores: load relief={load_relief_score}, loss reduction={loss_reduction_score}, sustainability={sustainability_score}, redundancy={redundancy_score}

Use your knowledge to assess:
1. What likely exists at or near these coordinates in Austin, Texas?
2. What are the zoning characteristics of this area?
3. Are there environmental concerns (flood plains, protected areas)?
4. Are there community sensitivity issues (proximity to schools, hospitals, residential)?
5. Is this location near existing transmission infrastructure?

Respond in this exact JSON format:
{
  'land_use': 'description of what is likely here',
  'zoning_assessment': 'industrial/commercial/residential/mixed — and whether utility use is feasible',
  'environmental_flags': ['list of concerns or empty array'],
  'community_sensitivity': 'low/medium/high with brief reason',
  'grid_proximity': 'description of nearby grid infrastructure',
  'feasibility': 'HIGH/MEDIUM/LOW',
  'reasoning': 'one paragraph synthesis'
}"

### src/agent/feasibility.py
- Use the Anthropic Python SDK (anthropic.Anthropic())
- Read ANTHROPIC_API_KEY from environment
- For each candidate in top_candidates.json:
  - Call claude-sonnet-4-6 with the site prompt
  - Parse the JSON response
  - Add a 'feasibility' key to the candidate dict with the parsed response
- Save the enriched list back to results/top_candidates.json
- Also save results/feasibility_report.md — a human-readable markdown report of all candidates ranked by feasibility verdict then composite score
- Run candidates concurrently using asyncio + anthropic's async client to stay within time budget
- Print progress as each candidate completes

Requirements:
- anthropic>=0.20.0
- Handle JSON parse errors gracefully (retry once, then mark feasibility as 'UNKNOWN')
- Add a 1 second delay between API calls to avoid rate limits if running synchronously