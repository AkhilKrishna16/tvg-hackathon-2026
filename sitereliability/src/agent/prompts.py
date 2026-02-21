def build_site_prompt(candidate: dict, city: str = "Austin, Texas, USA") -> str:
    rank  = candidate.get("rank", candidate.get("id", "?"))
    name  = candidate.get("name", f"Site #{rank}")
    lat   = candidate["lat"]
    lon   = candidate["lon"]

    # Scores — GPU optimizer and legacy schemas both supported
    lr  = candidate.get("load_relief_score",   "N/A")
    los = candidate.get("loss_reduction_score","N/A")
    sus = candidate.get("sustainability_score","N/A")
    red = candidate.get("redundancy_score",    "N/A")
    cs  = candidate.get("composite_score",     candidate.get("score", "N/A"))
    dist = candidate.get("nearest_existing_km")
    cov5 = candidate.get("coverage_5km_pct")

    extra = ""
    if dist is not None:
        extra += f"\nNearest existing substation: {dist:.2f} km"
    if cov5 is not None:
        extra += f"\nDemand coverage within 5 km: {cov5:.1f}%"

    return f"""You are a power infrastructure site selection expert. \
Analyze this candidate location for a new electrical substation:

Location: {lat}, {lon}
Name: {name}
City: {city}
Optimization scores:
  load_relief={lr}, loss_reduction={los},
  sustainability={sus}, redundancy={red}
Composite score: {cs}{extra}

Use your knowledge to assess:
1. What likely exists at or near these coordinates in {city}?
2. What are the zoning characteristics of this area?
3. Are there environmental concerns (flood plains, protected areas)?
4. Are there community sensitivity issues (proximity to schools, hospitals, residential)?
5. Is this location near existing transmission infrastructure?

Respond in this exact JSON format (no markdown, no code fences, just raw JSON):
{{
  "land_use": "description of what is likely here",
  "zoning_assessment": "industrial/commercial/residential/mixed — and whether utility use is feasible",
  "environmental_flags": ["list of concerns or empty array"],
  "community_sensitivity": "low/medium/high with brief reason",
  "grid_proximity": "description of nearby grid infrastructure",
  "feasibility": "HIGH/MEDIUM/LOW",
  "reasoning": "one paragraph synthesis"
}}"""
