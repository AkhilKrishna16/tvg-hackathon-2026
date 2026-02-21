def build_site_prompt(candidate: dict) -> str:
    return f"""You are a power infrastructure site selection expert. Analyze this candidate location for a new electrical substation:

Location: {candidate['lat']}, {candidate['lon']}
Name: {candidate.get('name', 'Unknown')}
Optimization scores: load_relief={candidate['load_relief_score']}, loss_reduction={candidate['loss_reduction_score']}, sustainability={candidate['sustainability_score']}, redundancy={candidate['redundancy_score']}
Composite score: {candidate['composite_score']}

Use your knowledge to assess:
1. What likely exists at or near these coordinates in Austin, Texas?
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
