#!/usr/bin/env python3
"""
Create a difficulty summary from MMMG evaluation results.

This script processes the MMMG test statistics and creates a summary
of task difficulty for each (school level, discipline) tuple, measured by GED.
"""

import json
from pathlib import Path

def create_difficulty_summary(input_json_path, output_json_path):
    """
    Process MMMG evaluation results and create difficulty summary.
    
    Args:
        input_json_path: Path to the recaptioned JSON with summed statistics
        output_json_path: Path to save the difficulty summary
    """
    
    # Load the input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Extract details
    details = data.get("Details", {})
    
    # Create difficulty summary
    difficulty_summary = []
    
    for level, disciplines in details.items():
        for discipline, stats in disciplines.items():
            cnt = stats["cnt"]
            if cnt == 0:
                continue
            
            # Normalize statistics by count (they are summed in the input)
            avg_ged = stats["GED"] / cnt
            avg_jaccard = stats["Jaccard"] / cnt
            avg_jaccard_edge = stats["Jaccard Edge"] / cnt
            avg_k_score_w = stats["K-score(w)"] / cnt
            
            # Create tuple entry
            tuple_entry = {
                "level": level,
                "discipline": discipline,
                "count": cnt,
                "avg_ged": avg_ged,
                "avg_jaccard": avg_jaccard,
                "avg_jaccard_edge": avg_jaccard_edge,
                "avg_k_score_w": avg_k_score_w,
                # GED is the primary difficulty metric (higher = harder)
                "difficulty_score": avg_k_score_w,
            }
            
            difficulty_summary.append(tuple_entry)
    
    # Sort by difficulty (ascending: easiest first)
    difficulty_summary.sort(key=lambda x: x["difficulty_score"], reverse=True)
    
    # Add difficulty ranks
    for idx, entry in enumerate(difficulty_summary):
        entry["difficulty_rank"] = idx + 1
        entry["difficulty_percentile"] = (idx + 1) / len(difficulty_summary) * 100
    
    # Create output structure
    output_data = {
        "total_tuples": len(difficulty_summary),
        "difficulty_range": {
            "min_k_score_w": difficulty_summary[0]["difficulty_score"],
            "max_k_score_w": difficulty_summary[-1]["difficulty_score"],
            "median_k_score_w": difficulty_summary[len(difficulty_summary) // 2]["difficulty_score"],
        },
        "tuples": difficulty_summary,
    }
    
    # Save to output file
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Created difficulty summary: {output_json_path}")
    print(f"   Total tuples: {len(difficulty_summary)}")
    print(f"   Difficulty range (k_score_w): {output_data['difficulty_range']['min_k_score_w']:.3f} - {output_data['difficulty_range']['max_k_score_w']:.3f}")
    print(f"   Easiest: {difficulty_summary[0]['level']}_{difficulty_summary[0]['discipline']} (k_score_w={difficulty_summary[0]['difficulty_score']:.3f})")
    print(f"   Hardest: {difficulty_summary[-1]['level']}_{difficulty_summary[-1]['discipline']} (k_score_w={difficulty_summary[-1]['difficulty_score']:.3f})")


if __name__ == "__main__":
    input_path = Path(__file__).parent / "Qwen-Image-Response-cfg_step1_summarize_recaptioned.json"
    # input_path = Path(__file__).parent / "Gemini-2.5-flash_step1_summarize_recaptioned.json"
    output_path = Path(__file__).parent / "mmmg_difficulty_summary.json"
    
    create_difficulty_summary(input_path, output_path)

