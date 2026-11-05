"""
Data Analysis and Synthesis Engine
Provides aggregation, frequency analysis, categorization of limitations,
and synthesis report generation.
"""

import pandas as pd
from collections import Counter
from typing import Dict, List, Any

LIMITATION_CATEGORIES = {
    "Compute & Latency Overhead": [
        "latency", "compute", "overhead", "time", "gpu", "memory", "bandwidth",
        "expensive", "cost", "tokens", "quadratically", "inference speed"
    ],
    "Generalization & Distribution Shift": [
        "out-of-distribution", "domain", "generalize", "deformable", "unseen",
        "calibration", "novel", "low-resource", "jargon", "robustness"
    ],
    "Data Quality, Bias & Representation": [
        "bias", "synthetic", "homogeneous", "culture", "dialect", "mode collapse",
        "curation", "annotation", "noise", "ground truth"
    ],
    "Safety, Hallucination & Reliability": [
        "hallucination", "jailbreak", "adversarial", "false positive", "refusal",
        "safety", "vulnerability", "infinite loop", "misattribution"
    ],
    "Hardware & Infrastructure Coupling": [
        "hardware", "cuda", "kernel", "ampere", "hopper", "sensor", "drift",
        "physical", "lighting", "rocm"
    ]
}

def categorize_limitation(limitation_text: str) -> str:
    """
    Classifies a stated limitation into a high-level taxonomy category.
    """
    text = limitation_text.lower()
    for category, keywords in LIMITATION_CATEGORIES.items():
        if any(kw in text for kw in keywords):
            return category
    return "Algorithmic & Theoretical Tradeoffs"

def compute_corpus_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates summary statistics across the analyzed research corpus.
    """
    total_papers = len(df)
    
    # Theme metrics
    theme_counts = df["primary_theme"].value_counts().to_dict()
    unique_themes = len(theme_counts)
    
    # Flatten lists
    all_methods = []
    for methods_list in df["methods"]:
        if isinstance(methods_list, list):
            all_methods.extend(methods_list)
        elif isinstance(methods_list, str):
            all_methods.append(methods_list)
            
    all_datasets = []
    for datasets_list in df["datasets"]:
        if isinstance(datasets_list, list):
            all_datasets.extend(datasets_list)
        elif isinstance(datasets_list, str):
            all_datasets.append(datasets_list)
            
    all_limitations = []
    limitation_cat_counts = Counter()
    for limitations_list in df["stated_limitations"]:
        if isinstance(limitations_list, list):
            for lim in limitations_list:
                all_limitations.append(lim)
                limitation_cat_counts[categorize_limitation(lim)] += 1
        elif isinstance(limitations_list, str):
            all_limitations.append(limitations_list)
            limitation_cat_counts[categorize_limitation(limitations_list)] += 1
            
    # Text length stats
    abstract_word_counts = df["abstract"].astype(str).apply(lambda x: len(x.split()))
    
    return {
        "total_papers": total_papers,
        "unique_themes": unique_themes,
        "theme_counts": theme_counts,
        "total_method_mentions": len(all_methods),
        "unique_methods": len(set(all_methods)),
        "top_methods": Counter(all_methods).most_common(15),
        "total_dataset_mentions": len(all_datasets),
        "unique_datasets": len(set(all_datasets)),
        "top_datasets": Counter(all_datasets).most_common(15),
        "total_limitations": len(all_limitations),
        "unique_limitations": len(set(all_limitations)),
        "limitation_categories": dict(limitation_cat_counts),
        "avg_abstract_words": round(abstract_word_counts.mean(), 1) if not abstract_word_counts.empty else 0,
        "min_abstract_words": int(abstract_word_counts.min()) if not abstract_word_counts.empty else 0,
        "max_abstract_words": int(abstract_word_counts.max()) if not abstract_word_counts.empty else 0
    }

def build_theme_method_matrix(df: pd.DataFrame, top_n_methods: int = 8) -> pd.DataFrame:
    """
    Creates a co-occurrence matrix between primary themes and top extracted methods.
    """
    all_methods = [m for sublist in df["methods"] for m in sublist]
    top_methods = [m for m, _ in Counter(all_methods).most_common(top_n_methods)]
    
    themes = sorted(df["primary_theme"].unique())
    matrix = {theme: {method: 0 for method in top_methods} for theme in themes}
    
    for _, row in df.iterrows():
        theme = row["primary_theme"]
        paper_methods = set(row["methods"]) if isinstance(row["methods"], list) else {row["methods"]}
        for m in top_methods:
            if m in paper_methods:
                matrix[theme][m] += 1
                
    return pd.DataFrame(matrix)

def generate_synthesis_markdown_report(df: pd.DataFrame, metrics: Dict[str, Any]) -> str:
    """
    Generates an executive research synthesis report in Markdown.
    """
    md = []
    md.append("# AI Research Abstract Organizer: Synthesis & Literature Report\n")
    md.append(f"**Total Papers Analyzed**: {metrics['total_papers']}  \n")
    md.append(f"**Identified Research Themes**: {metrics['unique_themes']}  \n")
    md.append(f"**Extracted Distinct Methodologies**: {metrics['unique_methods']}  \n")
    md.append(f"**Evaluation Benchmarks & Datasets**: {metrics['unique_datasets']}  \n")
    md.append(f"**Documented Research Limitations**: {metrics['total_limitations']}  \n\n")
    
    md.append("## 1. Executive Research Landscape Overview\n")
    md.append("This synthesis captures macro research directions, common architectural patterns, empirical evaluation benchmarks, and recurring technical limitations across the analyzed literature corpus.\n\n")
    
    md.append("### Primary Research Themes Distribution\n")
    md.append("| Research Theme | Papers Count | % of Corpus |\n")
    md.append("| :--- | :---: | :---: |\n")
    for theme, count in metrics["theme_counts"].items():
        pct = (count / metrics["total_papers"]) * 100
        md.append(f"| **{theme}** | {count} | {pct:.1f}% |\n")
    md.append("\n")
    
    md.append("## 2. Dominant Methodologies & Algorithmic Trends\n")
    md.append("The table below highlights the top methodologies and algorithms observed across the literature:\n\n")
    md.append("| Methodology / Technique | Frequency | Associated Themes |\n")
    md.append("| :--- | :---: | :--- |\n")
    
    # Map method to themes
    method_to_themes = {}
    for _, row in df.iterrows():
        t = row["primary_theme"]
        for m in row["methods"]:
            method_to_themes.setdefault(m, set()).add(t)
            
    for method, freq in metrics["top_methods"][:10]:
        themes_str = ", ".join(sorted(list(method_to_themes.get(method, []))))
        md.append(f"| `{method}` | {freq} | {themes_str} |\n")
    md.append("\n")
    
    md.append("## 3. Benchmark Datasets & Evaluation Environments\n")
    md.append("Identified empirical evaluation benchmarks and testing corpora:\n\n")
    md.append("| Benchmark / Dataset | Paper Mentions |\n")
    md.append("| :--- | :---: |\n")
    for dataset, freq in metrics["top_datasets"][:10]:
        md.append(f"| `{dataset}` | {freq} |\n")
    md.append("\n")
    
    md.append("## 4. Stated Limitations & Open Research Challenges\n")
    md.append("Aggregating explicit negative results, bottlenecks, and caveats reveals high-priority systemic frontiers:\n\n")
    md.append("| Limitation Category | Identified Instances | % Breakdown |\n")
    md.append("| :--- | :---: | :---: |\n")
    for cat, count in metrics["limitation_categories"].items():
        pct = (count / max(metrics["total_limitations"], 1)) * 100
        md.append(f"| **{cat}** | {count} | {pct:.1f}% |\n")
    md.append("\n")
    
    md.append("### Selected Notable Limitations from the Authors\n")
    for _, row in df.head(8).iterrows():
        title = row["title"]
        limitations = row["stated_limitations"]
        if isinstance(limitations, list) and limitations:
            md.append(f"- **{title}**:\n")
            for lim in limitations:
                md.append(f"  - *{lim}*\n")
    md.append("\n")
    
    md.append("## 5. Comprehensive Paper Index\n")
    for idx, row in df.iterrows():
        md.append(f"### {idx + 1}. {row['title']}\n")
        md.append(f"- **Primary Theme**: {row['primary_theme']}\n")
        methods_str = ", ".join(row['methods']) if isinstance(row['methods'], list) else str(row['methods'])
        datasets_str = ", ".join(row['datasets']) if isinstance(row['datasets'], list) else str(row['datasets'])
        md.append(f"- **Extracted Methods**: `{methods_str}`\n")
        md.append(f"- **Evaluation Datasets**: `{datasets_str}`\n")
        md.append(f"- **Key Contribution**: {row.get('key_contribution', 'N/A')}\n\n")
        
    return "".join(md)
