"""
Visualization Utilities
Provides modern Plotly interactive charts for theme clusters, methods, datasets,
and limitations distribution.
"""

from typing import Dict, Any, List
import pandas as pd

# Color palette: elegant academic / tech palette
PALETTE_PRIMARY = "#3B82F6"   # Indigo / Electric Blue
PALETTE_ACCENT = "#10B981"    # Emerald / Mint
PALETTE_WARN = "#F59E0B"      # Amber
PALETTE_DANGER = "#EF4444"    # Rose Red
PALETTE_PURPLE = "#8B5CF6"    # Violet
PALETTE_CYAN = "#06B6D4"      # Cyan

COLOR_SEQUENCE = [
    "#2563EB", "#059669", "#7C3AED", "#D97706", "#DC2626",
    "#0891B2", "#4F46E5", "#DB2777", "#65A30D", "#475569"
]

def create_theme_bar_chart(theme_counts: Dict[str, int]):
    """
    Creates an interactive horizontal bar chart for research theme distribution.
    """
    import plotly.express as px
    
    df = pd.DataFrame({
        "Research Theme": list(theme_counts.keys()),
        "Paper Count": list(theme_counts.values())
    }).sort_values(by="Paper Count", ascending=True)
    
    fig = px.bar(
        df,
        x="Paper Count",
        y="Research Theme",
        orientation="h",
        text="Paper Count",
        color="Paper Count",
        color_continuous_scale="Blues",
        title="Distribution of Research Themes"
    )
    
    fig.update_layout(
        title_font_size=16,
        xaxis_title="Number of Papers",
        yaxis_title="",
        showlegend=False,
        margin=dict(l=10, r=20, t=40, b=20),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_traces(
        textposition="outside",
        marker_line_color="#1E3A8A",
        marker_line_width=1
    )
    return fig

def create_theme_donut_chart(theme_counts: Dict[str, int]):
    """
    Creates a donut chart of research themes.
    """
    import plotly.express as px
    
    df = pd.DataFrame({
        "Theme": list(theme_counts.keys()),
        "Count": list(theme_counts.values())
    })
    
    fig = px.pie(
        df,
        names="Theme",
        values="Count",
        hole=0.45,
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Proportion of Corpus by Theme"
    )
    fig.update_layout(
        title_font_size=16,
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    return fig

def create_frequency_bar_chart(items_freq: List[tuple], title: str, x_label: str, color_hex: str = "#2563EB"):
    """
    Creates a clean horizontal bar chart for ranked methods or datasets.
    """
    import plotly.graph_objects as go
    
    # Take top 10
    top_items = items_freq[:10][::-1]  # reverse for ascending display
    names = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]
    
    fig = go.Figure(go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker=dict(
            color=color_hex,
            line=dict(color="rgba(0,0,0,0.15)", width=1)
        ),
        text=counts,
        textposition="outside"
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title=x_label, tickformat="d"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=30, t=40, b=20),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def create_limitation_category_chart(cat_counts: Dict[str, int]):
    """
    Creates a categorized bar chart representing research bottlenecks.
    """
    import plotly.express as px
    
    df = pd.DataFrame({
        "Category": list(cat_counts.keys()),
        "Count": list(cat_counts.values())
    }).sort_values(by="Count", ascending=False)
    
    fig = px.bar(
        df,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_sequence=["#DC2626", "#D97706", "#2563EB", "#7C3AED", "#059669"],
        text="Count",
        title="Identified Limitation Categories & Bottlenecks"
    )
    fig.update_layout(
        title_font_size=16,
        xaxis_title="",
        yaxis_title="Reported Bottlenecks",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=60),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(tickangle=-25)
    return fig

def create_abstract_length_histogram(lengths: List[int]):
    """
    Creates a distribution histogram of abstract word counts.
    """
    import plotly.express as px
    
    df = pd.DataFrame({"Word Count": lengths})
    fig = px.histogram(
        df,
        x="Word Count",
        nbins=12,
        color_discrete_sequence=["#4F46E5"],
        title="Abstract Word Count Distribution"
    )
    fig.update_layout(
        title_font_size=16,
        xaxis_title="Word Count per Abstract",
        yaxis_title="Number of Papers",
        margin=dict(l=10, r=10, t=40, b=20),
        height=320,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
