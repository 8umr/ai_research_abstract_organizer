# AI Research Abstract Organizer

An intelligent research literature workflow tool and Streamlit web application designed for academic researchers, data scientists, and technical leaders. 

Upload a CSV of paper titles and abstracts to automatically cluster research themes, extract proposed methodologies and algorithmic architectures, catalog benchmark datasets, and structure author-stated limitations and tradeoffs.

---

## Key Capabilities

- **Automated Research Theme Clustering**: Classifies papers into primary research domains (e.g., *Large Language Models & Reasoning*, *Multimodal & Vision-Language*, *Retrieval-Augmented Generation*, *Model Efficiency & Quantization*, *Robotics & Embodied AI*, *AI Alignment & Safety*).
- **Fine-Grained Methodology Extraction**: Isolates specific algorithms, model architectures, and training mechanisms (e.g., *Monte Carlo Tree Search*, *Direct Preference Optimization*, *Continuous-Time Latent Flow Matching*).
- **Benchmark & Dataset Mapping**: Catalogs evaluation suites and corpora (e.g., *GSM8K*, *MSR-VTT*, *MS-MARCO*, *SWE-bench*, *QM9*).
- **Stated Limitations Taxonomy**: Categorizes explicit caveats, computational bottlenecks, distribution shift vulnerabilities, and hardware dependencies.
- **Batch Processing with Progress Tracking**: Smooth iterative batch inference with real-time progress bars, rate-limiting delays, and graceful error handling.
- **Interactive Visual Dashboards**: Interactive Plotly charts for theme distributions, methodology ranking, benchmark frequency, limitation categories, and text statistics.
- **Interactive Search & Deep-Dive**: Filter by theme, search across full-text abstracts and extracted entities, and view structured side-by-side cards.
- **Multi-Format Export**: Download enriched datasets as CSV, structured JSON, or an executive literature synthesis report in Markdown (`.md`).
- **Offline / Interactive Demo Mode**: Includes an offline heuristic analyzer and a pre-packaged 12-paper benchmark dataset so you can explore the complete application immediately without configuring an API key.

---

## File Structure

```text
ai_research_abstract_organizer/
├── app.py                  # Primary Streamlit web application
├── requirements.txt        # Python dependency manifest
├── README.md               # Setup and user guide
├── sample_papers.csv       # Pre-packaged 12-paper academic benchmark dataset
└── utils/
    ├── __init__.py
    ├── api_client.py       # API integration, connection testing & structured JSON extraction
    ├── analyzer.py         # Metrics aggregation, limitation taxonomy & synthesis generation
    └── visualizations.py   # Publication-ready Plotly interactive visualizations
```

---

## Installation & Quickstart

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Extract and Set Up Environment
```bash
# Unzip the project archive
unzip ai_research_abstract_organizer.zip
cd ai_research_abstract_organizer

# Create a virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## API Configuration

The application integrates with the xAI REST API endpoint (`https://api.x.ai/v1/chat/completions`) using OpenAI-compatible structured messaging.

You can provide your API key in either of two ways:
1. **Directly in the Sidebar**: Paste your API key into the password-masked **API Key** field in the Streamlit sidebar.
2. **Environment Variable**: Set `XAI_API_KEY` before launching Streamlit:
   ```bash
   export XAI_API_KEY="your-api-key-here"
   streamlit run app.py
   ```

### Supported Models
- `grok-2-latest` (Default, high reasoning fidelity and structured JSON adherence)
- `grok-beta`
- Custom model strings (configurable in the sidebar)

### Offline / Demo Mode
If you do not have an active API key or wish to test the UI and analytical dashboards without expending API credits, toggle **"Offline / Interactive Demo"** in the sidebar. The application will use an intelligent rule-based taxonomy extractor.

---

## Input CSV Format Requirements

Your CSV file can have any number of metadata columns, but it must include at least two columns representing paper titles and abstracts.

The application automatically detects column names matching common variations, or you can select them manually from dropdowns:
- **Title Column**: `title`, `paper_title`, `name`, `headline`
- **Abstract Column**: `abstract`, `paper_abstract`, `summary`, `text`, `description`

### Example CSV Structure:
```csv
title,abstract
"Scaling Test-Time Compute for Mathematical Reasoning","Recent frontier language models demonstrate notable capabilities... We introduce SearchPolicy-X combining MCTS with DPO across MATH and GSM8K. Limitations include latency overhead..."
"Adaptive Dense Retrieval with Dynamic Chunking for Enterprise RAG","Retrieval-Augmented Generation systems suffer from retrieval noise... We introduce SemanticChunk-RAG evaluated on MS-MARCO. Stated limitations include indexing latency..."
```

A complete 12-paper benchmark dataset (`sample_papers.csv`) is included in the package for immediate testing.

---

## Dashboard Overview

Once batch processing completes, six comprehensive views become available:
1. **Executive Overview**: High-level KPIs, theme distribution bar & donut charts, top methodologies, benchmark frequency, and abstract text length statistics.
2. **Research Themes**: Granular clustering by primary research subfield with expandable paper cards.
3. **Methods & Benchmarks**: Tabular catalogs mapping specific methodologies to papers, alongside benchmark evaluation suites.
4. **Limitations & Bottlenecks**: Classification of limitations across *Compute & Latency*, *Generalization*, *Data Quality/Bias*, *Safety/Reliability*, and *Hardware Coupling*.
5. **Paper Explorer & Search**: Instant full-text and semantic keyword search with maturity stage filters.
6. **Export & Synthesis**: One-click downloads for enriched CSV, JSON, and a formatted Markdown synthesis report.

---

## License & Attribution
Designed for reproducible AI research analysis and literature reviews.
