"""
API Client and Information Extraction Engine
Supports remote API calls to xAI endpoints and offline heuristic extraction.
"""

import json
import re
import time
import requests
from typing import Dict, Any, Tuple, Optional

# API Endpoint constants
API_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-2-latest"
FALLBACK_MODELS = ["grok-2-latest", "grok-beta"]

SYSTEM_PROMPT = """You are an expert AI research scientist and meta-science literature analyst.
Your task is to analyze an academic paper title and abstract, then extract rigorous structured information.

You must respond ONLY with a valid JSON object following this exact schema:
{
  "primary_theme": "High-level research field (e.g., 'Large Language Models & Reasoning', 'Multimodal & Vision-Language', 'Retrieval-Augmented Generation', 'Model Efficiency & Quantization', 'Robotics & Embodied AI', 'Generative Video & Diffusion', 'Graph ML & Science', 'AI Alignment & Safety', 'Autonomous Agents & Software Engineering', 'Speech & Audio Foundation Models')",
  "sub_themes": ["2 to 3 specific subfield topics or granular themes"],
  "methods": ["List of distinct algorithms, architectures, mathematical frameworks, or novel techniques introduced or utilized (e.g. 'Monte Carlo Tree Search', 'Direct Preference Optimization', 'Dynamic Chunking')"],
  "datasets": ["List of benchmark datasets, corpora, or evaluation suites explicitly mentioned or clearly implied (e.g. 'GSM8K', 'MATH', 'MS-MARCO', 'ImageNet')"],
  "stated_limitations": ["List of explicit limitations, negative tradeoffs, computational bottlenecks, safety gaps, or failure modes acknowledged by authors"],
  "key_contribution": "A concise 1-2 sentence statement summarizing the primary technical breakthrough and reported metric gain",
  "maturity_stage": "One of: 'Empirical Benchmark', 'Novel Architecture', 'System & Infrastructure', 'Theoretical Formulation', 'Safety & Alignment Framework'"
}

Be thorough, precise, and objective. If a category (like datasets or limitations) is not explicitly named in the abstract, extract reasonable implicit evaluations or note 'Not explicitly stated in abstract'."""

def test_api_connection(api_key: str, model: str = DEFAULT_MODEL) -> Tuple[bool, str]:
    """
    Test the validity of the provided API key with a minimal request.
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty."
    
    endpoint = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Respond with the word 'OK' if you receive this."}
        ],
        "max_tokens": 10,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            return True, f"Connection successful! Model '{model}' responded."
        elif response.status_code == 401:
            return False, "Authentication failed (401): Invalid or unauthorized API key."
        elif response.status_code == 404:
            return False, f"Model not found (404): Model '{model}' may not be accessible with this key."
        else:
            return False, f"API error ({response.status_code}): {response.text[:200]}"
    except requests.exceptions.Timeout:
        return False, "Connection timed out after 15 seconds."
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"

def clean_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON from model output, stripping markdown code fences if present.
    """
    text = raw_text.strip()
    
    # Strip markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback regex search for JSON object
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse valid JSON from response: {text[:200]}")

def extract_paper_metadata_api(
    title: str, 
    abstract: str, 
    api_key: str, 
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Extract structured research metadata via the AI API endpoint.
    """
    endpoint = f"{API_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    
    user_prompt = f"TITLE: {title.strip()}\n\nABSTRACT:\n{abstract.strip()}"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    
    if response.status_code == 400 and "response_format" in response.text:
        # Retry without json_object constraint if model doesn't support the parameter
        payload.pop("response_format", None)
        response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    
    if response.status_code != 200:
        raise RuntimeError(f"API returned status {response.status_code}: {response.text[:300]}")
    
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    parsed = clean_json_response(content)
    
    # Ensure all required keys exist with safe fallbacks
    return normalize_metadata(parsed, title)

def normalize_metadata(data: Dict[str, Any], title: str) -> Dict[str, Any]:
    """
    Ensures standard structure and list types.
    """
    primary_theme = str(data.get("primary_theme", "General AI Research")).strip()
    
    def ensure_list(val):
        if isinstance(val, list):
            return [str(item).strip() for item in val if str(item).strip()]
        elif isinstance(val, str) and val.strip():
            return [str(val).strip()]
        return []

    sub_themes = ensure_list(data.get("sub_themes", []))
    methods = ensure_list(data.get("methods", []))
    datasets = ensure_list(data.get("datasets", []))
    limitations = ensure_list(data.get("stated_limitations", []))
    
    key_contribution = str(data.get("key_contribution", "Advances methodology in the evaluated domain.")).strip()
    maturity_stage = str(data.get("maturity_stage", "Novel Architecture")).strip()
    
    return {
        "primary_theme": primary_theme,
        "sub_themes": sub_themes,
        "methods": methods if methods else ["Unspecified Methodology"],
        "datasets": datasets if datasets else ["Domain Benchmark Suite"],
        "stated_limitations": limitations if limitations else ["No explicit limitations stated in abstract"],
        "key_contribution": key_contribution,
        "maturity_stage": maturity_stage
    }

def extract_paper_metadata_offline(title: str, abstract: str) -> Dict[str, Any]:
    """
    Intelligent heuristic rule-based metadata extractor for demo mode or offline testing.
    Analyzes title and abstract text against an extensive domain taxonomy.
    """
    text = f"{title} {abstract}".lower()
    
    # Theme heuristics
    themes_map = [
        ("Large Language Models & Reasoning", ["reasoning", "math", "gsm8k", "tree search", "mcts", "dpo", "test-time compute", "chain-of-thought", "self-correction", "language models"]),
        ("Multimodal & Vision-Language", ["multimodal", "video-text", "spatiotemporal", "image-caption", "vision and language", "omnialign", "msr-vtt", "clip", "vlm"]),
        ("Retrieval-Augmented Generation", ["retrieval-augmented", "rag", "dense retrieval", "chunking", "semanticchunk", "reranking", "colbert", "bi-encoder", "bm25"]),
        ("Model Efficiency & Quantization", ["quantization", "4-bit", "latency", "llm serving", "bandwidth", "smoothact", "cuda kernel", "pruning", "parameter scale", "a100"]),
        ("Robotics & Embodied AI", ["robotic", "manipulation", "tactile", "sensor fusion", "teleoperation", "dexterous", "policy learning", "motor control"]),
        ("Generative Video & Diffusion", ["flow matching", "diffusion", "video synthesis", "chronoflow", "temporal flow", "rotary positional", "fvd", "generative"]),
        ("Graph ML & Science", ["molecular", "equivariant", "graph transformer", "drug discovery", "interatomic", "qm9", "moleculenet", "pdbbind", "crystal"]),
        ("AI Alignment & Safety", ["jailbreak", "adversarial", "circuit", "refusal", "hallucination", "attribution", "factguard", "pathinspect", "steering", "safety"]),
        ("Autonomous Agents & Software Engineering", ["multi-agent", "swe-bench", "orchestration", "software engineering", "autonomous coding", "dialogue distillation", "synthdialog"])
    ]
    
    detected_theme = "General AI Foundations"
    for theme, keywords in themes_map:
        if any(kw in text for kw in keywords):
            detected_theme = theme
            break
            
    # Methods heuristics
    methods_candidates = [
        ("Monte Carlo Tree Search (MCTS)", ["monte carlo tree search", "mcts"]),
        ("Direct Preference Optimization (DPO)", ["direct preference optimization", "dpo"]),
        ("Contrastive Cross-Attention", ["contrastive", "cross-attention"]),
        ("Hierarchical Dynamic Chunking", ["dynamic chunking", "hierarchical semantic boundary", "semanticchunk"]),
        ("Dense Bi-Encoder & ColBERTv2", ["colbert", "bi-encoder", "dense retrieval"]),
        ("Post-Training Quantization (PTQ)", ["post-training quantization", "4-bit", "quantization", "smoothact"]),
        ("Diffusion Consistency Policy", ["diffusion consistency", "diffusion policy", "consistency models"]),
        ("Continuous-Time Latent Flow Matching", ["flow matching", "continuous-time latent flow"]),
        ("E(3)-Equivariant Graph Transformer", ["equivariant", "graph transformer", "equichem"]),
        ("Dynamic Fact Attribution Steering", ["fact attribution", "attribution verification", "attention-head steering"]),
        ("Multi-Agent Debate & Distillation", ["multi-agent debate", "critique filtering", "dialogue distillation"]),
        ("Activation Patching & Sparse Autoencoders", ["activation patching", "sparse autoencoder", "circuit discovery"]),
        ("Hierarchical Agentic Orchestration", ["hierarchical multi-agent", "agent orchestration"]),
        ("Latent Safety Vector Projection", ["latent safety vector", "perturbation smoothing"])
    ]
    
    found_methods = [name for name, kws in methods_candidates if any(kw in text for kw in kws)]
    if not found_methods:
        found_methods = ["Transformer Architecture Adaptation", "Empirical Evaluation Pipeline"]
        
    # Datasets heuristics
    datasets_candidates = [
        ("MATH & GSM8K", ["math", "gsm8k"]),
        ("OlympiadBench", ["olympiadbench"]),
        ("MSR-VTT & ActivityNet", ["msr-vtt", "activitynet"]),
        ("COCO Captions", ["coco captions"]),
        ("MS-MARCO", ["ms-marco"]),
        ("NaturalQuestions", ["naturalquestions"]),
        ("WikiText-2 & C4", ["wikitext", "c4"]),
        ("MMLU", ["mmlu"]),
        ("UCF-101 & WebVid-10M", ["ucf-101", "webvid"]),
        ("QM9 & MoleculeNet", ["qm9", "moleculenet"]),
        ("PDBbind", ["pdbbind"]),
        ("LongBench & L-Eval", ["longbench", "l-eval"]),
        ("MultiWOZ 2.4", ["multiwoz"]),
        ("SWE-bench Verified", ["swe-bench"]),
        ("HumanEval-X", ["humaneval"]),
        ("MM-SafetyBench & JailbreakBench", ["safetybench", "jailbreakbench", "advglue"])
    ]
    
    found_datasets = [name for name, kws in datasets_candidates if any(kw in text for kw in kws)]
    if not found_datasets:
        found_datasets = ["Standard Academic Benchmark Suite"]
        
    # Limitations heuristics
    limitations_candidates = [
        ("High inference latency and test-time compute overhead", ["latency overhead", "inference time", "computational expenditure"]),
        ("GPU memory consumption scaling bottleneck", ["gpu memory", "memory bandwidth", "memory overhead", "token bloat"]),
        ("Performance degradation on out-of-distribution or low-resource domains", ["out-of-distribution", "low-resource", "domain jargon"]),
        ("Hardware-specific CUDA kernel coupling", ["cuda kernel", "ampere", "hardware", "consumer hardware"]),
        ("Fragility to sensor drift and deformable physical dynamics", ["sensor calibration", "deformable soft objects", "lighting variations"]),
        ("Risk of mode collapse and lack of cultural diversity", ["mode collapse", "homogeneous", "cultural dialectal"]),
        ("Token cost and potential for infinite execution loops", ["token cost", "infinite execution loops"]),
        ("False positive safety attribution and OCR accuracy degradation", ["false positive", "ocr accuracy", "attribution warnings"])
    ]
    
    found_limitations = [name for name, kws in limitations_candidates if any(kw in text for kw in kws)]
    if not found_limitations:
        found_limitations = ["Limited exploration of multi-modal generalization", "Inference overhead under real-time constraints"]

    # Sub themes
    sub_themes = [f"{detected_theme} Optimization", "Empirical Evaluation"]
    
    return {
        "primary_theme": detected_theme,
        "sub_themes": sub_themes,
        "methods": found_methods[:4],
        "datasets": found_datasets[:4],
        "stated_limitations": found_limitations[:3],
        "key_contribution": f"Introduces novel methodology targeting {detected_theme.lower()} with substantial empirical performance gains over baseline systems.",
        "maturity_stage": "Novel Architecture" if "transformer" in text or "framework" in text else "Empirical Benchmark"
    }
