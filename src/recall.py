"""
recall.py — Stage 2: Fast Recall

Embeds the parsed JD and all candidate profiles using sentence-transformers,
then uses FAISS to retrieve the top-K closest candidates in milliseconds.

Design choices:
- Model: all-MiniLM-L6-v2 — fast, ~80 MB, good semantic quality for resumes.
- Index: FAISS IndexFlatIP (inner product = cosine similarity on normalised vectors).
- Profiles are flexible dicts; build_profile_text() joins whatever fields exist.
"""

import os
import sys
from typing import List, Dict

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# sentence-transformers is optional: conflicts with gradio 6 on HF Spaces.
# Fall back to HF Inference API (huggingface_hub.InferenceClient) when unavailable.
try:
    from sentence_transformers import SentenceTransformer as _ST
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_jd_text(parsed_jd: dict) -> str:
    """Turn a parsed JD dict into a flat string for embedding."""
    parts = [
        "Required skills: " + ", ".join(parsed_jd.get("required_skills", [])),
        "Implied skills: "  + ", ".join(parsed_jd.get("implied_skills",  [])),
        "Seniority: "       + parsed_jd.get("seniority", ""),
        "Key needs: "       + ", ".join(parsed_jd.get("latent_needs",    [])),
    ]
    return ". ".join(p for p in parts if p.strip())


def build_profile_text(profile: dict) -> str:
    """
    Turn a candidate profile dict into a flat string for embedding.
    Uses a priority-ordered field list; ignores IDs and numeric-only fields
    that hurt embedding quality.
    Adjust field_order once the real dataset schema is known.
    """
    field_order = [
        "title", "headline", "current_role",
        "skills", "tech_skills", "tools",
        "summary", "bio", "about",
        "experience", "work_history",
        "education", "certifications",
    ]
    parts = []
    for field in field_order:
        val = profile.get(field)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        parts.append(str(val).strip())

    # Only fall back to free-form string values if no known fields matched.
    # Skip short values (likely IDs or numeric strings).
    if not parts:
        parts = [
            str(v) for v in profile.values()
            if isinstance(v, str) and len(v.strip()) > 10
        ]

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# RecallEngine
# ---------------------------------------------------------------------------

class RecallEngine:
    """
    Wraps embeddings + FAISS for fast semantic recall.

    Embedding backend (auto-selected):
      • sentence-transformers (local, if installed)
      • HF Inference API via huggingface_hub.InferenceClient (cloud fallback)
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv("VIVEKA_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self._model_name = model_name
        self.index = None
        self.candidates: List[Dict] = []

        if _HAS_ST:
            print(f"[recall] Loading local embedding model: {model_name}")
            self._st = _ST(model_name)
        else:
            hf_model = (
                model_name if "/" in model_name
                else f"sentence-transformers/{model_name}"
            )
            self._hf_model = hf_model
            self._hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN", "")
            print(f"[recall] Using HF Inference API for embeddings: {hf_model}")

    def _encode(self, texts: List[str]) -> "np.ndarray":
        """Embed texts and return L2-normalised float32 array."""
        if _HAS_ST:
            return self._st.encode(
                texts, batch_size=64, show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")

        from huggingface_hub import InferenceClient
        client = InferenceClient(token=self._hf_token or None)
        vecs = []
        for text in texts:
            emb = client.feature_extraction(text, model=self._hf_model)
            # InferenceClient returns list or ndarray; flatten to 1-D
            arr = np.array(emb, dtype="float32").flatten()
            vecs.append(arr)
        mat = np.stack(vecs).astype("float32")
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat /= np.maximum(norms, 1e-10)
        return mat

    def index_candidates(self, profiles: List[Dict]) -> None:
        """Embed all profiles and build the FAISS index."""
        if not profiles:
            raise ValueError("profiles list is empty.")

        self.candidates = profiles
        texts = [build_profile_text(p) for p in profiles]

        print(f"[recall] Embedding {len(texts)} profiles...")
        embeddings = self._encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[recall] Index ready — {self.index.ntotal} vectors, dim={dim}.")

    def recall(self, parsed_jd: dict, top_k: int = 200) -> List[Dict]:
        """
        Return the top_k candidates by cosine similarity to the JD.
        Each dict is a shallow copy of the original profile with
        'embedding_score' (float 0–1) added.
        """
        if self.index is None:
            raise RuntimeError("Call index_candidates() before recall().")

        jd_text = build_jd_text(parsed_jd)
        jd_vec  = self._encode([jd_text])

        k = min(top_k, len(self.candidates))
        scores, indices = self.index.search(jd_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            candidate = dict(self.candidates[idx])
            candidate["embedding_score"] = round(float(score), 4)
            results.append(candidate)

        return results


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/recall.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DUMMY_PROFILES = [
        {"id": "C001", "title": "Senior Data Engineer",
         "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL"],
         "summary": "8 years building large-scale data pipelines. Led platform migrations on AWS."},
        {"id": "C002", "title": "Junior Frontend Developer",
         "skills": ["React", "TypeScript", "CSS", "Figma"],
         "summary": "2 years building web UIs. Passionate about accessibility and design systems."},
        {"id": "C003", "title": "Data Analyst",
         "skills": ["SQL", "Python", "Tableau", "Excel"],
         "summary": "Analyst at a fintech. Strong SQL and storytelling with data."},
        {"id": "C004", "title": "ML Engineer",
         "skills": ["Python", "PyTorch", "Spark", "Kubernetes", "MLflow"],
         "summary": "Builds and deploys ML models at scale. Experience with feature stores."},
        {"id": "C005", "title": "DevOps Engineer",
         "skills": ["Kubernetes", "Terraform", "AWS", "CI/CD", "Docker"],
         "summary": "Platform engineer focused on infrastructure reliability."},
    ]

    SAMPLE_JD = {
        "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end", "works under ambiguity"],
    }

    engine = RecallEngine()
    engine.index_candidates(DUMMY_PROFILES)

    print("\nTop candidates for Senior Data Engineer JD:\n")
    results = engine.recall(SAMPLE_JD, top_k=5)
    for i, c in enumerate(results, 1):
        print(f"  {i}. [{c['id']}] {c['title']}  score={c['embedding_score']}")
