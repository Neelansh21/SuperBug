"""
Guidelines Retrieval & Embeddings Module
=========================================

Local vector database retriever for the clinical guidelines RAG component.
Uses Gemini Embeddings to embed guideline sections (model configurable via
GEMINI_EMBEDDING_MODEL env var), and computes cosine similarity in memory
using NumPy.

Offline Fallback:
    If the GEMINI_API_KEY environment variable is missing, an automatic
    keyword-overlap fallback ensures the system remains testable and
    runnable without throwing connection errors.
"""

import os
import re
import logging

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)



# Project Path Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GUIDELINES_PATH = os.path.join(
    PROJECT_ROOT, "langchain_engine", "clinical_guidelines", "guidelines.md"
)


class ClinicalGuidelinesRetriever:
    """
    In-memory vector retriever for clinical guidelines.

    Parses guidelines.md into logical sections, computes their embeddings
    using Google GenAI, and queries them using cosine similarity.
    """

    def __init__(self, guidelines_path: str = None, api_key: str = None):
        if guidelines_path is None:
            guidelines_path = GUIDELINES_PATH
        self.guidelines_path = guidelines_path

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        self.sections = []
        self.embeddings_model = None
        self.section_embeddings = None

        self._load_and_parse_guidelines()

        if self.api_key:
            try:
                embedding_model_name = os.getenv(
                    "GEMINI_EMBEDDING_MODEL", "models/embedding-001"
                )
                self.embeddings_model = GoogleGenerativeAIEmbeddings(
                    model=embedding_model_name,
                    google_api_key=self.api_key,
                )
                self._compute_embeddings()
            except Exception as exc:
                logger.warning(
                    "Embeddings init failed, falling back to keyword retrieval: %s", exc
                )
                self.embeddings_model = None


    def _load_and_parse_guidelines(self):
        """Read guidelines.md and parse it into sections based on '## Section:' headers."""
        if not os.path.exists(self.guidelines_path):
            return

        with open(self.guidelines_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split the document by headers starting with "## Section:"
        raw_sections = content.split("## Section:")

        # First split contains the main title — skip it
        for raw_sec in raw_sections[1:]:
            lines = raw_sec.strip().split("\n")
            if not lines:
                continue

            title = lines[0].strip()
            body = "\n".join(lines[1:]).strip()

            self.sections.append({
                "title": title,
                "content": f"Pathogen/Resistance: {title}\n\n{body}",
                "raw_text": body,
            })



    def _compute_embeddings(self):
        """Pre-compute embeddings for all parsed sections."""
        if not self.sections or not self.embeddings_model:
            return

        texts = [sec["content"] for sec in self.sections]
        vectors = self.embeddings_model.embed_documents(texts)
        self.section_embeddings = np.array(vectors, dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between matrix a and query vector b."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a, axis=1)
        norm_b = np.linalg.norm(b)

        # Guard against division by zero
        denom = norm_a * norm_b
        denom[denom == 0] = 1.0

        return dot_product / denom

    def _keyword_retrieve(self, query: str, k: int = 1) -> list:
        """
        Fallback keyword-matching retrieval.

        Measures query word overlaps against clinical guideline sections
        when offline or when no API key is supplied.
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        scores = []

        for idx, sec in enumerate(self.sections):
            search_text = (sec["title"] + " " + sec["content"]).lower()
            text_words = set(re.findall(r"\w+", search_text))
            intersection = query_words.intersection(text_words)
            scores.append((len(intersection), idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scores[:k]]
        return [self.sections[idx] for idx in top_indices]

    def retrieve(self, query: str, k: int = 1) -> list:
        """
        Retrieve the top-k most relevant guideline sections for a query.

        Parameters
        ----------
        query : str
            Clinical search query (e.g. "Staphylococcus aureus Oxacillin").
        k : int
            Number of matching documents to return.

        Returns
        -------
        list[dict]
            List of matching section dictionaries containing title and content.
        """
        if not self.sections:
            return []

        # Fall back to keyword matching if no embedding model is configured
        if not self.embeddings_model or self.section_embeddings is None:
            return self._keyword_retrieve(query, k)

        try:
            query_vector = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
            similarities = self._cosine_similarity(self.section_embeddings, query_vector)
            top_indices = np.argsort(similarities)[::-1][:k]

            results = []
            for idx in top_indices:
                results.append(self.sections[idx])
            return results

        except Exception as exc:
            logger.warning("Embedding query failed, using keyword retrieval: %s", exc)
            return self._keyword_retrieve(query, k)
