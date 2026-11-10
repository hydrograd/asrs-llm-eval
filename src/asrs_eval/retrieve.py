"""Retrieve similar already-coded reports to show the model as worked examples.

Two retrievers with one interface, top_k(narrative, k, acn=None):

- Retriever (sparse): TF-IDF cosine similarity between the query narrative and the training
  narratives, computed on the fly.
- DenseRetriever: cosine similarity between dense sentence embeddings, precomputed by
  scripts/build_dense_index.py for every test report and committed, so the runner, CI and the
  Docker image never need the embedding model or PyTorch. Exact search over ~20k vectors; no
  approximate index is needed at this scale.

Both retrieve only from the training split, so no test report (or its label) can leak into a
prompt. Each example is shown as its synopsis plus its human label, to keep prompts short on
free-tier token budgets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Retriever:
    def __init__(self, train: pd.DataFrame):
        self.train = train.reset_index(drop=True)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000,
                                   sublinear_tf=True, stop_words="english")
        self.X = self.vec.fit_transform(self.train["narrative"])

    def top_k(self, narrative: str, k: int, acn: int | None = None) -> pd.DataFrame:
        q = self.vec.transform([narrative])
        sims = (self.X @ q.T).toarray().ravel()
        idx = np.argsort(-sims, kind="stable")[:k]
        out = self.train.iloc[idx][["acn", "synopsis", "primary_problem"]].copy()
        out["similarity"] = sims[idx]
        return out.reset_index(drop=True)


class DenseRetriever:
    """Looks up precomputed dense-embedding neighbours by report number (ACN)."""

    def __init__(self, train: pd.DataFrame, neighbours_path):
        import gzip
        import json
        with gzip.open(neighbours_path, "rt") as f:
            d = json.load(f)
        self.meta = {k: v for k, v in d.items() if k != "neighbours"}
        self.neighbours = {int(a): v for a, v in d["neighbours"].items()}
        self.train = train.set_index("acn")

    def top_k(self, narrative: str, k: int, acn: int | None = None) -> pd.DataFrame:
        if acn is None or int(acn) not in self.neighbours:
            raise KeyError(f"no precomputed dense neighbours for ACN {acn}")
        pairs = self.neighbours[int(acn)][:k]
        acns = [int(a) for a, _ in pairs]
        out = self.train.loc[acns, ["synopsis", "primary_problem"]].reset_index()
        out["similarity"] = [float(s) for _, s in pairs]
        return out
