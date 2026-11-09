"""Retrieve similar already-coded reports to show the model as worked examples.

Keyword retrieval: TF-IDF cosine similarity between the query narrative and the training
narratives. The examples come only from the training split, so no test report (or its label)
can leak into a prompt. Each example is shown as its synopsis plus its human label, to keep
prompts short on free-tier token budgets.
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

    def top_k(self, narrative: str, k: int) -> pd.DataFrame:
        q = self.vec.transform([narrative])
        sims = (self.X @ q.T).toarray().ravel()
        idx = np.argsort(-sims, kind="stable")[:k]
        out = self.train.iloc[idx][["acn", "synopsis", "primary_problem"]].copy()
        out["similarity"] = sims[idx]
        return out.reset_index(drop=True)
