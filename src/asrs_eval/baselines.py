"""Majority-class and TF-IDF + logistic regression baselines.

Every LLM number is reported next to these. They are trained on the train split and
scored on the same test pool the LLMs are evaluated on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class MajorityClass:
    """Always predict the most common training label, with the training class frequencies
    as its probability vector (the best constant probabilistic forecast)."""

    def fit(self, texts, labels):
        freq = pd.Series(labels).value_counts(normalize=True).sort_index()
        self.classes_ = freq.index.to_numpy()
        self.prior_ = freq.to_numpy()
        self.majority_ = freq.idxmax()
        return self

    def predict(self, texts):
        return np.full(len(texts), self.majority_, dtype=object)

    def predict_proba(self, texts):
        return np.tile(self.prior_, (len(texts), 1))


def tfidf_logreg(seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                                   random_state=seed)),
    ])


def all_baselines(seed: int) -> dict:
    return {"majority": MajorityClass(), "tfidf_logreg": tfidf_logreg(seed)}
