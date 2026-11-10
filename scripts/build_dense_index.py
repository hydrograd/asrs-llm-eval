"""Build the dense-retrieval neighbours and the retrieval-quality comparison.

    pip install -e ".[dense]"
    python scripts/build_dense_index.py

1. Embeds every narrative (train and test) with a small open sentence-embedding model,
   pinned to an exact revision, and normalises the vectors.
2. For every test report, finds the 20 most similar TRAINING reports by cosine similarity
   (exact search) and writes data/processed/dense_neighbours.json.gz. The runner reads this
   file, so CI and Docker never need the model or PyTorch.
3. Measures retrieval quality without any LLM, for sparse (TF-IDF) and dense alike, over the
   whole test pool: how often the k nearest training reports share the analyst's code, and
   the accuracy of a k-nearest-neighbour majority vote. Writes results/retrieval_quality.json.
"""
from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from asrs_eval import data, metrics as M
from asrs_eval.retrieve import Retriever

ROOT = Path(__file__).resolve().parents[1]
MODEL = "BAAI/bge-small-en-v1.5"
N_NEIGHBOURS = 20


def knn_quality(neigh_labels: list[list[str]], truth: list[str], k: int, b: dict) -> dict:
    prec = np.array([np.mean([l == t for l in nl[:k]]) for nl, t in zip(neigh_labels, truth)])
    vote = [Counter(nl[:k]).most_common(1)[0][0] for nl in neigh_labels]
    corr = np.array([v == t for v, t in zip(vote, truth)], dtype=float)
    lo, hi = M.bootstrap_ci(prec, np.mean, b["n_resamples"], b["ci"], b["seed"])
    alo, ahi = M.bootstrap_ci(corr, np.mean, b["n_resamples"], b["ci"], b["seed"])
    return {"label_agreement_at_k": {"estimate": float(prec.mean()), "ci95": [lo, hi]},
            "knn_vote_accuracy": {"estimate": float(corr.mean()), "ci95": [alo, ahi]},
            "_per_report_agreement": prec}


def main():
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import HfApi

    cfg = yaml.safe_load((ROOT / "config/experiment.yaml").read_text())
    b = {**cfg["bootstrap"], "seed": cfg["seed"]}
    k = cfg["retrieval"]["k_examples"]
    df = data.load_processed(ROOT / cfg["data"]["processed_path"])
    train, test = df[df.split == "train"].reset_index(drop=True), df[df.split == "test"].reset_index(drop=True)

    revision = HfApi().model_info(MODEL).sha
    model = SentenceTransformer(MODEL, revision=revision, device="cpu")
    t0 = time.time()
    enc = lambda s: model.encode(s.tolist(), batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=True, convert_to_numpy=True)
    Etr, Ete = enc(train.narrative), enc(test.narrative)
    print(f"embedded {len(train)} + {len(test)} narratives in {time.time() - t0:.0f}s")

    sims = Ete @ Etr.T
    top = np.argsort(-sims, axis=1, kind="stable")[:, :N_NEIGHBOURS]
    neighbours = {int(test.acn[i]): [[int(train.acn[j]), round(float(sims[i, j]), 5)] for j in top[i]]
                  for i in range(len(test))}
    out = ROOT / cfg["retrieval"]["dense_neighbours_path"]
    with gzip.open(out, "wt", compresslevel=9) as f:
        json.dump({"model": MODEL, "revision": revision, "max_seq_length": model.max_seq_length,
                   "dim": int(Etr.shape[1]), "similarity": "cosine (normalised dot product), exact",
                   "corpus": "training split narratives", "k": N_NEIGHBOURS,
                   "neighbours": neighbours}, f)
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")

    # retrieval quality, no LLM involved
    tl = train.label.to_numpy()
    dense_labels = [[tl[j] for j in top[i]] for i in range(len(test))]
    sparse = Retriever(train)
    sparse_labels = [list(sparse.top_k(n, N_NEIGHBOURS)["primary_problem"]) for n in test.narrative]
    # sparse returns raw ASRS labels; fold them like the human labels
    keep = set(df.label)
    sparse_labels = [[l if l in keep else "Other" for l in nl] for nl in sparse_labels]
    truth = test.label.tolist()
    q = {"k": k, "n_test_reports": len(test), "model": MODEL, "revision": revision}
    for name, labels in [("sparse_tfidf", sparse_labels), ("dense_embeddings", dense_labels)]:
        q[name] = knn_quality(labels, truth, k, b)
    est, lo, hi = M.paired_diff_ci(q["dense_embeddings"].pop("_per_report_agreement"),
                                   q["sparse_tfidf"].pop("_per_report_agreement"),
                                   b["n_resamples"], b["ci"], b["seed"])
    q["dense_minus_sparse_label_agreement"] = {"estimate": est, "ci95": [lo, hi]}
    (ROOT / "results/retrieval_quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps({kk: v for kk, v in q.items() if kk in ("sparse_tfidf", "dense_embeddings",
                                                              "dense_minus_sparse_label_agreement")}, indent=1))


if __name__ == "__main__":
    main()
