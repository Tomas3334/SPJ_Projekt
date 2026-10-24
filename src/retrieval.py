"""Presné vyhľadávanie najbližších susedov (kosínusová podobnosť) a kNN prenos hesiel."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp


def top_n_neighbors(Q, X, n: int, batch: int = 1024, device: str | None = None):
    """Pre každý riadok Q nájde n najpodobnejších riadkov X.

    Vektory sú L2-normalizované, takže skalárny súčin = kosínusová podobnosť.
    Pri ~40 tis. dokumentoch je presné vyhľadávanie maticovým súčinom rovnocenné indexu
    FAISS IndexFlatIP (ktorý použil RUC Team), len bez ďalšej závislosti.
    Vracia (indexy, podobnosti), oboje tvaru (len(Q), n), zoradené zostupne.
    """
    n = min(n, X.shape[0])
    nq = Q.shape[0]
    all_idx = np.zeros((nq, n), dtype=np.int64)
    all_sim = np.zeros((nq, n), dtype=np.float32)

    use_torch = False
    if not sp.issparse(X):
        try:
            import torch
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            use_torch = dev == "cuda"
        except ImportError:
            pass

    if use_torch:
        import torch
        Xt = torch.from_numpy(X).to("cuda", dtype=torch.float16)
        for s in range(0, nq, batch):
            q = torch.from_numpy(Q[s:s + batch]).to("cuda", dtype=torch.float16)
            sims = (q @ Xt.T).float()
            v, i = torch.topk(sims, n, dim=1)
            all_idx[s:s + batch] = i.cpu().numpy()
            all_sim[s:s + batch] = v.cpu().numpy()
        return all_idx, all_sim

    XT = X.T.tocsr() if sp.issparse(X) else X.T
    for s in range(0, nq, batch):
        sims = Q[s:s + batch] @ XT
        if sp.issparse(sims):
            sims = sims.toarray()
        sims = np.asarray(sims, dtype=np.float32)
        part = np.argpartition(-sims, n - 1, axis=1)[:, :n]
        psims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-psims, axis=1)
        all_idx[s:s + batch] = np.take_along_axis(part, order, axis=1)
        all_sim[s:s + batch] = np.take_along_axis(psims, order, axis=1)
    return all_idx, all_sim


def knn_predict(nb_idx: np.ndarray, nb_sim: np.ndarray, index_subjects: list[list[str]],
                k_neighbors: int, power: float, top_k: int = 50,
                fallback: list[str] | None = None) -> list[list[str]]:
    """kNN prenos hesiel (zjednodušene podľa RUC Team, SemEval-2025 Task 5).

    skóre(heslo) = súčet  sim(dokument, sused)^power  cez k_neighbors najbližších
    trénovacích dokumentov, ktoré dané heslo majú. Vyšší power zvýhodní najbližších susedov.
    Ak je kandidátov menej ako top_k, doplnia sa najčastejšie heslá z trénovacích dát.
    """
    preds = []
    fallback = fallback or []
    for row_idx, row_sim in zip(nb_idx[:, :k_neighbors], nb_sim[:, :k_neighbors]):
        scores: dict[str, float] = defaultdict(float)
        for j, s in zip(row_idx, row_sim):
            w = max(float(s), 0.0) ** power
            for code in index_subjects[j]:
                scores[code] += w
        ranked = [c for c, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))][:top_k]
        if len(ranked) < top_k:
            seen = set(ranked)
            ranked += [c for c in fallback if c not in seen][: top_k - len(ranked)]
        preds.append(ranked)
    return preds


def most_frequent_subjects(subject_lists: list[list[str]], n: int = 50) -> list[str]:
    cnt = Counter(c for subs in subject_lists for c in subs)
    return [c for c, _ in cnt.most_common(n)]
