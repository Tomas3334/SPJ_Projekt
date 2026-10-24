"""Precision@k, Recall@k, F1@k rovnako ako oficiálny skript llms4subjects-evaluation.py.

- P@k = |gold ∩ pred[:k]| / k ;  R@k = |gold ∩ pred[:k]| / |gold|
- v každej skupine (typ záznamu × jazyk) sa P a R spriemerujú cez záznamy, F1 sa počíta z priemerov
- riadok "Overall" = priemer skupín (makro priemer), tak ako v oficiálnom skripte
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

K_VALUES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]


def _f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def _group_table(rows: list[tuple[str, list[str], list[str]]], k_values) -> dict:
    """rows: (skupina, gold, pred) -> {skupina: {metrika: hodnota}}"""
    acc = defaultdict(lambda: defaultdict(list))
    for group, gold, pred in rows:
        if not gold:
            continue
        g = set(gold)
        for k in k_values:
            hit = len(g & set(pred[:k]))
            acc[group][f"precision@{k}"].append(hit / k)
            acc[group][f"recall@{k}"].append(hit / len(g))
    table = {}
    for group, m in acc.items():
        out = {"n": len(m[f"recall@{k_values[0]}"])}
        for k in k_values:
            p = sum(m[f"precision@{k}"]) / len(m[f"precision@{k}"])
            r = sum(m[f"recall@{k}"]) / len(m[f"recall@{k}"])
            out[f"precision@{k}"], out[f"recall@{k}"], out[f"f1@{k}"] = p, r, _f1(p, r)
        table[group] = out
    return table


def evaluate(records, predictions: list[list[str]], k_values=K_VALUES) -> dict[str, pd.DataFrame]:
    """Vráti tabuľky: 'type_lang', 'type', 'language' (každá s riadkom Overall)."""
    levels = {
        "type_lang": lambda r: f"{r.rec_type}/{r.lang}",
        "type": lambda r: r.rec_type,
        "language": lambda r: r.lang,
    }
    result = {}
    for name, key in levels.items():
        rows = [(key(r), r.subjects, p) for r, p in zip(records, predictions)]
        df = pd.DataFrame(_group_table(rows, k_values)).T.sort_index()
        overall = df.drop(columns="n").mean()
        overall["n"] = df["n"].sum()
        df.loc["Overall"] = overall
        result[name] = df
    return result


def average_recall(df: pd.DataFrame, k_values=K_VALUES) -> float:
    """Hlavná metrika súťaže SemEval-2025 Task 5: priemer R@k cez k = 5..50 (riadok Overall)."""
    return float(df.loc["Overall", [f"recall@{k}" for k in k_values]].mean())


def summary_line(df: pd.DataFrame) -> str:
    o = df.loc["Overall"]
    return (f"avgR={average_recall(df):.4f} | "
            + " | ".join(f"P@{k}={o[f'precision@{k}']:.4f} R@{k}={o[f'recall@{k}']:.4f} F1@{k}={o[f'f1@{k}']:.4f}"
                         for k in (5, 10, 20, 50)))
