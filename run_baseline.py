"""Baseline pre LLMs4Subjects (SemEval-2025 Task 5), kolekcia tib-core.

Metódy:
  knn   – prenos GND hesiel z najpodobnejších trénovacích dokumentov (podľa RUC Team) [HLAVNÝ BASELINE]
  label – priame porovnanie dokumentu s názvami hesiel GND (zero-shot, podľa silp_nlp / TartuNLP)

Príklady:
  python run_baseline.py --split dev  --method knn --k-neighbors 10 20 30 50 --power 1 2 4
  python run_baseline.py --split test --method knn --k-neighbors 30 --power 2
  python run_baseline.py --split dev  --method knn --encoder tfidf
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data import load_gnd, load_split, gnd_label_text  # noqa: E402
from evaluation import average_recall, evaluate, summary_line  # noqa: E402
from retrieval import knn_predict, most_frequent_subjects, top_n_neighbors  # noqa: E402

TOP_K = 50


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/llms4subjects/shared-task-datasets",
                    help="priečinok shared-task-datasets z repozitára jd-coderepos/llms4subjects")
    ap.add_argument("--collection", default="tib-core-subjects", choices=["tib-core-subjects", "all-subjects"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"],
                    help="dev = ladenie (index: train); test = finálne vyhodnotenie (index: train+dev)")
    ap.add_argument("--method", default="knn", choices=["knn", "label"])
    ap.add_argument("--encoder", default="dense", choices=["dense", "tfidf"])
    ap.add_argument("--model", default="intfloat/multilingual-e5-base",
                    help="model zo sentence-transformers / Hugging Face (iba pre --encoder dense)")
    ap.add_argument("--k-neighbors", type=int, nargs="+", default=[30], help="počet susedov (môže byť viac hodnôt = grid)")
    ap.add_argument("--power", type=float, nargs="+", default=[2.0], help="exponent váhy podobnosti (grid)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None, help="cuda / cpu (predvolene automaticky)")
    ap.add_argument("--limit", type=int, default=None, help="iba prvých N dopytov (rýchly test funkčnosti)")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    coll = Path(args.data_root) / "TIBKAT" / args.collection / "data"

    # 1) Dáta ---------------------------------------------------------------
    train = load_split(coll / "train")
    if args.split == "dev":
        index_recs, query_recs = train, load_split(coll / "dev")
    else:
        index_recs = train + load_split(coll / "dev")
        query_recs = load_split(coll / "test" / "gold-standard-testset")
    if args.limit:
        query_recs = query_recs[: args.limit]
    print(f"[data] index: {len(index_recs)} záznamov | dopyty ({args.split}): {len(query_recs)}")

    # 2) Enkóder --------------------------------------------------------------
    if args.encoder == "dense":
        from encoders import DenseEncoder
        enc = DenseEncoder(args.model, device=args.device, batch_size=args.batch_size,
                           cache_dir=Path(args.out_dir) / "cache")
        enc_name = args.model.split("/")[-1]
    else:
        from encoders import TfidfEncoder
        enc = TfidfEncoder().fit([r.text for r in index_recs])
        enc_name = "tfidf"

    run_name = args.run_name or f"{args.method}_{enc_name}_{args.collection}_{args.split}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    split_tag = "train" if args.split == "dev" else "train+dev"
    Q = enc.encode([r.text for r in query_recs], kind="doc",
                   cache_key=f"{args.collection}_{args.split}_q{len(query_recs)}")

    # 3) Predikcia ------------------------------------------------------------
    grid_rows = []
    if args.method == "knn":
        X = enc.encode([r.text for r in index_recs], kind="doc",
                       cache_key=f"{args.collection}_{split_tag}_idx")
        nb_idx, nb_sim = top_n_neighbors(Q, X, n=max(args.k_neighbors), device=args.device)
        index_subjects = [r.subjects for r in index_recs]
        fallback = most_frequent_subjects(index_subjects, TOP_K)
        best = None
        for kn, pw in itertools.product(args.k_neighbors, args.power):
            preds = knn_predict(nb_idx, nb_sim, index_subjects, kn, pw, TOP_K, fallback)
            tables = evaluate(query_recs, preds)
            ar = average_recall(tables["type_lang"])
            print(f"[knn] k_neighbors={kn:<4} power={pw:<4} {summary_line(tables['type_lang'])}")
            grid_rows.append({"k_neighbors": kn, "power": pw, "avg_recall": ar,
                              **{c: tables["type_lang"].loc["Overall", c]
                                 for c in tables["type_lang"].columns if c != "n"}})
            if best is None or ar > best[0]:
                best = (ar, {"k_neighbors": kn, "power": pw}, preds, tables)
        _, best_params, preds, tables = best
    else:
        gnd = load_gnd(Path(args.data_root) / "GND" / "dataset" /
                       ("GND-Subjects-tib-core.json" if args.collection == "tib-core-subjects"
                        else "GND-Subjects-all.json"))
        codes = [g["Code"] for g in gnd]
        L = enc.encode([gnd_label_text(g) for g in gnd], kind="label", cache_key=f"gnd_{args.collection}")
        nb_idx, _ = top_n_neighbors(Q, L, n=TOP_K, device=args.device)
        preds = [[codes[j] for j in row] for row in nb_idx]
        tables = evaluate(query_recs, preds)
        best_params = {}
        print(f"[label] {summary_line(tables['type_lang'])}")

    # 4) Uloženie ---------------------------------------------------------------
    pred_dir = run_dir / "predictions"
    for rec, p in zip(query_recs, preds):
        out = pred_dir / rec.rec_type / rec.lang / f"{rec.rec_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"dcterms:subject": p}, f, indent=1)

    with pd.ExcelWriter(run_dir / "metrics.xlsx") as xw:
        tables["type_lang"].to_excel(xw, sheet_name="Record Type and Language")
        tables["type"].to_excel(xw, sheet_name="Record Type")
        tables["language"].to_excel(xw, sheet_name="Language")
    for name, df in tables.items():
        df.round(4).to_csv(run_dir / f"metrics_{name}.csv")
    if grid_rows:
        pd.DataFrame(grid_rows).round(4).to_csv(run_dir / "grid.csv", index=False)

    info = {
        "run_name": run_name, "method": args.method, "encoder": args.encoder,
        "model": args.model if args.encoder == "dense" else "tfidf(1-2gram, sublinear, min_df=2)",
        "collection": args.collection, "split": args.split, "index": split_tag,
        "n_index": len(index_recs), "n_queries": len(query_recs), "best_params": best_params,
        "avg_recall": average_recall(tables["type_lang"]),
        "overall": {c: round(float(tables["type_lang"].loc["Overall", c]), 4)
                    for c in tables["type_lang"].columns if c != "n"},
        "runtime_s": round(time.time() - t0, 1),
        "python": platform.python_version(), "platform": platform.platform(),
    }
    with open(run_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n=== VÝSLEDOK ===")
    print(f"beh: {run_name} | najlepšie parametre: {best_params}")
    print(summary_line(tables["type_lang"]))
    print(tables["language"][[f"recall@{k}" for k in (5, 10, 20, 50)]].round(4).to_string())
    print(f"predikcie: {pred_dir}\nmetriky:   {run_dir / 'metrics.xlsx'}\nčas: {info['runtime_s']} s")


if __name__ == "__main__":
    main()
