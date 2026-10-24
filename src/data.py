"""Načítanie záznamov TIBKAT (JSON-LD) a slovníka GND pre úlohu LLMs4Subjects."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

RECORD_TYPES = ["Article", "Book", "Conference", "Report", "Thesis"]
LANGUAGES = ["de", "en"]


@dataclass
class Record:
    rec_id: str            # názov súboru bez prípony, napr. 3A1743761929
    rec_type: str          # Article / Book / Conference / Report / Thesis
    lang: str              # de / en
    title: str
    abstract: str
    subjects: list[str] = field(default_factory=list)  # gold GND kódy, napr. "gnd:4124897-1"

    @property
    def text(self) -> str:
        """Vstup modelu: názov + abstrakt (podľa zadania úlohy)."""
        return f"{self.title}. {self.abstract}".strip()


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return _as_text(value.get("@value", ""))
    return str(value)


def _main_node(graph: list[dict]) -> dict:
    """Uzol samotného záznamu. Oficiálny eval skript berie posledný prvok @graph."""
    node = graph[-1]
    if "title" not in node:  # poistka, ak by poradie nesedelo
        for cand in graph:
            if "title" in cand:
                return cand
    return node


def _subjects(node: dict) -> list[str]:
    subj = node.get("dcterms:subject", [])
    if isinstance(subj, dict):
        subj = [subj]
    out = []
    for s in subj:
        code = s.get("@id") if isinstance(s, dict) else s
        if code and code not in out:
            out.append(code)
    return out


def load_split(split_dir: str | Path) -> list[Record]:
    """Načíta všetky *.jsonld zo štruktúry <split_dir>/<Typ>/<jazyk>/<id>.jsonld."""
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Priečinok s dátami neexistuje: {split_dir}")
    records: list[Record] = []
    for path in sorted(split_dir.glob("*/*/*.jsonld")):
        rec_type, lang = path.parent.parent.name, path.parent.name
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        node = _main_node(data["@graph"])
        records.append(Record(
            rec_id=path.stem,
            rec_type=rec_type,
            lang=lang,
            title=_as_text(node.get("title")),
            abstract=_as_text(node.get("abstract")),
            subjects=_subjects(node),
        ))
    if not records:
        raise ValueError(f"V {split_dir} sa nenašli žiadne záznamy .jsonld")
    return records


def load_gnd(path: str | Path) -> list[dict]:
    """Slovník GND (GND-Subjects-tib-core.json): zoznam {Code, Name, Alternate Name, ...}."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gnd_label_text(entry: dict) -> str:
    """Textová reprezentácia hesla GND: hlavný názov + alternatívne názvy."""
    alts = [a for a in entry.get("Alternate Name", []) or [] if a]
    return "; ".join([entry.get("Name", "")] + alts[:5])
