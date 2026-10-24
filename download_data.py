"""Stiahne z GitHubu iba potrebnú časť oficiálneho repozitára jd-coderepos/llms4subjects
(kolekcia tib-core: train/dev/test + slovník GND + oficiálny eval skript), cca 400 MB.

Vyžaduje nainštalovaný git (na Windows: https://git-scm.com/download/win).
Použitie:  python download_data.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/jd-coderepos/llms4subjects.git"
# Pevný commit = rovnaké dáta pre všetkých (reprodukovateľnosť). Overené 24. 10. 2026.
COMMIT = "098c35752392f13e2131911ec8aa2d8c1e895e00"
TARGET = Path("data") / "llms4subjects"
PATHS = [
    "/shared-task-datasets/TIBKAT/tib-core-subjects/data/",
    "/shared-task-datasets/GND/dataset/GND-Subjects-tib-core.json",
    "/shared-task-eval-script/",
    "/submission-format/",
]


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if shutil.which("git") is None:
        sys.exit("CHYBA: git nie je nainštalovaný alebo nie je v PATH.")
    if (TARGET / "shared-task-datasets").exists():
        print(f"Dáta už existujú v {TARGET} – preskakujem.")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO, str(TARGET)])
    run(["git", "sparse-checkout", "init", "--no-cone"], cwd=TARGET)
    run(["git", "sparse-checkout", "set", *PATHS], cwd=TARGET)
    run(["git", "checkout", COMMIT], cwd=TARGET)
    n = sum(1 for _ in (TARGET / "shared-task-datasets").rglob("*.jsonld"))
    print(f"Hotovo: {n} záznamov .jsonld v {TARGET / 'shared-task-datasets'}")


if __name__ == "__main__":
    main()
