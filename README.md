# Baseline: kNN prenos GND hesiel (LLMs4Subjects, kolekcia tib-core)

Funkčný a reprodukovateľný baseline pre 1. etapu projektu *Automatické priraďovanie predmetových
tagov k dokumentom z knižničného katalógu* (SemEval-2025 Task 5 – LLMs4Subjects).

**Úloha:** zo zadaného názvu a abstraktu záznamu z katalógu TIBKAT (EN alebo DE) vrátiť
zoradený zoznam 50 predmetových hesiel GND.

## Čo baseline robí

```
názov + abstrakt ──► embedding (multilingual-e5-base, bez dolaďovania)
                          │
                          ▼
     kosínusová podobnosť so všetkými trénovacími záznamami
                          │
                          ▼
     k najpodobnejších záznamov ──► ich GND heslá ──► skóre(heslo) = Σ sim^p
                                                         │
                                                         ▼
                                              top 50 hesiel (JSON, oficiálny formát)
```

* **Hlavný baseline (`--method knn`)** je zjednodušená reimplementácia prístupu **RUC Team**.
  Tento tím vyhral kvantitatívne hodnotenie kolekcie tib-core v SemEval-2025 Task 5
  (priemerný recall 0,66). RUC použil model Arctic-Embed 2.0 a index Faiss. My používame
  verejný model `intfloat/multilingual-e5-base` a presné vyhľadávanie maticovým súčinom,
  ktoré dáva rovnaké výsledky ako Faiss `IndexFlatIP`. Ich doplnkové príznaky (pozícia termínu,
  výskyt v názve) vynechávame. Verejný kód RUC Team nie je dostupný.
* **Kontrolný variant (`--encoder tfidf`)** robí ten istý kNN, ale nad TF-IDF vektormi
  (bez neurónovej siete, beží za ~1 min na CPU). Ukazuje, koľko pridáva sémantický model.
* **Zero-shot variant (`--method label`)** porovnáva dokument priamo s názvami hesiel GND
  (prístup silp_nlp / TartuNLP). Slúži na porovnanie.

Nič sa netrénuje ani nedolaďuje, nepoužíva sa LLM ani platené API.

## Požiadavky

| | |
|---|---|
| OS | otestované na Linuxe; návod je písaný pre Windows (PowerShell) |
| Python | **3.12** (oficiálny eval skript vyžaduje ≥ 3.12) |
| git | potrebný na stiahnutie dát ([git-scm.com](https://git-scm.com/download/win)) |
| GPU | odporúčaná NVIDIA (cieľová: RTX 3060 Laptop 6 GB); funguje aj CPU, len pomalšie |
| disk | ~0,5 GB dáta + ~1,1 GB model |

Testované verzie: torch 2.14, sentence-transformers 6.2, transformers 5.20, scikit-learn 1.9, numpy 2.5, pandas 3.0.

## Návod na spustenie (Windows / PowerShell)

Všetky príkazy sa spúšťajú v priečinku `spj_baseline`.

### 1. Virtuálne prostredie
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
(Ak PowerShell zablokuje aktiváciu: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

### 2. PyTorch s CUDA + ostatné balíky
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```
Ak by `cu128` nefungoval, vyberte príkaz pre Windows / pip / CUDA na
[pytorch.org/get-started](https://pytorch.org/get-started/locally/).
Posledný príkaz musí vypísať `True` a názov GPU. Ak vypíše `False`, aktualizujte ovládač NVIDIA.
Bez GPU všetko funguje aj na CPU, len pomalšie.

### 3. Dáta
```powershell
python download_data.py
```
Stiahne len potrebnú časť oficiálneho repozitára
[jd-coderepos/llms4subjects](https://github.com/jd-coderepos/llms4subjects) na pevnom commite
`098c3575`. Výsledok je v `data/llms4subjects/`:

| množina | priečinok | záznamov |
|---|---|---|
| train | `…/tib-core-subjects/data/train` | 32 043 |
| dev | `…/tib-core-subjects/data/dev` | 5 354 |
| test (s gold heslami) | `…/tib-core-subjects/data/test/gold-standard-testset` | 4 910 |

### 4. Rýchly test funkčnosti (~1 min)
```powershell
python run_baseline.py --split dev --encoder tfidf --limit 300
```

### 5. Ladenie parametrov na dev množine
Index = train, dopyty = dev. Skript vyskúša všetky kombinácie a vypíše tabuľku:
```powershell
python run_baseline.py --split dev --method knn --k-neighbors 30 50 100 150 200 300 --power 1 8 16 32 64 128 256
```
Pri prvom spustení sa stiahne model (~1,1 GB) a zakódujú sa texty (~5 min na RTX 3060 Laptop).
Embeddingy sa uložia do `runs/cache/`, takže ďalšie behy trvajú ~1 min. Z výpisu
(alebo z `runs/knn_multilingual-e5-base_tib-core-subjects_dev/grid.csv`) zoberte kombináciu
s najvyšším `avgR` (priemerný recall@5…50, hlavná metrika súťaže). Ak optimum padne na kraj gridu,
grid rozšírte.

U nás vyšlo optimum **k = 200, power = 64** (avgR na dev 0,6085). Embeddingy e5 majú podobnosti
sústredené v úzkom pásme (~0,8–0,9), preto pri power 1–4 nie je takmer žiadny rozdiel. Vysoký
exponent zostrí váženie v prospech najbližších susedov, podobne ako nízka teplota pri softmaxe.

### 6. Finálne vyhodnotenie na teste (iba raz, s parametrami z dev)
Index = train + dev, dopyty = gold test:
```powershell
python run_baseline.py --split test --method knn --k-neighbors 200 --power 64
```

### 7. Kontrolné a porovnávacie varianty
```powershell
# TF-IDF kNN (parametre z dev: k=300, p=3)
python run_baseline.py --split dev  --encoder tfidf --k-neighbors 100 150 200 300 --power 1 1.5 2 3
python run_baseline.py --split test --encoder tfidf --k-neighbors 300 --power 3
# zero-shot porovnanie s heslami GND
python run_baseline.py --split test --method label
```

### 8. (Voliteľné) Overenie oficiálnym eval skriptom
```powershell
python data\llms4subjects\shared-task-eval-script\llms4subjects-evaluation.py `
  --team_name knn_e5 `
  --true_labels_dir data\llms4subjects\shared-task-datasets\TIBKAT\tib-core-subjects\data\test\gold-standard-testset `
  --pred_labels_dir runs\knn_multilingual-e5-base_tib-core-subjects_test\predictions `
  --results_dir runs\official
```
Naša implementácia metrík (`src/evaluation.py`) dáva **rovnaké čísla** ako oficiálny skript.
Overené na TF-IDF behu: R@5 0,4808, F1@10 0,1906.

## Výstupy

`runs/<názov_behu>/`
* `predictions/<Typ>/<jazyk>/<id>.json` – top 50 hesiel v oficiálnom formáte `{"dcterms:subject": [...]}`
* `metrics.xlsx` – rovnaké 3 hárky ako oficiálny skript (typ × jazyk, typ, jazyk), P/R/F1 pre k = 5…50
* `metrics_*.csv`, `grid.csv` (ladenie), `run_info.json` (parametre, verzie, čas behu)

## Metodika vyhodnotenia

* Metriky: Precision@k, Recall@k, F1@k pre k = 5, 10, …, 50, rovnako ako v oficiálnom skripte.
  P a R sa priemerujú v rámci skupín typ × jazyk, F1 sa počíta z priemerov a „Overall“ je priemer skupín.
  Systémy v súťaži sa zoraďovali podľa **priemerného recallu** cez všetky k (`avgR`).
* Parametre (`k_neighbors`, `power`) sa ladia **iba na dev**. Test sa spúšťa raz s vybranými parametrami.
* Pri teste sa do indexu pridáva aj dev množina (train + dev). Ide o štandardný postup, test sa nikde nepoužíva na učenie.
* Priemerne má záznam len ~2,4 správneho hesla. P@k je preto zhora obmedzené (napr. P@50 ≤ ~0,05),
  dôležitejší je recall.

## Výsledky (tib-core)

| variant | dáta | parametre (z dev) | avgR | P@5 | R@5 | F1@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|---|---|---|---|
| kNN TF-IDF | dev | k=300, p=3 | 0.5958 | 0.1833 | 0.4565 | 0.2582 | 0.5266 | 0.5868 | 0.6602 |
| kNN TF-IDF | **test** | k=300, p=3 | **0.6270** | 0.1957 | 0.4808 | 0.2749 | 0.5532 | 0.6208 | 0.6965 |
| kNN multilingual-e5-base | dev | k=200, p=64 | 0.6085 | 0.1875 | 0.4705 | 0.2649 | 0.5424 | 0.6020 | 0.6692 |
| **kNN multilingual-e5-base (hlavný baseline)** | **test** | k=200, p=64 | **0.6371** | 0.1969 | 0.4837 | 0.2767 | 0.5642 | 0.6307 | 0.7041 |
| zero-shot (doc → heslo) e5 | test | – | 0.2448 | 0.0535 | 0.1293 | 0.0751 | 0.1714 | 0.2281 | 0.3196 |

Recall@5 podľa jazyka (test, úroveň „Language“):

| variant | de | en |
|---|---|---|
| kNN TF-IDF | 0.4699 | 0.4637 |
| kNN e5 | 0.4943 | 0.4255 |
| zero-shot e5 | 0.1759 | 0.1003 |

Pre porovnanie, oficiálne výsledky SemEval-2025 na tib-core (avgR, test): RUC Team 0.66 (1. miesto),
Annif 0.59, LA²I²F 0.58, silp_nlp 0.49, TartuNLP 0.40. Baseline organizátorov (GPT-4o) skončil
pod všetkými účastníkmi.

TF-IDF výsledky boli namerané na CPU (2 jadrá, ~1 min na beh). Výsledky e5 boli namerané na
Windows 11, Python 3.12.3 a RTX 3060 Laptop. Test trval 227 s vrátane kódovania. Na tom istom
počítači sa zopakovali aj čísla TF-IDF testu funkčnosti, čo potvrdzuje reprodukovateľnosť.

## Riešenie problémov

* **`CUDA out of memory`** → `--batch-size 16`
* **`torch.cuda.is_available()` je `False`** → nainštalovaná je CPU verzia torch. Odinštalujte ju (`pip uninstall torch`) a zopakujte krok 2.
* **Eval skript hlási `SyntaxError` pri f-stringu** → používate Python < 3.12.
* **Pomalé na CPU** → `--model intfloat/multilingual-e5-small` (menší model).

## Štruktúra

```
spj_baseline/
├── README.md             – tento návod
├── requirements.txt
├── download_data.py      – stiahnutie dát (git sparse checkout, pevný commit)
├── run_baseline.py       – hlavný skript: dáta → embeddingy → kNN → predikcie → metriky
├── src/
│   ├── data.py           – načítanie JSON-LD záznamov a slovníka GND
│   ├── encoders.py       – multilingual-e5 (sentence-transformers) a TF-IDF
│   ├── retrieval.py      – vyhľadávanie susedov a kNN skórovanie hesiel
│   └── evaluation.py     – P/R/F1@k zhodné s oficiálnym skriptom
└── podklady_do_spravy.md – text o baseline do správy + záznam do prílohy o AI
```
