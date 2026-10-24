# Podklady do správy – baseline (člen 3)

> Návrh textu do kapitoly 3 (nahrádza odsek „Ako baseline bolo navrhnuté vyhľadávanie…“).

## 3.x Baseline riešenie

Ako baseline bol zvolený prenos predmetových hesiel z najpodobnejších trénovacích záznamov
(k-najbližších susedov, kNN). Prístup vychádza zo systému RUC Team, ktorý v úlohe
SemEval-2025 LLMs4Subjects dosiahol prvé miesto v kvantitatívnom hodnotení kolekcie tib-core [1].
Spojený názov a abstrakt dokumentu sa pomocou viacjazyčného embeddingového modelu
multilingual-e5-base (bez dolaďovania) prevedie na vektor. Potom sa vyhľadá k trénovacích
záznamov s najvyššou kosínusovou podobnosťou. Každé heslo GND, ktoré sa vyskytuje
v anotáciách týchto susedov, dostane skóre rovné súčtu podobností sim^p príslušných susedov.
Výstupom je 50 hesiel s najvyšším skóre. Parametre k a p sa ladili na vývojovej množine.
Najlepšie vyšli k = 200 a p = 64. Vysoký exponent je potrebný, lebo podobnosti embeddingov e5
sú sústredené v úzkom pásme (približne 0,8–0,9). Exponent tak zostrí váženie v prospech
najbližších susedov, podobne ako nízka teplota pri funkcii softmax. Pri p = 1–4 sa výsledky
takmer nelíšili.

Výber baseline bol odôvodnený nasledovne:

1. **Overená účinnosť na zvolenej kolekcii.** Prístup založený na podobných dokumentoch zvíťazil
   na kolekcii tib-core, s ktorou projekt pracuje [1].
2. **Nezávislosť od jazyka hesiel.** Heslá GND sú takmer výlučne nemecké, kým približne
   polovica záznamov je anglická. Porovnávanie dokumentu s inými dokumentmi a preberanie ich
   hesiel tento nesúlad obchádza. Pri priamom porovnávaní anglického textu s nemeckými názvami
   hesiel by nesúlad zostal.
3. **Nízke výpočtové nároky a reprodukovateľnosť.** Nevyžaduje dolaďovanie, LLM ani platené API.
   Beží na bežnom notebooku s GPU a v prípade TF-IDF variantu aj na CPU.
4. **Nadväznosť na 2. etapu.** Zoradený zoznam kandidátov z kNN priamo slúži ako vstup pre
   plánované opätovné zoradenie pomocou LLM. Prínos LLM tak možno merať oproti tomu istému
   zoznamu kandidátov.

Oproti systému RUC Team bol prístup zjednodušený: namiesto modelu Arctic-Embed 2.0 bol použitý
verejne dostupný multilingual-e5-base. Namiesto indexu Faiss sa použilo presné vyhľadávanie
maticovým súčinom, ktoré je pri ~37 tisíc dokumentoch výsledkovo rovnocenné. Vynechali sme
dodatočné príznaky (pozíciu termínu a výskyt v názve či abstrakte). Pôvodný kód RUC Team nie je
verejne dostupný, preto ide o vlastnú reimplementáciu.

Ako kontrolný variant bol ten istý kNN postup spustený nad reprezentáciou TF-IDF (slová
a bigramy). Ukazuje, aký podiel výsledku pripadá na samotný princíp prenosu hesiel a aký
na sémantický embeddingový model. Na porovnanie bol vyhodnotený aj zero-shot variant, ktorý
porovnáva dokument priamo s názvami hesiel GND (podľa silp_nlp a TartuNLP [1]).

**Experimentálny protokol.** Pri ladení tvorila index trénovacia množina (32 043 záznamov)
a dopyty vývojová množina (5 354 záznamov). Pri finálnom vyhodnotení tvorili index trénovacia
a vývojová množina a dopyty oficiálna testovacia množina s referenčnými heslami (4 910 záznamov,
verzia repozitára 098c3575). Experimenty s modelom e5 bežali na notebooku s GPU NVIDIA RTX 3060
Laptop (Windows 11, Python 3.12). Finálny beh na testovacej množine vrátane kódovania textov
trval necelé 4 minúty. Metriky P@k, R@k a F1@k (k = 5, …, 50) sa počítali rovnako ako
v oficiálnom hodnotiacom skripte. Zhodu sme overili spustením oficiálneho skriptu.

**Prvé výsledky (tib-core, test):**

| variant | avgR | P@5 | R@5 | F1@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|---|---|
| **kNN multilingual-e5-base (k=200, p=64) – baseline** | **0.6371** | 0.1969 | 0.4837 | 0.2767 | 0.5642 | 0.6307 | 0.7041 |
| kNN TF-IDF (k=300, p=3) – kontrolný variant | 0.6270 | 0.1957 | 0.4808 | 0.2749 | 0.5532 | 0.6208 | 0.6965 |
| zero-shot e5 (dokument → heslo) | 0.2448 | 0.0535 | 0.1293 | 0.0751 | 0.1714 | 0.2281 | 0.3196 |

Na vývojovej množine: kNN e5 avgR 0,6085, kNN TF-IDF 0,5958.

Recall@5 podľa jazyka dokumentu (test):

| variant | nemčina | angličtina |
|---|---|---|
| kNN e5 | 0.4943 | 0.4255 |
| kNN TF-IDF | 0.4699 | 0.4637 |
| zero-shot e5 | 0.1759 | 0.1003 |

**Interpretácia.**

- **Pozícia voči súťaži.** Baseline dosiahol na oficiálnej testovacej množine priemerný recall
  0,637. Na rebríčku SemEval-2025 (tib-core) by sa umiestnil za víťazným RUC Team (0,66)
  a pred druhým systémom Annif (0,59). Baseline organizátorov s GPT-4o skončil pod všetkými
  účastníkmi.
- **Rozhoduje princíp prenosu hesiel, nie typ reprezentácie.** Sémantický model priniesol oproti
  TF-IDF len malé zlepšenie (+0,010 avgR na teste, +0,013 na dev). Väčšinu výkonu teda dáva
  samotný princíp prenosu hesiel z podobných dokumentov.
- **Jazyk.** Model e5 bol lepší na nemeckých záznamoch, TF-IDF na anglických. To naznačuje
  možný prínos kombinácie oboch prístupov.
- **Zero-shot variant je výrazne slabší** (0,245). Na anglických záznamoch dosiahol takmer
  polovičný R@5 oproti nemeckým (0,100 oproti 0,176). Potvrdzuje to problém jazykového nesúladu
  medzi anglickými textami a nemeckými heslami GND, ktorý kNN prístup obchádza.
- **Typy záznamov.** Najhoršie výsledky boli pri záverečných prácach (Thesis), pri ktorých
  sú témy špecifickejšie.
- **Nadväznosť na 2. etapu.** R@50 dosahuje 0,70, takže zoznam 50 kandidátov obsahuje približne
  70 % správnych hesiel. Opätovné zoradenie pomocou LLM teda môže zlepšiť najmä metriky pri
  malom k (R@5 = 0,48). Na heslá mimo zoznamu kandidátov však nedosiahne.

**Poznámky k dátam (pre člena 2):**
- V priemere má záznam iba ~2,4 referenčného hesla. P@k je preto zhora obmedzené.
- Približne 7–8 % dev a testovacích záznamov má rovnaký názov ako niektorý trénovací záznam
  a ~2 % má zhodný aj názov a abstrakt (napr. iné vydania tej istej knihy). Ide o vlastnosť
  oficiálneho rozdelenia dát, nie o chybu. kNN z nej však profituje, čo treba pri interpretácii
  spomenúť.
- Počty záznamov v repozitári (train 32 043 / dev 5 354 / test 4 910) sa líšia od čísel
  v kapitole 3 (41 902 / 6 980). Treba zjednotiť a uviesť zdroj.

---

## Záznam do prílohy „Použitie LLM a AI agentov“

| | |
|---|---|
| **Nástroj / model** | Claude (Anthropic), model Claude Opus 5.5, aplikácia Claude s prístupom k súborom projektu |
| **Účel** | (1) preštudovanie zadania, webu súťaže LLMs4Subjects a oficiálneho repozitára; (2) výber a zdôvodnenie baseline; (3) návrh a implementácia kódu baseline (`run_baseline.py`, `src/*.py`, `download_data.py`); (4) spustenie TF-IDF variantu a overenie zhody metrík s oficiálnym skriptom; (5) návrh rozsahu ladenia parametrov a interpretácia výsledkov; (6) návrh README a tohto textu. Variant s modelom e5 spustil člen tímu na vlastnom počítači podľa README. |
| **Prompty** | *(sem vložiť doslovne vlastné správy z konverzácie, napr. „Potrebujem aby si mi pomohol vypracovať moju časť projektu…“, „pridal som do priečinka … ešte jeden dokument…“, „Áno, napíš mi kód a baseline priamo do priečinka…“)* |
| **Relevantné výstupy** | zdrojový kód v priečinku `spj_baseline`, README, výsledky TF-IDF variantu (tabuľka vyššie), návrh textu tejto časti správy |
| **Ľudská kontrola** | *(doplniť: čo ste overili sami – spustenie na vlastnom PC, kontrola čísel, úpravy textu)* |
