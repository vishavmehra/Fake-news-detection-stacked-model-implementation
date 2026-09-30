# Fake News Detection — Deep Learning, Machine Learning, Stacked Ensembles and Transformers

This project studies automatic fake-news detection across **five public datasets** and three
generations of models:

1. **Classic deep learning** — Uni/Bi-directional LSTM, Uni/Bi-directional GRU and 1-D CNN (Keras).
2. **Classic machine learning** — SVM, Logistic Regression, Decision Tree, KNN and two Random Forests
   on TF-IDF / term-frequency features (scikit-learn).
3. **A custom stacked ensemble** that feeds the predictions of all DL + ML models into a Random Forest
   meta-classifier.
4. **Fine-tuned transformers** (DistilBERT, BERT, RoBERTa) combined with TF-IDF and metadata baselines,
   probability ensembles, threshold tuning, a stacked meta-learner and — on LIAR — a metadata
   **Graph Convolutional Network (GCN)**.

The classic pipeline is developed on **ISOT** (long news articles) and then applied to **LIAR**
(short political claims). The transformer pipeline is developed on LIAR and then re-trained from
scratch on **GossipCop**, **PolitiFact** and **PolyglotFakeFacts** to test how well it generalises.

---

## Table of contents

- [Repository structure](#repository-structure)
- [Datasets](#datasets)
- [Label convention](#label-convention)
- [Part 1 — ISOT: DL vs ML and the stacked ensemble](#part-1--isot-dl-vs-ml-and-the-stacked-ensemble)
- [Part 2 — LIAR](#part-2--liar)
- [Part 3 — Cross-dataset transformer ensembles](#part-3--cross-dataset-transformer-ensembles)
- [Results summary](#results-summary)
- [Evaluation notes and known limitations](#evaluation-notes-and-known-limitations)
- [How to run](#how-to-run)
- [Citations](#citations)

---

## Repository structure

```
.
├── datasets/
│   ├── ISOT/                    Fake.csv, True.csv
│   ├── LIAR/                    train.csv, valid.csv, test.csv
│   ├── GossipCop/               gossipcop_fake.csv, gossipcop_real.csv
│   ├── PolitiFact/              politifact_fake.csv, politifact_real.csv
│   └── PolyglotFakeFacts/       80.xlsx, 20.xlsx, Fake.xlsx, Real.xlsx
├── notebooks/
│   ├── 01_ISOT/
│   │   ├── 01_bilstm_bigru_baseline.ipynb        Bi-LSTM and Bi-GRU baseline
│   │   ├── 02_dl_vs_ml_comparison.ipynb          Bi-LSTM, Bi-GRU, CNN vs 6 ML models
│   │   └── 03_stacked_ensemble.ipynb             5 DL + 6 ML models + stacked RF meta-model
│   ├── 02_LIAR/
│   │   ├── 01_dl_ml_stacked_ensemble.ipynb       ISOT pipeline applied to LIAR (binary + 6-class)
│   │   └── 02_transformer_ensemble_gnn.ipynb     DistilBERT/BERT/RoBERTa + credibility + GCN
│   └── 03_cross_dataset_transformers/
│       ├── gossipcop_transformer_ensemble.ipynb
│       ├── politifact_transformer_ensemble.ipynb
│       └── polyglotfakefacts_transformer_ensemble.ipynb
├── requirements.txt
└── README.md
```

| Notebook | Dataset | Environment it was run in |
|---|---|---|
| `01_ISOT/*` | ISOT | Google Colab, TensorFlow 2.18 / Keras 3.8 |
| `02_LIAR/01_dl_ml_stacked_ensemble` | LIAR | Google Colab, TensorFlow 2.18 / Keras 3.8 |
| `02_LIAR/02_transformer_ensemble_gnn` | LIAR | Local Windows + NVIDIA RTX 2060, PyTorch 2.5.1 (CUDA 12.1), Transformers 5.9 |
| `03_cross_dataset_transformers/*` | GossipCop, PolitiFact, PolyglotFakeFacts | Local Windows + NVIDIA RTX 2060, PyTorch 2.5.1 (CUDA 12.1), Transformers 5.9 |

All notebooks are committed **with their outputs**, so every number in this README can be checked
directly in the notebook without re-running it.

---

## Datasets

| Dataset | Folder | Size | Content | Task(s) used |
|---|---|---|---|---|
| **ISOT Fake News** | `datasets/ISOT` | 44,898 articles (21,417 true / 23,481 fake) | Full news articles with `title`, `text`, `subject`, `date`. True articles are from Reuters; fake articles are from sites flagged by PolitiFact / Wikipedia. | Binary |
| **LIAR** | `datasets/LIAR` | 12,836 statements (train 10,269 / valid 1,284 / test 1,283) | Short political claims from PolitiFact with 6 truth labels and speaker metadata | Binary + 6-class |
| **GossipCop** (FakeNewsNet) | `datasets/GossipCop` | 22,140 headlines (5,323 fake / 16,817 real) | Entertainment / celebrity news: `id`, `news_url`, `title`, `tweet_ids` | Binary |
| **PolitiFact** (FakeNewsNet) | `datasets/PolitiFact` | 1,056 headlines (432 fake / 624 real) | Political news: `id`, `news_url`, `title`, `tweet_ids` | Binary |
| **PolyglotFakeFacts** | `datasets/PolyglotFakeFacts` | 10,206 articles (4,912 fake / 5,294 real) in 18 languages, official 80 / 20 split | Headline, original text, English translation, keywords, URL/domain, language | Binary |

### ISOT columns

| Column | Description |
|---|---|
| `title` | Article headline |
| `text` | Article body (the only field used for modelling) |
| `subject` | Topic category — `politicsNews`, `worldnews` (true) or `News`, `politics`, `left-news`, `Government News`, `US_News`, `Middle-east` (fake). Dropped before modelling because it alone reveals the label. |
| `date` | Publication date (dropped) |

### LIAR columns

The CSVs have generic headers `column00 … column13`; the notebooks rename them to:

| # | Name | Description |
|---|---|---|
| 0 | `Id` | Statement JSON id |
| 1 | `Label` | `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true` |
| 2 | `Statement` | The claim text |
| 3 | `Subject` | Comma-separated topics |
| 4 | `Speaker` | Who made the claim |
| 5 | `Job Title` | Speaker's job |
| 6 | `State` | Speaker's state |
| 7 | `Party` | Speaker's party |
| 8–12 | `Barely True Counts`, `False Counts`, `Half True Counts`, `Mostly True Counts`, `Pants on Fire Counts` | Speaker's credit history (how often each verdict was given to the speaker) |
| 13 | `Context` | Venue / location of the statement |

Label distribution (train / valid / test):

| Label | Train | Valid | Test |
|---|---|---|---|
| pants-fire | 842 | 116 | 92 |
| false | 1,998 | 263 | 250 |
| barely-true | 1,657 | 237 | 214 |
| half-true | 2,123 | 248 | 267 |
| mostly-true | 1,966 | 251 | 249 |
| true | 1,683 | 169 | 211 |

### GossipCop and PolitiFact columns

GossipCop (celebrity / entertainment fact-checks from GossipCop.com) and PolitiFact (political
fact-checks from PolitiFact.com) are two separate datasets that share the same file format.
They are kept in separate folders and each notebook trains only on its own dataset.

| Column | Description |
|---|---|
| `id` | News id |
| `news_url` | Article URL — the notebooks extract its **domain** (e.g. `people.com`) |
| `title` | Headline — the main text input |
| `tweet_ids` | Tab-separated ids of tweets that shared the article (not used) |

The fake / real label comes from which file a row is in (`*_fake.csv` vs `*_real.csv`).
These files are the "minimal" version of FakeNewsNet from
[KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet/tree/master/dataset).

### PolyglotFakeFacts

Multilingual fake / real news in 18 languages (Arabic, Armenian, Azerbaijani, Bulgarian, Czech,
English, Finnish, French, Georgian, German, Hungarian, Italian, Lithuanian, Romanian, Russian,
Slovak, Spanish, Swedish). Fake articles come from outlets flagged by EUvsDisinfo; real articles
come from official / editorially credible outlets. Source: Mendeley Data, DOI `10.17632/gff8bmr4ff`.

| File | Rows | Used for |
|---|---|---|
| `Fake.xlsx` | 4,912 (all fake) | Full fake portion of the dataset (not read by the notebook) |
| `Real.xlsx` | 5,294 (all real) | Full real portion of the dataset (not read by the notebook) |
| `80.xlsx` | ~8,165 | Official 80 % split → training + validation |
| `20.xlsx` | 2,041 (982 fake / 1,059 real) | Official 20 % split → held-out test set |

Columns in every file: `gathering date`, `news date`, `url`, `domain`, `language`, `keywords`,
`news headline`, `news original text`, `english translated version`, `label`.

Column names are lower-cased and non-alphanumerics replaced with `_` before use:

| Column | Used for |
|---|---|
| `label` | Target — `fake`/`false`/`0` → 0, `real`/`true`/`1` → 1 |
| `news_headline` | `statement:` part of the model input |
| `english_translated_version` | `context:` part (first 1,800 characters) |
| `news_original_text` | Fallback for `context:` when there is no translation |
| `keywords` | `subject:` part + metadata feature |
| `domain` / `url` | `source domain:` part + metadata feature |
| `language` | `language:` part + metadata feature |

---

## Label convention

Every notebook uses the same convention:

| Value | Meaning |
|---|---|
| **0** | **Fake** |
| **1** | **Real / True** |

For LIAR binary classification the six verdicts are collapsed as:
`pants-fire`, `false`, `barely-true` → **0 (fake)** and `half-true`, `mostly-true`, `true` → **1 (real)**.
For 6-class classification: `pants-fire`=0, `false`=1, `barely-true`=2, `half-true`=3,
`mostly-true`=4, `true`=5.

---

## Part 1 — ISOT: DL vs ML and the stacked ensemble

Notebooks: `notebooks/01_ISOT/`

### 1.1 Exploration

- Loaded `True.csv` (label 1) and `Fake.csv` (label 0) and concatenated them (44,898 rows).
- Class-balance count plot, word clouds for true and fake text, subject frequency bar plot.
- Checked nulls (none) and removed **209 duplicate rows** → 44,689 articles.
- Shuffled (`random_state=42`) and kept only `text` + `label`.

### 1.2 Pre-processing pipeline

1. **Stop-word removal** with the NLTK English list.
2. **Lemmatisation** with spaCy `en_core_web_sm` (`nlp.pipe`, batch 100, 3 processes).
3. **Normalisation** (`normlize()`):
   - lower-casing
   - collapsing repeated characters (`"sooo"` → `"so"`)
   - removing digits
   - removing punctuation
   - collapsing multiple spaces

### 1.3 Train / test split and vectorisation

| Models | Split | Vectoriser |
|---|---|---|
| DL | 75 / 25 (`train_test_split`, 33,516 / 11,173) — 20 % of train used for validation during `fit` | Keras `Tokenizer(num_words=10000)`, padded/truncated to `max_len=300` |
| ML | 75 / 25 (`random_state=42`) | `TfidfVectorizer(max_features=300)`; RF2 uses `CountVectorizer(max_features=300)` |

### 1.4 Deep-learning architectures

All use `Embedding(10000, 128, input_length=300)`, Adam, binary cross-entropy and `EarlyStopping(monitor="val_loss", patience=2)`.

| Model | Layers | Training |
|---|---|---|
| Uni-LSTM | Embedding → LSTM(64, seq) → LSTM(32) → Dense(64, relu) → Dropout(0.5) → Dense(1, sigmoid) | lr 1e-4, batch 30, ≤10 epochs |
| Bi-LSTM | Embedding → BiLSTM(64, seq) → BiLSTM(32) → Dense(64, relu) → Dropout(0.5) → Dense(1) | lr 1e-4, batch 30, ≤10 epochs |
| Uni-GRU | Embedding → GRU(64, seq) → GRU(32) → Dense(32, relu) → Dropout(0.5) → Dense(1) | lr 1e-4, batch 30, ≤10 epochs |
| Bi-GRU | Embedding → BiGRU(64, seq) → BiGRU(32) → Dense(32, relu) → Dropout(0.5) → Dense(1) | lr 1e-4, batch 30, ≤10 epochs |
| CNN | Embedding → Conv1D(128, kernel 5, relu) → GlobalMaxPooling1D → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid) | Adam default lr, batch 64, ≤10 epochs, restore best weights |

### 1.5 Machine-learning models

| Model | Parameters | Features |
|---|---|---|
| SVM | `SVC(C=1)` | TF-IDF |
| Logistic Regression | `C=1, max_iter=1000` | TF-IDF |
| Decision Tree | `max_depth=5` | TF-IDF |
| KNN | `n_neighbors=9` | TF-IDF |
| Random Forest 1 | `n_estimators=400, max_depth=40` | TF-IDF |
| Random Forest 2 | `n_estimators=300, max_depth=40` | Term frequency (CountVectorizer) |

### 1.6 Stacked ensemble (`03_stacked_ensemble.ipynb`)

The binary predictions of the **6 ML models** and the **5 DL models** on the test set are
concatenated into an 11-column matrix and passed to a meta-classifier:

```python
RandomForestClassifier(n_estimators=300, max_depth=40, random_state=42)
```

A combined comparison table, a grouped bar chart of Accuracy / Precision / Recall / F1 for every
model, and the confusion matrix of the stacked model are produced at the end.
(Please read the [evaluation notes](#evaluation-notes-and-known-limitations) for how this
stacked score should be interpreted.)

### 1.7 ISOT results

Test set = 11,173 articles. Precision / recall / F1 are for the positive class (1 = real).

**`01_bilstm_bigru_baseline.ipynb`**

| Model | Train acc | Test acc |
|---|---|---|
| Bi-LSTM | 0.9986 | 0.9948 |
| Bi-GRU | 0.9989 | 0.9945 |

**`02_dl_vs_ml_comparison.ipynb`**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Bi-LSTM | 0.9960 | 0.9932 | 0.9983 | 0.9958 |
| Bi-GRU | 0.9950 | 0.9956 | 0.9938 | 0.9947 |
| CNN | 0.9970 | 0.9964 | 0.9974 | 0.9969 |
| SVM | 0.9938 | 0.9930 | 0.9940 | 0.9935 |
| Logistic Regression | 0.9842 | 0.9823 | 0.9843 | 0.9833 |
| Decision Tree | 0.9940 | 0.9901 | 0.9974 | 0.9937 |
| KNN | 0.8618 | 0.8771 | 0.8241 | 0.8497 |
| Random Forest 1 | 0.9973 | 0.9975 | 0.9968 | 0.9972 |
| Random Forest 2 | **0.9976** | 0.9974 | 0.9975 | **0.9975** |

**`03_stacked_ensemble.ipynb`**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Uni-LSTM | 0.9616 | 0.9624 | 0.9564 | 0.9594 |
| Bi-LSTM | 0.9923 | 0.9990 | 0.9847 | 0.9918 |
| Uni-GRU | 0.9736 | 0.9921 | 0.9519 | 0.9716 |
| Bi-GRU | 0.9872 | 0.9979 | 0.9751 | 0.9863 |
| CNN | 0.9975 | 0.9974 | 0.9974 | 0.9974 |
| SVM | 0.9938 | 0.9930 | 0.9940 | 0.9935 |
| Logistic Regression | 0.9842 | 0.9823 | 0.9843 | 0.9833 |
| Decision Tree | 0.9939 | 0.9899 | 0.9974 | 0.9936 |
| KNN | 0.8618 | 0.8771 | 0.8241 | 0.8497 |
| Random Forest 1 | 0.9974 | 0.9975 | 0.9970 | 0.9973 |
| Random Forest 2 | 0.9976 | 0.9974 | 0.9975 | 0.9975 |
| **Stacked ensemble (RF meta)** | **0.9985** | 0.9987 | 0.9981 | 0.9984 |

Observations: bidirectional recurrent models clearly beat their unidirectional versions
(Bi-LSTM 99.23 % vs Uni-LSTM 96.16 %), the CNN is the strongest single DL model, and the tree
ensembles are the strongest ML models even on only 300 features. KNN is the weakest model.

---

## Part 2 — LIAR

### 2.1 `01_dl_ml_stacked_ensemble.ipynb` — the ISOT pipeline on LIAR

What was done:

1. Concatenated `valid`, `train` and `test` (12,836 statements), renamed the 14 columns, explored
   missing values and the label distribution.
2. Built two targets: **binary** (fake/real) and **6-class**.
3. Pre-processed the `Statement` only: stop-word removal → spaCy lemmatisation → normalisation
   (same as ISOT). Metadata columns were dropped.
4. Keras tokenizer (`num_words=10000`, `<OOV>` token), `max_len=300`, `embed_size=100`;
   stratified 80 / 20 split for DL models.
5. Trained **Uni-LSTM, Bi-LSTM, Uni-GRU, Bi-GRU, CNN** for both binary (sigmoid) and 6-class
   (softmax) targets — lr 1e-4, batch 30, ≤5 epochs, early stopping with best-weight restore.
6. Trained the **6 ML models** (TF-IDF / TF, 300 features, 75 / 25 split) for both targets.
7. **Stacked ensemble**: the 75 / 25 test portion is split in half (stratified); the RF meta-model
   is trained on the predictions for one half and evaluated on the other half (1,605 statements).

Binary results:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Uni-LSTM | 0.5584 | 0.5584 | 1.0000 | 0.7166 |
| Bi-LSTM | 0.5927 | 0.6545 | 0.5732 | 0.6112 |
| Uni-GRU | 0.5584 | 0.5584 | 1.0000 | 0.7166 |
| Bi-GRU | 0.6036 | 0.6185 | 0.7573 | 0.6809 |
| CNN | 0.6114 | 0.6127 | 0.8264 | 0.7037 |
| SVM | 0.6248 | 0.6396 | 0.7596 | 0.6944 |
| Logistic Regression | 0.6217 | 0.6416 | 0.7385 | 0.6866 |
| Decision Tree | 0.5921 | 0.5975 | 0.8368 | 0.6972 |
| KNN | 0.5575 | 0.6094 | 0.5891 | 0.5991 |
| Random Forest 1 | 0.6217 | 0.6244 | 0.8179 | 0.7082 |
| Random Forest 2 | 0.6248 | 0.6307 | 0.7996 | 0.7052 |
| **Stacked ensemble** | **0.7103** | 0.7080 | 0.8235 | **0.7614** |

6-class results (macro precision / recall / F1):

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Uni-LSTM | 0.2056 | 0.0343 | 0.1667 | 0.0568 |
| Bi-LSTM | 0.2313 | 0.1248 | 0.1953 | 0.1335 |
| Uni-GRU | 0.2056 | 0.0343 | 0.1667 | 0.0568 |
| Bi-GRU | 0.2504 | 0.1668 | 0.2162 | 0.1815 |
| CNN | 0.2395 | 0.1226 | 0.2006 | 0.1492 |
| SVM | 0.2527 | 0.2671 | 0.2235 | 0.2155 |
| Logistic Regression | 0.2568 | 0.2537 | 0.2331 | 0.2314 |
| Decision Tree | 0.2231 | 0.3022 | 0.1842 | 0.1244 |
| KNN | 0.2175 | 0.2192 | 0.2080 | 0.1988 |
| Random Forest 1 | 0.2586 | 0.2571 | 0.2260 | 0.2147 |
| Random Forest 2 | 0.2646 | 0.2890 | 0.2336 | 0.2204 |
| **Stacked ensemble** | **0.3028** | 0.2879 | 0.2872 | **0.2868** |

Observations: LIAR claims are only one sentence long, so the models that were near-perfect on ISOT
drop to ~60 % binary accuracy. Uni-LSTM and Uni-GRU collapse to predicting the majority class
(recall 1.0 in binary; one class in 6-class). Stacking still adds ~9 points of binary accuracy
over the best single model. These results motivated the transformer pipeline below.

### 2.2 `02_transformer_ensemble_gnn.ipynb` — transformers, credibility features and a GCN

This notebook uses the **official LIAR train / valid / test splits**.

**Model input.** Instead of the bare statement, each example becomes one string that contains the
claim and its metadata (no stop-word / punctuation removal, since transformers need natural text):

```
statement: <claim> subject: <topics> speaker: <speaker> job title: <job> state: <state>
party: <party> context: <venue> speaker_history: barely_true=… false=… half_true=… mostly_true=… pants_fire=…
```

**Transformer fine-tuning** (Hugging Face `Trainer`):

| Model | Learning rate | Epochs | Batch | Max length |
|---|---|---|---|---|
| `distilbert-base-uncased` | 3e-5 | 2 | 16 | 128 |
| `bert-base-uncased` | 3e-5 | 2 | 16 | 128 |
| `roberta-base` | 2e-5 | 2 | 16 | 128 |

Common settings: class-weighted cross-entropy (`compute_class_weight("balanced")`), weight decay
0.02, ~8 % warm-up, fp16, evaluation + checkpoint every epoch, best checkpoint chosen by validation
**macro-F1**, early stopping (patience 1), seed 42.

Models trained:

- **binary** — each transformer on train, validated on valid.
- **binary_refit** — each transformer on *(train − 8 % inner split) + original valid*; the 8 % inner
  split of train is used for early stopping.
- **multiclass** — each transformer on the 6-class target.
- **fake_subtype / real_subtype** — DistilBERT on {pants-fire, false, barely-true} and
  {half-true, mostly-true, true} separately (for the hierarchical model).

**Extra (non-transformer) components:**

| Component | What it does |
|---|---|
| TF-IDF baseline | Word 1–2-gram (90k) + char 3–5-gram (40k) TF-IDF → Logistic Regression (balanced, C=2) |
| LIAR credibility baseline | Speaker credit-history counts, their ratios, total, fake-ratio, real-ratio, credibility margin, text length, number of subjects **plus** smoothed target encoding (α = 50) of speaker, subject, party, state, context and job title → Logistic Regression |
| LIAR metadata GCN | Graph with one node per claim and one node per metadata value (speaker, each subject, party, state, context, job title); claim↔value edges + self-loops, symmetric normalised adjacency. Claim nodes carry the standardised credibility features, value nodes a one-hot type. 2-layer GCN (hidden 128, dropout 0.3), AdamW lr 0.01, ≤220 epochs, early stopping on validation macro-F1 (patience 25). |

**Ensembles:**

| Ensemble | How |
|---|---|
| Majority vote | Hard vote of all transformers (ties broken by average probability) |
| Highest-sum / average probability | Mean of the softmax probabilities |
| Validation-weighted average | Probabilities weighted by each model's validation macro-F1 |
| Validation-selected average | Exhaustive search over every subset of the 6 binary models; the subset with the best validation macro-F1 is used |
| Tuned threshold | P(fake) threshold searched on validation (0.05–0.95, 181 steps) to maximise accuracy |
| Stacked meta ensemble | Logistic Regression trained on the **validation** set using all transformer probabilities + their average + TF-IDF probabilities + credibility probabilities + credibility features, then applied to test |
| Transformer + GCN blend | `α·transformer_avg + (1−α)·GCN`, α tuned on validation (best α = 0.94) |
| Hierarchical (6-class) | Binary gate decides fake vs real (threshold tuned on validation), then the 6-class probabilities — or the dedicated subtype models — pick the fine-grained label |

All tables are written to `optimized_summaries_simple_liar_novel_v7/`.

**Binary results (test, 1,283 statements: 556 fake / 727 real):**

| Method | Accuracy | Macro-F1 | Fake recall | Real recall |
|---|---|---|---|---|
| **Ensemble: validation-selected avg prob** | **0.7646** | **0.7584** | 0.6978 | 0.8157 |
| Ensemble: validation-selected + tuned threshold | 0.7646 | 0.7584 | 0.6978 | 0.8157 |
| Ensemble: majority vote | 0.7607 | 0.7552 | 0.7050 | 0.8033 |
| Ensemble: highest-sum / avg prob | 0.7607 | 0.7552 | 0.7050 | 0.8033 |
| Ensemble: validation-weighted avg prob | 0.7599 | 0.7545 | 0.7050 | 0.8019 |
| Ensemble: transformer + LIAR metadata GCN | 0.7576 | 0.7524 | 0.7068 | 0.7964 |
| RoBERTa (refit) | 0.7537 | 0.7480 | 0.6960 | 0.7978 |
| RoBERTa | 0.7529 | 0.7486 | 0.7176 | 0.7799 |
| DistilBERT (refit) | 0.7521 | 0.7477 | 0.7140 | 0.7813 |
| DistilBERT | 0.7514 | 0.7466 | 0.7086 | 0.7840 |
| BERT (refit) | 0.7490 | 0.7405 | 0.6547 | 0.8212 |
| BERT | 0.7443 | 0.7412 | 0.7320 | 0.7538 |
| Ensemble: stacked transformer + credibility meta | 0.7436 | 0.7391 | 0.7068 | 0.7717 |
| Baseline: LIAR credibility | 0.7295 | 0.7258 | 0.7068 | 0.7469 |
| GCN: LIAR metadata graph | 0.7288 | 0.7263 | 0.7320 | 0.7263 |
| Baseline: TF-IDF + metadata | 0.6617 | 0.6545 | 0.5971 | 0.7111 |

**6-class results (test):**

| Method | Accuracy | Macro-F1 |
|---|---|---|
| **Ensemble: stacked transformer + credibility meta** | **0.4482** | **0.4546** |
| Ensemble: transformer + LIAR metadata GCN | 0.4482 | 0.4530 |
| Ensemble: specialised hierarchical subtype | 0.4451 | 0.4440 |
| BERT | 0.4443 | 0.4499 |
| Ensemble: highest-sum / avg prob | 0.4419 | 0.4461 |
| Ensemble: hierarchical binary + six-class | 0.4419 | 0.4447 |
| Ensemble: majority vote | 0.4412 | 0.4465 |
| Ensemble: validation-weighted avg prob | 0.4396 | 0.4438 |
| GCN: LIAR metadata graph | 0.4380 | 0.4427 |
| DistilBERT | 0.4170 | 0.4194 |
| Ensemble: validation-selected avg prob | 0.4162 | 0.4196 |
| Baseline: LIAR credibility | 0.4123 | 0.4206 |
| Baseline: TF-IDF + metadata | 0.3001 | 0.3068 |
| RoBERTa | 0.2081 | 0.0574 |

Observations: adding speaker metadata and credit history to the input and fine-tuning
transformers lifts LIAR binary accuracy from ~62 % (TF-IDF ML models on the statement alone) to
**76.5 %**, and 6-class accuracy from ~26 % to **44.8 %**. RoBERTa failed to converge on the
6-class task in this run (it predicts a single class), which is why it hurts plain averaging.

---

## Part 3 — Cross-dataset transformer ensembles

Notebooks: `notebooks/03_cross_dataset_transformers/`

The three notebooks share the same "v7" code; only the loader differs. Each one trains
**fresh** DistilBERT, BERT and RoBERTa models on its own dataset (no weights are shared with LIAR).

### 3.1 Data preparation

| Dataset | Model input text | Split |
|---|---|---|
| GossipCop / PolitiFact | `statement: <title> context: source domain <domain from news_url>` | Duplicate texts removed, then stratified **70 / 15 / 15** train / valid / test (seed 42); texts appearing in an earlier split are removed from later splits |
| PolyglotFakeFacts | `statement: <headline> context: <English translation or original text, first 1,800 chars> subject: <keywords> source domain: <domain> language: <language>` | Official split: `80.xlsx` → 85 % train / 15 % valid, `20.xlsx` → test; cross-split duplicates removed |

Resulting split sizes:

| Dataset | Train (fake / real) | Valid (fake / real) | Test (fake / real) |
|---|---|---|---|
| GossipCop | 14,544 (3,433 / 11,111) | 3,117 (736 / 2,381) | 3,117 (736 / 2,381) |
| PolitiFact | 691 (300 / 391) | 148 (64 / 84) | 149 (65 / 84) |
| PolyglotFakeFacts | 6,259 (3,340 / 2,919) | 1,105 (590 / 515) | 1,801 (982 / 819) |

### 3.2 Models and ensembles

- **Transformers:** DistilBERT (lr 3e-5), BERT (lr 3e-5), RoBERTa (lr 2e-5); 2 epochs, batch 16,
  max length 128, weight decay 0.01, class-weighted loss, best checkpoint by validation macro-F1,
  early stopping (patience 2), fp16.
- **TF-IDF baseline:** word 1–2-gram (50k) + char 3–5-gram (50k) → Logistic Regression (balanced, C=2).
- **Dataset-metadata baseline:** smoothed target encoding (α = 30) of available metadata
  (`domain`, `source_file`, `language`, `keywords`, …) + text / domain lengths → Logistic Regression.
- **Ensembles:** majority vote, average probability, validation-weighted average, tuned P(fake)
  threshold, and a stacked Logistic-Regression meta-model trained on the validation-set
  probabilities of all transformers + TF-IDF + metadata baseline.

Each run writes report tables to `external_summaries_v7/<dataset>/`
(`binary_results_table.csv`, `threshold_tuning_table.csv`, `validation_model_weights.csv`,
per-class precision / recall / F1 tables, and `binary_test_predictions.csv`) and plots a
confusion matrix for every method.

### 3.3 Results (test set)

**GossipCop** (3,117 headlines):

| Method | Accuracy | Macro-F1 | Fake recall | Real recall |
|---|---|---|---|---|
| **Ensemble: tuned accuracy threshold** | **0.8762** | 0.8085 | 0.5965 | 0.9626 |
| RoBERTa | 0.8643 | **0.8151** | 0.7378 | 0.9034 |
| Ensemble: majority vote | 0.8624 | 0.8133 | 0.7405 | 0.9000 |
| Baseline: TF-IDF | 0.8624 | 0.8128 | 0.7364 | 0.9013 |
| Ensemble: average prob | 0.8620 | 0.8138 | 0.7473 | 0.8975 |
| Ensemble: validation-weighted avg prob | 0.8620 | 0.8138 | 0.7473 | 0.8975 |
| BERT | 0.8553 | 0.8065 | 0.7473 | 0.8887 |
| DistilBERT | 0.8534 | 0.8013 | 0.7228 | 0.8937 |
| Baseline: dataset metadata ⚠ | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Ensemble: stacked transformer + metadata meta ⚠ | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**PolitiFact** (149 headlines):

| Method | Accuracy | Macro-F1 | Fake recall | Real recall |
|---|---|---|---|---|
| **Baseline: TF-IDF** | **0.9195** | **0.9178** | 0.8923 | 0.9405 |
| BERT | 0.9060 | 0.9037 | 0.8615 | 0.9405 |
| Ensemble: majority vote | 0.8926 | 0.8890 | 0.8154 | 0.9524 |
| Ensemble: average prob | 0.8926 | 0.8890 | 0.8154 | 0.9524 |
| Ensemble: validation-weighted avg prob | 0.8926 | 0.8890 | 0.8154 | 0.9524 |
| Ensemble: tuned accuracy threshold | 0.8926 | 0.8884 | 0.8000 | 0.9643 |
| RoBERTa | 0.8859 | 0.8811 | 0.7846 | 0.9643 |
| DistilBERT | 0.8725 | 0.8691 | 0.8154 | 0.9167 |
| Baseline: dataset metadata ⚠ | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Ensemble: stacked transformer + metadata meta ⚠ | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**PolyglotFakeFacts** (1,801 articles):

| Method | Accuracy | Macro-F1 | Fake recall | Real recall |
|---|---|---|---|---|
| Ensemble: stacked transformer + metadata meta | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Baseline: dataset metadata | 0.9994 | 0.9994 | 0.9990 | 1.0000 |
| Baseline: TF-IDF | 0.9961 | 0.9961 | 0.9949 | 0.9976 |
| Ensemble: tuned accuracy threshold | 0.9717 | 0.9714 | 0.9857 | 0.9548 |
| Ensemble: majority vote | 0.9711 | 0.9708 | 0.9827 | 0.9573 |
| Ensemble: average prob | 0.9711 | 0.9708 | 0.9827 | 0.9573 |
| Ensemble: validation-weighted avg prob | 0.9706 | 0.9703 | 0.9817 | 0.9573 |
| BERT | 0.9650 | 0.9647 | 0.9674 | 0.9621 |
| RoBERTa | 0.9628 | 0.9624 | 0.9715 | 0.9524 |
| DistilBERT | 0.9622 | 0.9618 | 0.9786 | 0.9524 |

⚠ = affected by label leakage, see the next section.

---

## Results summary

Best test accuracy per dataset (excluding the rows marked ⚠):

| Dataset | Task | Best method | Accuracy | Macro-F1 |
|---|---|---|---|---|
| ISOT | Binary | Random Forest 2 (single model) / CNN | 0.9976 / 0.9975 | 0.9975 (F1) |
| ISOT | Binary | Stacked RF ensemble (see note 1) | 0.9985 | 0.9984 (F1) |
| LIAR | Binary | Classic stacked ensemble (statement only) | 0.7103 | 0.70 |
| LIAR | Binary | Transformer ensemble (validation-selected) | **0.7646** | 0.7584 |
| LIAR | 6-class | Stacked transformer + credibility meta | **0.4482** | 0.4546 |
| GossipCop | Binary | Transformer ensemble + tuned threshold | 0.8762 | 0.8085 |
| PolitiFact | Binary | TF-IDF + Logistic Regression | 0.9195 | 0.9178 |
| PolyglotFakeFacts | Binary | TF-IDF + Logistic Regression (see note 4) | 0.9961 | 0.9961 |

---

## Evaluation notes and known limitations

These points matter when quoting the numbers above (for example in a report or paper):

1. **ISOT stacked ensemble is scored on the data it was trained on.** In
   `01_ISOT/03_stacked_ensemble.ipynb` the meta-classifier is fitted with
   `meta_clf.fit(meta_input, y_test)` and then evaluated on the same `meta_input`, so 0.9985 is a
   training score. In addition, the DL models use a split made **without** `random_state` while the
   ML models use `random_state=42`, so the ML prediction columns are for different articles than
   `y_test`. A fair version trains the meta-model on out-of-fold (or validation) predictions from a
   single shared split and scores it on a held-out part — as done in the LIAR notebook (`run_stacked_ensemble`).
2. **ISOT is an "easy" dataset.** Every real article is from Reuters and most begin with a
   `CITY (Reuters) -` dateline, so near-perfect scores partly reflect source style rather than
   veracity. The `subject` column alone also separates the classes (it is correctly dropped).
3. **GossipCop / PolitiFact metadata leakage (rows marked ⚠).** The dataset-metadata baseline
   target-encodes `source_file`, which is `*_fake.csv` or `*_real.csv` — i.e. the label itself.
   That is why the metadata baseline and the stacked meta-ensemble reach exactly 100 %. Removing
   `"source_file"` from `DATASET_METADATA_CANDIDATES` fixes it. The transformer, TF-IDF and
   vote/average ensemble rows do **not** use `source_file` and are valid.
4. **PolyglotFakeFacts source-domain shortcut.** `source_file` does not leak here (train rows come
   from `80.xlsx` and test rows from `20.xlsx`), but the `domain` is included both in the text and in
   the metadata features. Because fake and real articles come from different sets of outlets, the
   domain nearly determines the label, which explains the ~99.9 % TF-IDF / metadata scores.
   Transformer-only results (~96–97 %) are the better measure of content-based detection.
5. **LIAR classic notebook split.** The official splits are merged and re-split randomly, and the DL
   models (80 / 20 split) and the stacked-ensemble test portion (75 / 25 split) are different
   random splits, so some statements scored by the stack were seen by the DL models during training.
6. **LIAR metadata.** The speaker credit-history counts are strong features; it has been reported
   in the literature that these counts can include the verdict of the statement being classified,
   so metadata-heavy LIAR results may be optimistic compared with statement-only results.
7. **Single runs.** Every result comes from one run with seed 42 (Keras runs are unseeded); small
   differences between models (< 0.5 %) are within run-to-run variation.

---

## How to run

### 1. Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

A CUDA GPU is strongly recommended for the transformer notebooks (they were run on an RTX 2060,
6 GB, with fp16).

### 2. Point the notebooks at `datasets/`

The notebooks still contain the paths from the machines they were run on. Change these lines:

**`notebooks/01_ISOT/*.ipynb`** (cell that reads the CSVs, originally Google Drive paths):

```python
true_data = pd.read_csv('../../datasets/ISOT/True.csv')
fake_data = pd.read_csv('../../datasets/ISOT/Fake.csv')
```

**`notebooks/02_LIAR/01_dl_ml_stacked_ensemble.ipynb`**:

```python
train_data = pd.read_csv('../../datasets/LIAR/train.csv')
test_data  = pd.read_csv('../../datasets/LIAR/test.csv')
valid_data = pd.read_csv('../../datasets/LIAR/valid.csv')
```

**`notebooks/02_LIAR/02_transformer_ensemble_gnn.ipynb`** (config cell):

```python
PROJECT_DIR = Path.cwd()                       # outputs are written next to the notebook
DATA_DIR = Path("../../datasets/LIAR")
```

**`notebooks/03_cross_dataset_transformers/*.ipynb`** (config cell):

```python
PROJECT_DIR = Path.cwd()
DATA_DIR = Path("../../datasets/GossipCop")            # gossipcop_transformer_ensemble.ipynb
# DATA_DIR = Path("../../datasets/PolitiFact")         # politifact_transformer_ensemble.ipynb
# DATA_DIR = Path("../../datasets/PolyglotFakeFacts")  # polyglotfakefacts_transformer_ensemble.ipynb
```

If you use Google Colab, upload the `datasets/` folder to Drive and use the Drive path instead.

### 3. Useful switches in the transformer notebooks

| Variable | Meaning |
|---|---|
| `FORCE_RETRAIN` | `True` retrains from scratch; `False` reuses saved models in the results folder |
| `TRAIN_MODELS` / `RUN_*_TRAINING` | Skip training and only evaluate saved models |
| `MAX_ROWS_PER_CLASS` | Set to e.g. `500` for a quick debug run; `None` for the full data |
| `OPTIONAL_SLOW_MODELS` | Larger models (e.g. `roberta-large`, `albert-large-v2`) that can be added to the model list |

---

## Citations

If you use these datasets, please cite the original authors:

- **ISOT** — H. Ahmed, I. Traore, S. Saad. *Detection of Online Fake News Using N-Gram Analysis and
  Machine Learning Techniques.* ISDDC 2017; and *Detecting opinion spams and fake news using text
  classification.* Security and Privacy, 2018. (University of Victoria ISOT Research Lab)
- **LIAR** — W. Y. Wang. *"Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News
  Detection.* ACL 2017.
- **FakeNewsNet (GossipCop, PolitiFact)** — K. Shu, D. Mahudeswaran, S. Wang, D. Lee, H. Liu.
  *FakeNewsNet: A Data Repository with News Content, Social Context and Spatiotemporal Information
  for Studying Fake News on Social Media.* arXiv:1809.01286, 2018.
  Data: <https://github.com/KaiDMML/FakeNewsNet>
- **PolyglotFakeFacts** — *PolyglotFakeFacts: A multilingual dataset of fake and real news*,
  Mendeley Data, DOI `10.17632/gff8bmr4ff`.
- **Transformers** — BERT (Devlin et al., 2019), DistilBERT (Sanh et al., 2019),
  RoBERTa (Liu et al., 2019), via Hugging Face Transformers.
