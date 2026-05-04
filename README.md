# ESG Report Greenwashing Detector
---
SRM Institute of Science and Technology
BTech CSE CORE B, 4th Year
BT26COREB007

- KABIR SHARMA (RA2211003030071)

- JOBSON K JOBY (RA2211003030107)

- UTKARSH GUPTA (RA2211003030116)

- HARSH TIWARI (RA2211003030079)


A multi-engine NLP pipeline that analyzes corporate ESG / BRSR reports and produces a **composite greenwashing risk score** by combining lexicon-based analysis, ClimateBERT transformer models, and LLM-powered scrutiny.

---


## Key Features

- **PDF & TXT ingestion** — Extract and clean text from ESG reports in PDF or pre-cleaned `.txt` format
- **Synonym-expanded keyword engine** — Uses WordNet to automatically expand ESG, greenwashing, and substantive keyword lists
- **ESG Pillar Breakdown** — Calculates keyword distribution across Environmental, Social, and Governance pillars
- **Greenwashing detection** — Measures density and ratio of aspirational vs. substantive language
- **ClimateBERT sentence-level analysis** — Runs 3 fine-tuned RoBERTa models (sentiment, commitment, specificity) + hedging detection on up to 600 sentences
- **LLM-powered critique** — Sends the full report to an LLM via OpenRouter for a "harsh critic" assessment with structured JSON output
- **Weighted composite score** — Combines all three engines: `0.3 × Lexicon + 0.3 × BERT + 0.5 × LLM`
- **TF-IDF cross-company comparison** — Compare ESG focus across multiple companies in the same industry
- **Red flags & reasoning** — Extracts specific red flags and a final assessment from the LLM engine

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/esg-greenwashing-detector.git
cd esg-greenwashing-detector
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install PyPDF2 nltk pandas numpy matplotlib seaborn scikit-learn wordcloud openpyxl
pip install transformers torch   # For ClimateBERT (Engine 2)
pip install openai               # For LLM critique (Engine 3)
```

### 4. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
```

### 5. Set Up OpenRouter API Key

For Engine 3 (LLM critique), you need an [OpenRouter](https://openrouter.ai/) API key:

```python
key = "sk-or-v1-your-api-key-here"
```


---
## Description

Corporate sustainability reports (ESG, BRSR, CSR) often contain a mix of genuine commitments and aspirational language designed to project a greener image than reality warrants — a practice known as **greenwashing**. This project automates the detection of greenwashing risk by analyzing the full text of a company's ESG report through three independent scoring engines and producing a single, weighted composite risk score.


---
## Usage

### Quick Start (Single Report)

Open the notebook in Jupyter or Google Colab and update the configuration cell:

```python
# ── Configuration ──
PDF_FOLDER = ""                    # Leave empty if file is in the working directory
INDUSTRY_NAME = "FMCG"            # Industry for context-aware analysis
txt_path = "HUL.txt"              # Path to pre-cleaned .txt report
company_name = "HUL"              # Display name
controversy_level = "Unknown"     # "Low", "Medium", "High", or "Unknown"
```

Then run all cells sequentially. The pipeline will:

1. **Load & clean** the report text
2. **Expand keywords** via WordNet and save to `expanded_esg_keywords.txt`
3. **Run Engine 1** (Lexicon) → greenwashing ratio, density, ESG pillar breakdown
4. **Run Engine 2** (ClimateBERT) → sentence-level sentiment, commitment, specificity + hedging
5. **Run Engine 3** (LLM) → structured JSON critique with red flags
6. **Compute the final weighted score**

### Batch Analysis (Multiple Reports)

To analyze multiple companies in one industry:

```python
COMPANY_INFO = {
    "company_a.txt": ("Company A", "Low"),
    "company_b.txt": ("Company B", "Medium"),
    "company_c.txt": ("Company C", "High"),
}

results_df = analyze_industry_reports(PDF_FOLDER, INDUSTRY_NAME, COMPANY_INFO)
```

Use `tfidf_esg_analysis()` to compare ESG focus across companies:

```python
texts_dict = {
    "Company A": text_a,
    "Company B": text_b,
}
tfidf_df = tfidf_esg_analysis(texts_dict)
```

---

## Architecture & Methodology

### Pipeline Overview

```
┌─────────────────────────┐
│   ESG Report (PDF/TXT)  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Text Extraction &      │
│  Cleaning (PyPDF2 +     │
│  ASCII normalization)   │
└────────────┬────────────┘
             │
     ┌───────┼───────┐
     │       │       │
     ▼       ▼       ▼
┌─────────┐┌─────────┐┌─────────┐
│ Engine 1 ││ Engine 2 ││ Engine 3 │
│ Lexicon  ││ Climate- ││   LLM   │
│ Analysis ││  BERT    ││ Critique │
│          ││          ││          │
│ GW ratio ││ Sentiment││ Scrutiny │
│ GW dens. ││ Commitm. ││ Confid.  │
│          ││ Specific.││ Red flags│
│          ││ Hedging  ││          │
└────┬─────┘└────┬─────┘└────┬─────┘
     │           │           │
     ▼           ▼           ▼
   w₁=0.3     w₂=0.3     w₃=0.5
     │           │           │
     └───────────┼───────────┘
                 ▼
     ┌───────────────────────┐
     │   Final Composite     │
     │   Greenwashing Score  │
     │       (0 – 1)         │
     └───────────────────────┘
```

---

### Engine 1 — Lexicon-Based Analysis

- **Keyword expansion** — Base ESG, greenwashing-indicator, and substantive-action word lists are expanded via WordNet synonyms (~138 greenwashing terms, ~149 substantive terms)
- **Greenwashing density** — `(aspirational word count / total words) × 1000`
- **Greenwashing ratio** — `aspirational word count / substantive word count`
- **Risk classification** — `High` if ratio > 1.5 or density > 18; `Medium` if ratio > 1 or density > 10; else `Low`
- **Lexicon score** — `0.65 × GW_ratio + 0.35 × GW_density`, normalized to [0, 1]

**ESG Pillar Analysis** also runs in this engine, counting keyword mentions across Environmental, Social, and Governance categories to determine the report's dominant focus area.

---

### Engine 2 — ClimateBERT Transformer Analysis

Three fine-tuned **ClimateBERT** (DistilRoBERTa) models from Hugging Face are applied sentence-by-sentence to up to 600 sentences:

- **Sentiment** (`climatebert/distilroberta-base-climate-sentiment`) — Measures positive/opportunity framing vs. neutral/risk
- **Commitment** (`climatebert/distilroberta-base-climate-commitment`) — Detects whether language signals genuine commitment (`yes`/`no`)
- **Specificity** (`climatebert/distilroberta-base-climate-specificity`) — Classifies whether claims are specific (`spec`) or non-specific (`non`)

A **hedging detector** augments the transformers by counting phrases like *"aims to"*, *"strives for"*, *"on track to"* per sentence.

**BERT Greenwashing Risk** is computed as:

```
gw_bert_raw = 0.71 × avg_sentiment + 0.14 × avg_commitment
            − 0.86 × avg_specificity − 0.71 × avg_hedging

gw_bert_risk = clamp((gw_bert_raw + 2.0) / 4.0,  0, 1)
```

*Interpretation: High positive sentiment + low specificity + hedging → higher greenwashing risk.*

---

### Engine 3 — LLM Critique (OpenRouter)

The full report text is sent to an LLM (via [OpenRouter](https://openrouter.ai/)) with a system prompt instructing it to act as a *"harsh, no-nonsense business and sustainability critic"*. The LLM returns structured JSON:

```json
{
  "overall_verdict": "High Risk | Medium Risk | Low Risk | Clean",
  "scrutiny_confidence": 0.0 – 1.0,
  "deflection_analysis": "...",
  "language_analysis": "...",
  "key_red_flags": ["flag 1", "flag 2"],
  "final_take": "2-3 harsh sentences"
}
```

The `scrutiny_confidence` value (0–1) is used as the LLM engine's score.

---

### Final Composite Score

```python
finalized_score = 0.3 × lexicon_score + 0.3 × BERT_score + 0.5 × LLM_score
```

- **Lexicon (weight 0.3)** — Fast, interpretable baseline; limited to keyword counting
- **ClimateBERT (weight 0.3)** — Sentence-level semantic analysis; computationally expensive but robust
- **LLM (weight 0.5)** — Highest weight — captures nuance, context, and industry knowledge that statistical methods miss

> **Note:** Weights sum to 1.1 (not 1.0). This is by design in the current notebook and may be normalized in a future version. See [Limitations](#limitations--known-issues).

---

## Demo / Expected Output

When you run the notebook on a report (e.g., HUL's BRSR 2024-25), you will see:

**Lexicon Engine Output:**
```
Greenwashing:
  Risk: Medium
  Ratio: 0.42
  Aspirational words: 1523
  Substantive words: 3624

ESG Focus:
  Environmental: 28.3%
  Social: 41.2%
  Governance: 30.5%
  Dominant: Social
```

**ClimateBERT Engine Output:**
```
============================================================
CLIMATEBERT + HEDGING ANALYSIS RESULTS
============================================================
BERT Greenwashing Risk Score : 0.xxx / 1.0
Raw Score                    : x.xxx
Avg Positive Sentiment       : 0.xxx
Avg Commitment               : 0.xxx
Avg Specificity              : 0.xxx
Avg Hedging Ratio            : 0.xxx
Sentences Analyzed           : 600
============================================================
```

**LLM Engine Output:**
```json
{
  "overall_verdict": "Medium Risk",
  "scrutiny_confidence": 0.72,
  "deflection_analysis": "...",
  "language_analysis": "...",
  "key_red_flags": ["flag 1", "flag 2", "flag 3"],
  "final_take": "..."
}
```

**Final Score:**
```
Finalized Greenwashing Score: 0.XX / 1.0
```

---

## Tech Stack & Dependencies

- **PyPDF2** — Extract text from PDF reports
- **nltk** — Tokenization, lemmatization, stopwords, WordNet synonyms
- **pandas / numpy** — DataFrames and numerical operations
- **matplotlib / seaborn / wordcloud** — Charts and word clouds
- **scikit-learn** — TF-IDF vectorization for cross-company comparison
- **transformers / torch** — ClimateBERT pipeline inference
- **openai** — OpenRouter API client for LLM critique
- **openpyxl** — (Optional) Export results to `.xlsx`

**Python version:** 3.12+ recommended (developed on 3.12/3.13)

---

## Project Structure

```
esg-greenwashing-detector/
│
├── ESG_Report_Sentiment_Analysis_complete_model.ipynb   # Main analysis notebook
├── README.md                                             # This file
├── expanded_esg_keywords.txt                             # Auto-generated keyword file
│
├── *.txt                                                 # Pre-cleaned ESG report files
│                                                         #   (e.g., HUL.txt)
│
└── *.pdf                                                 # Raw PDF ESG reports
                                                          #   (optional — if using PDF ingestion)
```

### Key Functions

- `extract_text_from_pdf(pdf_path)` — Extracts raw text from a PDF using PyPDF2
- `clean_text(text)` — Normalizes whitespace, strips non-ASCII, keeps essential punctuation
- `get_synonyms(word)` — Retrieves WordNet synonyms for keyword expansion
- `expand_keywords_with_synonyms(keywords_list)` — Expands a keyword list with up to N synonyms per word
- `detect_greenwashing(text)` — Computes greenwashing density, ratio, and risk level (Engine 1)
- `calculate_esg_importance(text, company_name)` — Calculates ESG pillar keyword distribution
- `tfidf_esg_analysis(texts_dict)` — TF-IDF comparison of ESG focus across multiple companies
- `analyze_single_esg_report(path, name, industry)` — Full single-report analysis pipeline
- `analyze_industry_reports(folder, industry, info)` — Batch analysis across multiple reports
- `load_keywords_from_file(filename)` — Loads keyword lists from the generated `.txt` file
- `count_hedging(text, hedging_list)` — Counts hedging phrases per sentence (Engine 2 helper)
- `analyze_report(clean_text)` — ClimateBERT + hedging full analysis (Engine 2)

---

## Configuration & Customization

### Modifying Keyword Lists

The base keywords are defined in `BASE_ESG_KEYWORDS`, `GREENWASHING_INDICATORS`, and `SUBSTANTIVE_WORDS` dictionaries within the notebook. You can:

- **Add industry-specific terms** (e.g., for Oil & Gas: `"flaring"`, `"methane leak"`)
- **Adjust synonym expansion depth** via `max_synonyms_per_word` (default: 2)
- **Load keywords from file** by editing `expanded_esg_keywords.txt`

### Adjusting Scoring Weights

The final composite score weights can be modified:

```python
w1 = 0.3   # Lexicon weight
w2 = 0.3   # ClimateBERT weight
w3 = 0.5   # LLM weight
finalized_score = w1 * lexicon_score + w2 * BERT_score + w3 * LLM_score
```

### Changing the LLM Model

The default model is `stepfun/step-3.5-flash:free` via OpenRouter. You can swap this for any model available on the platform:

```python
completion = client.chat.completions.create(
    model="google/gemini-2.0-flash-exp:free",  # Example alternative
    ...
)
```

---

## Limitations & Known Issues

- **Global variable `text`** — The `analyze_single_esg_report` function sets a `global text` variable, which can cause issues in concurrent/parallel execution. This should be refactored for production use.
- **Weight sum > 1.0** — Engine weights (0.3 + 0.3 + 0.5 = 1.1) do not sum to 1.0. The final score can exceed 1.0 for high-risk reports. Consider normalizing.
- **PDF extraction quality** — `PyPDF2` struggles with complex PDF layouts (tables, multi-column, image-heavy). For best results, use pre-cleaned `.txt` files.
- **ClimateBERT speed** — Processing 600 sentences through 3 transformer models is slow on CPU (~30–60 min). Use a GPU runtime (Google Colab T4/A100) for faster inference.
- **API key in notebook** — The OpenRouter API key is hardcoded in the notebook. Use environment variables in production.
- **LLM rate limits** — Free-tier OpenRouter models may have rate limits or availability issues. The notebook's LLM cell may error with `KeyboardInterrupt` or `404` if the model endpoint changes.

---


