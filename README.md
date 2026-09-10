# Hiver AI Support Agent

An intelligent, grounded customer support agent built using real-world customer support interaction data (Customer Support on Twitter dataset). The system classifies incoming queries into brand-specific intents, retrieves relevant historical resolutions, generates grounded responses, and makes explicit, auditable escalation decisions.

---

## Current Development Status

**Status:** Phase 1 Scaffolding Completed.

- Repository structure and modular architecture established.
- Configuration, baseline interfaces, evaluation harness skeletons, and CLI scripts initialized.
- No model weights, production pipelines, synthetic evaluation results, or hardcoded secrets have been deployed yet.
- Awaiting raw dataset schema inspection to drive preprocessing and intent taxonomy derivation.

---

## Planned Architecture

The agent is designed around a 4-stage modular pipeline:

```
                  +--------------------------+
                  | Incoming Customer Query  |
                  +-------------+------------+
                                |
                                v
               +--------------------------------+
               | 1. Intent Classifier           |
               | (Predict intent & confidence)  |
               +---------------+----------------+
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
+-----------------------+            +-------------------------+
| 2. Case Retriever     |            | 4. Escalation Engine    |
| (Retrieve historical  |----------->| (Decision: AUTO_HANDLE  |
| brand resolutions)    |            |  or ESCALATE + reason)  |
+-----------+-----------+            +------------+------------+
            |                                     |
            v                                     |
+-----------------------+                         |
| 3. Response Generator |                         |
| (Grounded generation  |                         |
| using brand evidence) |                         |
+-----------+-----------+                         |
            |                                     |
            +------------------+------------------+
                               |
                               v
                  +--------------------------+
                  |  Final Agent Output      |
                  |  - Predicted Intent      |
                  |  - Grounded Response     |
                  |  - Decision & Reason     |
                  |  - Supporting Evidence   |
                  +--------------------------+
```

1. **Intent Classification**: Classifies customer queries into brand-specific support intents derived from empirical data analysis.
2. **Resolution Retrieval**: Indexes historical dialogues to retrieve top-k similar resolutions, providing verifiable evidence.
3. **Response Generation**: Generates brand-aligned support responses strictly grounded in the retrieved historical resolutions to minimize hallucinations.
4. **Escalation Engine**: Determines whether to `AUTO_HANDLE` the query or `ESCALATE` to a human agent, outputting an explicit decision reason and supporting context.

---

## Planned Evaluation Framework

The project includes an empirical, multi-tier evaluation system:

1. **Manually Labelled Golden Evaluation Set**:
   - 150–250 curated and verified customer-support interactions.
   - Ground truth labels for intent, escalation decision, and resolution validity.

2. **Benchmark Baselines**:
   - **Trivial Baseline**: Majority-class predictor and static canned response.
   - **Simple Baseline**: TF-IDF vectorization with logistic regression for intent and cosine similarity for resolution retrieval.

3. **Automated Evaluation Metrics**:
   - **Classification**: Precision, Recall, Macro-F1, Accuracy across intents.
   - **Escalation**: Precision, Recall, False Auto-Handle Rate (safety-critical).
   - **Retrieval**: Hit@K, Mean Reciprocal Rank (MRR).

4. **LLM-as-a-Judge & Human Agreement**:
   - Multi-dimensional rubric: Groundedness, Appropriateness, Resolution Validity, Escalation Soundness.
   - Inter-annotator agreement metrics (Cohen’s Kappa / Correlation) between human annotations and LLM judge.

5. **Failure Analysis & Decision Log**:
   - Detailed breakdown of 5 major failure modes.
   - Decision log documenting 10–15 non-obvious engineering trade-offs.

---

## Repository Structure

```
hiver-ai-support-agent/
│
├── data/
│   ├── raw/                # Original raw dataset (ignored by git)
│   ├── processed/          # Cleaned dialogue turns and features (ignored by git)
│   └── golden/             # Manually labelled golden evaluation set (150-250 examples)
│
├── src/                    # Core AI support agent source code
│   ├── __init__.py
│   ├── config.py           # Configuration loading and parameter definitions
│   ├── preprocessing.py    # Raw data cleaning, thread parsing, and filtering
│   ├── taxonomy.py         # Brand-specific intent taxonomy definition
│   ├── intent_classifier.py# Intent classification module
│   ├── retrieval.py        # Similar historical resolution retrieval
│   ├── response_generator.py# Grounded response generation
│   ├── escalation.py       # Escalation decision engine (AUTO_HANDLE vs ESCALATE)
│   └── pipeline.py         # End-to-end orchestration pipeline
│
├── baselines/              # Comparison baselines
│   ├── __init__.py
│   ├── majority_baseline.py# Trivial baseline (majority class predictor)
│   └── tfidf_baseline.py   # Simple baseline (TF-IDF + classification / retrieval)
│
├── evaluation/             # Metrics, evaluation harness, and judging
│   ├── __init__.py
│   ├── metrics.py          # Automated metric calculations (F1, MRR, Hit@K)
│   ├── evaluate.py         # Full evaluation suite runner
│   ├── llm_judge.py        # LLM-as-a-Judge scoring module
│   └── human_agreement.py  # Human-LLM agreement correlation & Kappa
│
├── scripts/                # Execution CLI scripts
│   ├── __init__.py
│   ├── prepare_data.py     # Script to process raw data
│   ├── build_golden_set.py # Script to sample and format golden set
│   └── run_pipeline.py     # Script to run pipeline on single or batch queries
│
├── outputs/                # Evaluation reports, confusion matrices, logs (ignored by git)
├── tests/                  # Unit and contract tests
│   ├── __init__.py
│   └── test_pipeline.py    # Pipeline sanity and configuration tests
├── report/                 # Final technical report (maximum 6 pages)
│
├── requirements.txt        # Minimal Python dependencies
├── .env.example            # Environment configuration template
├── .gitignore              # Git ignore rules for data, cache, and secrets
└── README.md               # Project documentation
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Virtual environment tool (`venv`)

### Setup
```bash
# Clone the repository
git clone https://github.com/RudraTripathi8866/hiver-ai-support-agent.git
cd hiver-ai-support-agent

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install Phase 1 dependencies
pip install -r requirements.txt

# Create local environment config
cp .env.example .env
```
