# LSL-t5

Fine-tuning pipeline for translating Latvian text into LSL glosses using a vocabulary-pruned mT5-small model.

### Read the Handbook here: https://lucyferyx.github.io/LSL-handbook

## Project structure

```
LSL-t5/
├── code/
│   ├── 1_model_download.ipynb   # One-time: download mt5-small from HuggingFace (~28 min)
│   ├── 1_pruning.ipynb          # One-time: prune mt5-small vocab to Latvian/gloss tokens only
│   ├── 2_training.ipynb         # Fine-tune pruned model on sentence–gloss pairs
│   ├── glosses_check.ipynb      # Dataset validation and gloss frequency analysis
│   ├── list_generate.ipynb      # Utility for generating training data lists
│   └── testing-simple.ipynb     # Quick inference tests on a saved model
├── txt/
│   ├── latvian_sentences_*.txt  # Latvian input sentences (paired by index)
│   └── lsl_glosses_*.txt        # Corresponding LSL gloss sequences
├── models/                      # Generated — not in repo
│   ├── mt5-small/               # Downloaded base model
│   ├── mt5-pruned/              # Pruned model (vocab reduced from 250k → ~3k tokens)
│   └── mt5-lsl-model/           # Training runs, numbered by ID
│       └── <run_id>/
│           ├── pytorch_model.bin
│           └── metadata.json
├── requirements.txt
└── README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the [mt5-small](https://huggingface.co/google/mt5-small) model:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/mt5-small",
    local_dir="models/mt5-small"
)
```

## Training a model

Run the notebooks in order:

1. **`1_pruning.ipynb`** scans `txt/` to find all tokens used in your data, then strips the mt5-small vocabulary down to only those tokens. Saves the pruned model to `models/mt5-pruned/`. Re-run only if your training data changes.

2. **`2_training.ipynb`** fine-tunes the pruned model. Each run is saved to a numbered directory under `models/mt5-lsl-model/` with a `metadata.json` summary (loss, epochs, example outputs).

## Training data format

Each numbered batch file must be line-aligned, e.g. line `N` in `latvian_sentences_K.txt` corresponds to line `N` in `lsl_glosses_K.txt`.

```
# latvian_sentences_1.txt        # lsl_glosses_1.txt
Sveiks, [NAME].                  sveiks [NAME]
Kādā vārdā tevi saukt?           kāds vārds tu saukt
```