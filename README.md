# Low-Resource Multilingual ASR for Zambian Languages: Nyanja, Tonga & Bemba

A final-year research project building and comparing **monolingual** and **multilingual** Automatic Speech Recognition (ASR) models for three low-resource Zambian languages: **Nyanja**, **Tonga**, and **Bemba**.

Models are fine-tuned from [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) using a CTC objective and evaluated using Word Error Rate (WER) and Character Error Rate (CER).

---

## Project Goals

| Goal | Description |
|------|-------------|
| Monolingual models | One dedicated ASR model per language (Nyanja, Tonga, Bemba) |
| Multilingual model | A single model trained on all three languages combined |
| Comparison | Side-by-side WER/CER evaluation charts and summary table |

---

## Project Structure

```
.
├── configs/
│   ├── nyanja_config.yaml          # Nyanja monolingual config
│   ├── tonga_config.yaml           # Tonga monolingual config
│   ├── bemba_config.yaml           # Bemba monolingual config
│   └── multilingual_config.yaml   # Multilingual config
├── data/
│   ├── nyanja/
│   │   ├── audio/                  # .wav/.flac audio files
│   │   └── transcriptions/        # Matching .txt transcription files
│   ├── tonga/  (same layout)
│   └── bemba/  (same layout)
├── src/
│   ├── data_preparation/
│   │   ├── prepare_dataset.py     # Dataset loading, splitting & vocab building
│   │   └── preprocess.py         # Text normalisation pipeline
│   ├── models/
│   │   ├── trainer.py             # Shared Wav2Vec2 CTC fine-tuning logic
│   │   ├── monolingual/
│   │   │   ├── train_nyanja.py
│   │   │   ├── train_tonga.py
│   │   │   └── train_bemba.py
│   │   └── multilingual/
│   │       └── train_multilingual.py
│   ├── evaluation/
│   │   ├── evaluate.py            # Per-model WER/CER evaluation
│   │   └── compare_models.py     # Side-by-side comparison + charts
│   └── utils/
│       ├── metrics.py             # WER & CER computation (jiwer)
│       ├── audio_utils.py         # Audio loading & filtering
│       └── vocab_builder.py      # Character vocabulary construction
├── outputs/                       # Training checkpoints & evaluation results
├── tests/                         # Unit tests (pytest)
└── requirements.txt
```

---

## Data Format

Place audio and transcription files for each language under `data/<language>/`:

```
data/nyanja/
    audio/
        utt001.wav
        utt002.wav
        ...
    transcriptions/
        utt001.txt      ← plain text, one utterance per file
        utt002.txt
        ...
```

**Alternatively**, provide a `manifest.csv` (or `manifest.tsv`) with columns:

```
audio_path,transcription
data/nyanja/audio/utt001.wav,ndiyo bwino
data/nyanja/audio/utt002.wav,zikomo kwambiri
```

---

## Step-by-Step Workflow

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare datasets

```bash
# Monolingual – prepare one language at a time
python -m src.data_preparation.prepare_dataset --config configs/nyanja_config.yaml
python -m src.data_preparation.prepare_dataset --config configs/tonga_config.yaml
python -m src.data_preparation.prepare_dataset --config configs/bemba_config.yaml

# Multilingual – combine all three languages
python -m src.data_preparation.prepare_dataset \
    --config configs/multilingual_config.yaml \
    --multilingual
```

### 3. Train monolingual models

```bash
python -m src.models.monolingual.train_nyanja
python -m src.models.monolingual.train_tonga
python -m src.models.monolingual.train_bemba
```

### 4. Train the multilingual model

```bash
python -m src.models.multilingual.train_multilingual
```

### 5. Evaluate each model

```bash
# Evaluate each monolingual model on its own test set
python -m src.evaluation.evaluate \
    --model_dir outputs/nyanja \
    --dataset_path outputs/nyanja/dataset \
    --language nyanja \
    --output_dir outputs/nyanja/evaluation

python -m src.evaluation.evaluate \
    --model_dir outputs/tonga \
    --dataset_path outputs/tonga/dataset \
    --language tonga \
    --output_dir outputs/tonga/evaluation

python -m src.evaluation.evaluate \
    --model_dir outputs/bemba \
    --dataset_path outputs/bemba/dataset \
    --language bemba \
    --output_dir outputs/bemba/evaluation

# Evaluate the multilingual model per language
python -m src.evaluation.evaluate \
    --model_dir outputs/multilingual \
    --dataset_path outputs/nyanja/dataset \
    --language nyanja \
    --output_dir outputs/multilingual/evaluation

python -m src.evaluation.evaluate \
    --model_dir outputs/multilingual \
    --dataset_path outputs/tonga/dataset \
    --language tonga \
    --output_dir outputs/multilingual/evaluation

python -m src.evaluation.evaluate \
    --model_dir outputs/multilingual \
    --dataset_path outputs/bemba/dataset \
    --language bemba \
    --output_dir outputs/multilingual/evaluation
```

### 6. Compare models

```bash
python -m src.evaluation.compare_models --results_dir outputs
```

This produces:
- A printed comparison table (WER & CER for each model × language)
- `outputs/comparison/wer_comparison.png` — grouped bar chart
- `outputs/comparison/cer_comparison.png` — grouped bar chart
- `outputs/comparison/performance_heatmap.png` — heatmap
- `outputs/comparison/comparison_results.csv` — raw results

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Languages

| Language | ISO Code | Region |
|----------|----------|--------|
| Nyanja (Chichewa) | ny | Eastern/Central Zambia |
| Tonga | toi | Southern Zambia |
| Bemba | bem | Northern Zambia |

---

## Model Architecture

- **Base model**: [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- **Fine-tuning**: CTC (Connectionist Temporal Classification)
- **Feature encoder**: Frozen during fine-tuning (transfer learning)
- **Vocabulary**: Character-level (language-specific for monolingual; shared for multilingual)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate — fraction of words incorrectly transcribed |
| **CER** | Character Error Rate — fraction of characters incorrectly transcribed |

Lower is better for both metrics.
