# Low-Resource ASR for Zambian Languages (Bemba, Nyanja, Tonga)

**Final Year Research Project**  
**University of Zambia (UNZA)**  
**School of Natural and Applied Sciences**  
**Department of Computing and Informatics**  
**Academic Year: 2026**

---

> **Low-Resource Automatic Speech Recognition for Zambian Languages: A Comparative Analysis of Pre-Trained Models on Bemba, Nyanja, and Tonga**

This repository contains Jupyter notebooks for **monolingual Automatic Speech Recognition (ASR)** on three low-resource Zambian languages — **Bemba**, **Nyanja**, and **Tonga**. Pre-trained speech models are fine-tuned on the Zambezi Voice and evaluated with **Word Error Rate (WER)** and **Character Error Rate (CER)**.

---

## Live demo and models

| Resource | Link |
|----------|------|
| **Interactive demo (ZamVoice)** | [https://huggingface.co/spaces/buumba641/ZamVoice](https://huggingface.co/spaces/buumba641/ZamVoice) |
| **All fine-tuned models** | [https://huggingface.co/buumba641](https://huggingface.co/buumba641) |
| **Multilingual ASR dataset** | [https://huggingface.co/datasets/buumba641/Zambia-MultiLingual-ASR-Dataset](https://huggingface.co/datasets/buumba641/Zambia-MultiLingual-ASR-Dataset) |

Upload or record audio in **Nyanja, Tonga, or Bemba** on the demo Space to get a transcription (with optional English translation).

---

## Project information

| Field | Details |
|-------|---------|
| **Student** | Buumba Chinjila |
| **Institution** | University of Zambia (UNZA) |
| **School** | School of Natural and Applied Sciences |
| **Department** | Department of Computing and Informatics |
| **Academic year** | 2026 |
| **Proposal date** | 20 March 2026 |
| **Hugging Face** | [@buumba641](https://huggingface.co/buumba641) |
| **Contact** | [buumbachinjla@gmail.com](mailto:buumbachinjla@gmail.com) |

---

## Motivation

Many Zambian communities communicate primarily orally, while most digital services remain English-first. Local-language ASR can improve:

- **Accessibility** — voice interfaces, transcription, and captioning in local languages  
- **Digital record-keeping** — meetings, consultations, and field reporting  
- **Inclusion** — better access for users with limited English literacy  

---

## Problem statement

Developing ASR for Zambian languages is difficult because of:

1. **Limited labelled speech data** — roughly 22–28 hours per language in Zambezi Voice / BembaSpeech  
2. **Limited compute** — large-scale training from scratch is often impractical in a student setting  

The project therefore focuses on **transfer learning**: fine-tuning strong pre-trained models under these constraints.

---

## Aim

To implement and evaluate **monolingual ASR pipelines** for Bemba, Nyanja, and Tonga by **fine-tuning and comparing open-source pre-trained models** on public Zambian speech corpora.

---

## Objectives

1. **Model benchmarking** — Fine-tune and compare multiple architectures (Whisper, WavLM, Wav2Vec2 XLS-R, MMS).  
2. **Proxy-language adaptation** — Where Whisper has no native token for the target language, use a related Bantu language token (Shona for Nyanja/Tonga, Swahili for Bemba).  
3. **Performance evaluation** — Measure **WER** (primary) and **CER** (secondary), with example transcriptions and character-level confusion analysis.  

---

## Dataset

| Language | Primary corpus | Source |
|----------|----------------|--------|
| **Bemba** | [BembaSpeech](https://github.com/unza-speech-lab/BembaSpeech) | UNZA Speech Lab |
| **Nyanja** | [zambezi-voice-nyanja](https://github.com/unza-speech-lab/zambezi-voice-nyanja) | Zambezi Voice |
| **Tonga** | [zambezi-voice](https://github.com/unza-speech-lab/zambezi-voice) (`tonga/toi`) | Zambezi Voice |

Paper: [Zambezi Voice: A Multilingual Speech Corpus for Zambian Languages](https://arxiv.org/abs/2306.04428).

> Dataset download and licence terms are not redistributed in this repository. Follow each corpus’s official instructions (Zambezi Voice is typically CC BY-NC-ND).

---

## Models evaluated

| Model family | Variants | Architecture | Decoding |
|--------------|----------|--------------|----------|
| **Whisper** | tiny, base, small, medium | Encoder–decoder | Greedy; optional proxy language token |
| **WavLM** | base+ | Encoder + CTC | Greedy CTC |
| **Wav2Vec2 XLS-R** | 300M | Encoder + CTC | Greedy CTC |
| **MMS** | 300M | Encoder + CTC | Greedy CTC |

### Best checkpoints on Hugging Face

| Language | Best model | Hugging Face model ID |
|----------|------------|------------------------|
| **Nyanja** | Whisper Medium (Shona proxy) | [`buumba641/Nyanja-whisper-medium-shona-proxy`](https://huggingface.co/buumba641/Nyanja-whisper-medium-shona-proxy) |
| **Tonga** | Whisper Medium (Shona proxy) | [`buumba641/tonga-whisper-medium-shona-proxy`](https://huggingface.co/buumba641/tonga-whisper-medium-shona-proxy) |
| **Bemba** | Whisper Medium (Swahili proxy) | [`buumba641/bemba-whisper-medium-siwahili-final`](https://huggingface.co/buumba641/bemba-whisper-medium-siwahili-final) |

Full model list: [huggingface.co/buumba641/models](https://huggingface.co/buumba641/models).

---

## Results (held-out test set)

### Best model per language

![Best WER per language](wer_best_per_language.png)

| Language | Best model | **WER** | **CER** |
|----------|------------|---------|---------|
| **Nyanja** | Whisper Medium (Shona proxy) | **21.5%** | 5.4% |
| **Tonga** | Whisper Medium (Shona proxy) | **31.0%** | 5.8% |
| **Bemba** | Whisper Medium (Swahili proxy) | **35.6%** | 6.4% |

### Full comparison by language

![WER by model and language](wer_all_models.png)

| Language | Model | WER | CER |
|----------|-------|-----|-----|
| Nyanja | Whisper Medium (Shona proxy) | **21.5%** | 5.4% |
| Nyanja | Whisper Small | 29.1% | 8.3% |
| Nyanja | Whisper Base | 29.7% | 8.8% |
| Nyanja | WavLM base+ (CTC) | 51.2% | 11.5% |
| Nyanja | Whisper Tiny | 69.6% | 31.2% |
| Tonga | Whisper Medium (Shona proxy) | **31.0%** | 5.8% |
| Tonga | MMS-300M (CTC) | 34.1% | 6.5% |
| Tonga | XLSR-300M (CTC) | 41.3% | 7.7% |
| Tonga | Whisper Small | 44.3% | 11.3% |
| Tonga | WavLM base+ (CTC) | 45.4% | 7.8% |
| Tonga | Whisper Base (Shona proxy) | 46.9% | 14.9% |
| Tonga | Whisper Base | 51.1% | 14.3% |
| Tonga | Whisper Tiny (Shona proxy) | 60.5% | 22.9% |
| Tonga | Whisper Tiny | 61.4% | 26.4% |
| Bemba | Whisper Medium (Swahili proxy) | **35.6%** | 6.4% |
| Bemba | MMS-300M (CTC) | 36.8% | 6.3% |
| Bemba | Whisper Small | 48.8% | 9.6% |
| Bemba | Whisper Base | 52.6% | 12.2% |
| Bemba | Whisper Tiny | 56.9% | 14.2% |
| Bemba | XLSR-300M (CTC)* | 99.9% | 22.4% |

\*Bemba XLSR-300M (~100% WER) is treated as a failed run (checkpoint, vocabulary, or decoding issue) and is excluded from ranking conclusions.

### Main findings

1. **Whisper Medium with a related Bantu proxy language** is the strongest approach for all three languages.  
2. **Larger Whisper models perform better** — Tiny is consistently weakest; Medium is best where evaluated with a proxy.  
3. **MMS-300M** is the strongest CTC alternative and is competitive with Whisper Medium on Bemba.  
4. **CER is much lower than WER** for the better models (often about 5–15%), which suggests many errors are short substitutions or spacing rather than total failure.  

**Recommendation:** Prefer **Whisper Medium + Shona proxy** (Nyanja, Tonga) or **Swahili proxy** (Bemba) when accuracy matters; use **MMS-300M** when a lighter CTC model is needed.

---

## Repository contents

| File | Description |
|------|-------------|
| `ASR_BEMBA_WHISPER_medium_proxy_swahili_clean.ipynb` | Fine-tune Whisper Medium on Bemba (Swahili proxy) |
| `ASR_NYANJA_WHISPER_medium_proxy_shona_clean.ipynb` | Fine-tune Whisper Medium on Nyanja (Shona proxy) |
| `ASR_TONGA_WHISPER_medium_proxy_shona_clean.ipynb` | Fine-tune Whisper Medium on Tonga (Shona proxy) |
| `Evaluation_all_clean.ipynb` | Evaluate all checkpoints: WER, CER, examples, character confusion matrices |
| `wer_best_per_language.png` | Chart: best test WER per language |
| `wer_all_models.png` | Chart: full WER comparison by language |

### Typical workflow

1. Clone the language corpus and install dependencies.  
2. Load and clean transcripts; resample audio to 16 kHz.  
3. Fine-tune the chosen model (Whisper or CTC).  
4. Evaluate on the held-out test split.  
5. Push the best checkpoint to Hugging Face Hub.  

---

## Suggested environment

- Python 3.10+  
- PyTorch (CUDA recommended)  
- Hugging Face `transformers`, `datasets`, `evaluate`, `accelerate`  
- `jiwer` (WER / CER)  
- Jupyter or Google Colab  

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate accelerate jiwer huggingface_hub
```

For Hub uploads and the demo, set a token (do not commit tokens to git):

```bash
export HF_TOKEN=hf_your_token_here
```

---

## Ethics, privacy, and licensing

- Only public research datasets are used (Zambezi Voice, BembaSpeech).  
- This notebooks-only repository does not collect private user audio.  
- Respect each dataset’s licence when redistributing data or derived models.  
- Model weights are published under the terms of their base models and training data licences.  

---

## Acknowledgements

- [UNZA Speech Lab](https://github.com/unza-speech-lab) — Zambezi Voice and BembaSpeech  
- OpenAI (Whisper), Meta (MMS, Wav2Vec2), Microsoft (WavLM)  
- Hugging Face — model hosting and Spaces  

---

## Contact

For questions or collaboration:

- Open an issue in this repository  
- Email: **buumbachinjla@gmail.com**  
- Hugging Face: [buumba641](https://huggingface.co/buumba641)  
- Demo: [ZamVoice Space](https://huggingface.co/spaces/buumba641/ZamVoice)
