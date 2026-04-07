# Low-Resource ASR for Zambian Languages (Bemba, Nyanja, Tonga)
**Final Year Research Project (UNZA — Department of Computing and Informatics, 2026)**

This repository contains **Jupyter notebooks** for a final year research project titled:

> **Low-Resource Automatic Speech Recognition for Zambian Languages: A Comparative Analysis of Pre-Trained Models on Bemba, Nyanja, and Tonga**

The work focuses on **monolingual Automatic Speech Recognition (ASR)** for three low-resource Zambian languages—**Bemba, Nyanja, and Tonga**—using the **Zambezi Voice** dataset (UNZA Speech Lab). The main goal is to **fine-tune and benchmark pre-trained speech models** in extremely low-resource conditions and evaluate performance using **Word Error Rate (WER)**.

---

## Project Information
- **Student:** Buumba Chinjila  
- **Institution:** The University of Zambia (UNZA), School of Natural and Applied Sciences  
- **Academic Year:** 2026  
- **Proposal Submission Date:** March 20, 2026  

---

## Motivation
Many Zambian communities primarily communicate orally, yet most digital services are English-first. ASR for local languages can improve:
- accessibility (voice interfaces, transcription, captioning)
- digital record keeping (meetings, consultations, reporting)
- inclusion for users with limited English literacy

---

## Problem Statement
ASR development for Zambian languages faces:
- **Limited labeled data** (~22–24 hours per language in Zambezi Voice)
- **Limited compute**, making large-scale training difficult

---

## Aim
To implement and evaluate **monolingual ASR pipelines** for Bemba, Nyanja, and Tonga by **fine-tuning and comparing open-source pre-trained models** on Zambezi Voice subsets.

---

## Objectives
1. **Model Benchmarking:** Fine-tune and compare multiple pre-trained models (e.g., **XLS-R, Whisper, MMS, HuBERT**).
2. **Data Augmentation:** Use techniques such as **speed perturbation** and **SpecAugment** where appropriate.
3. **Performance Evaluation:** Evaluate using **WER** (primary) and optionally **CER** (secondary), including error analysis.

---

## Dataset
Primary dataset:
- **Zambezi Voice** (UNZA Speech Lab)

Languages used in this project:
- **Bemba**
- **Nyanja**
- **Tonga**

> Note: Dataset access/setup is not included in this repository. Please follow the dataset’s official instructions and license terms.

---

## Models (Planned / Evaluated)
This project benchmarks pre-trained model families such as:

| Model Family | Example Variants | Architecture |
|---|---|---|
| **XLS-R** | 0.3B, 1B | Encoder (CTC) |
| **Whisper** | tiny, base, small | Encoder–Decoder |
| **MMS** | 1B | Encoder (CTC) |
| **HuBERT** | Base, Large | Encoder (CTC) |

---

## Repository Contents
This repo currently contains **ASR notebooks only**.

Typical notebook topics include:
- data inspection / EDA
- preprocessing (resampling, text normalization)
- fine-tuning (per language, per model)
- evaluation (WER/CER)
- error analysis and result tables

---

### Suggested environment (edit to match your notebooks)
- Python 3.10+
- PyTorch
- Hugging Face `transformers`, `datasets`, `evaluate`
- `jiwer` (WER)
- `librosa` / `soundfile`
- Jupyter / JupyterLab

---

## Results
Results (WER/CER tables, plots, and observations) will be added/updated as experiments progress.

---

## Ethics, Privacy, and Licensing
- Only public/anonymized datasets are used (primary: Zambezi Voice)
- No user audio is collected in this notebooks-only repository
- Zambezi Voice must be used with full attribution and according to its license terms

---

## Contact
For questions or collaboration, please open an issue in this repository.
or send a mail to buumbachinjla@gmail.com
