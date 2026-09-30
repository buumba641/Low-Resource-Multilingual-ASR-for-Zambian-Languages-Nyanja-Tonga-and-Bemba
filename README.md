# Low-Resource ASR for Zambian Languages

### A Comparative Analysis of Pre-Trained Speech Recognition Models for Bemba, Nyanja, and Tonga

**Final Year Research Project — University of Zambia (UNZA)**
**School of Natural and Applied Sciences · Department of Computing and Informatics · 2026**

---

## 📌 Overview

This repository contains the implementation, experiments, evaluation notebooks, and results for a final-year research project investigating **Automatic Speech Recognition (ASR) for low-resource Zambian languages**.

The project focuses on three languages:

* 🇿🇲 **Bemba**
* 🇿🇲 **Nyanja**
* 🇿🇲 **Tonga**

The central research question is:

> **How effectively can pre-trained multilingual speech models be adapted to low-resource Zambian languages using the limited speech data and computing resources available in a student research environment?**

Rather than training ASR systems from scratch, the project uses **transfer learning**. Pre-trained speech recognition models are fine-tuned on Zambian speech data and then compared using **Word Error Rate (WER)** and **Character Error Rate (CER)**.

The project also includes **ZamVoice**, an interactive application that demonstrates how the resulting ASR models can be used in a practical local-language voice interface.

---

# 🎯 Project Objective

## Research Objective

The main objective is:

> **To implement, fine-tune, and comparatively evaluate pre-trained ASR models for Bemba, Nyanja, and Tonga using low-resource Zambian speech datasets.**

The research investigates whether models originally trained on large and multilingual speech datasets can be effectively adapted to Zambian languages despite having relatively small amounts of labelled speech data.

The models investigated include:

* OpenAI Whisper
* Meta MMS
* Wav2Vec2 XLS-R
* Microsoft WavLM

The models are evaluated under a common experimental setup and compared using WER and CER.

---

## 💻 Repository / Program Objective

This repository is designed to provide a reproducible implementation of the research pipeline.

The notebooks and supporting files allow a researcher to:

```text
Speech Dataset
      │
      ▼
Load Audio + Transcripts
      │
      ▼
Data Cleaning
      │
      ▼
Audio Preprocessing
      │
      ▼
Model Preparation
      │
      ▼
Fine-Tuning
      │
      ▼
Evaluation
      │
      ├── Word Error Rate (WER)
      ├── Character Error Rate (CER)
      ├── Example Predictions
      └── Character Error Analysis
      │
      ▼
Best Model
      │
      ▼
Hugging Face Model
      │
      ▼
ZamVoice Demo
```

The repository therefore serves both as:

1. **The implementation of the final-year research experiments**, and
2. **A reproducible starting point for further ASR research on Zambian languages.**

---

# 🚀 What the Program Does

At a high level, the project takes **recorded speech in Bemba, Nyanja, or Tonga** and trains an ASR model to produce the corresponding written transcription.

For example:

```text
Audio
  │
  │  "Muli shani?"
  ▼
ASR Model
  │
  ▼
"Muli shani?"
```

The model is trained using paired:

```text
Audio → Transcription
```

examples.

After training, previously unseen test audio is passed through the model:

```text
Unseen speech
      ↓
Fine-tuned ASR model
      ↓
Predicted transcription
      ↓
Compare with reference transcription
      ↓
WER / CER
```

This makes it possible to quantitatively determine how well each model recognizes each language.

---

# 🌍 Languages

| Language   | Dataset       | Approx. Training Data |
| ---------- | ------------- | --------------------: |
| **Bemba**  | BembaSpeech   |             ~28 hours |
| **Nyanja** | Zambezi Voice |             ~25 hours |
| **Tonga**  | Zambezi Voice |             ~22 hours |

The experiments are **monolingual**.

This means that a separate ASR model is fine-tuned for each target language rather than training one model to recognize all three languages simultaneously.

For example:

```text
Bemba audio  → Bemba ASR model  → Bemba transcription

Nyanja audio → Nyanja ASR model → Nyanja transcription

Tonga audio  → Tonga ASR model  → Tonga transcription
```

---

# 🧠 Why Pre-Trained Models?

Training a modern speech recognition model from scratch requires extremely large speech datasets and substantial computational resources.

The available Zambian datasets contain only approximately **22–28 hours of speech per language**.

Therefore, this project uses **transfer learning**.

A model that has already learned general speech representations from large datasets is adapted to the target Zambian language.

Conceptually:

```text
Large multilingual speech dataset
              │
              ▼
      Pre-trained model
              │
              │ Fine-tuning
              ▼
   Small Zambian speech dataset
              │
              ▼
     Zambian ASR model
```

This approach allows low-resource languages to benefit from knowledge learned from much larger speech corpora.

---

# 🔬 Models Investigated

| Model              | Architecture    | Variants / Size           | Decoding       |
| ------------------ | --------------- | ------------------------- | -------------- |
| **Whisper**        | Encoder–Decoder | Tiny, Base, Small, Medium | Autoregressive |
| **WavLM**          | Encoder + CTC   | Base+                     | Greedy CTC     |
| **Wav2Vec2 XLS-R** | Encoder + CTC   | 300M                      | Greedy CTC     |
| **MMS**            | Encoder + CTC   | 300M                      | Greedy CTC     |

Whisper was investigated across multiple model sizes to examine the relationship between model capacity and ASR performance.

---

# 🗣️ Proxy Language Adaptation

Whisper does not provide native language tokens for all three target languages.

Therefore, an experimental **proxy-language approach** was investigated.

The experiments use:

| Target Language | Proxy Language |
| --------------- | -------------- |
| Nyanja          | Shona          |
| Tonga           | Shona          |
| Bemba           | Swahili        |

The proxy language is used as part of Whisper's language-conditioning mechanism. The resulting models are still **fine-tuned and evaluated on the target Zambian language**.

This should therefore not be interpreted as translating the target language into the proxy language.

For example:

```text
Nyanja speech
      ↓
Whisper + Shona language conditioning
      ↓
Fine-tuned on Nyanja
      ↓
Nyanja transcription
```

The proxy-language experiments investigate whether linguistic similarity can provide useful conditioning for languages that are not directly represented by Whisper's language tokens.

---

# 📊 Evaluation

The primary evaluation metric is:

### Word Error Rate (WER)

WER measures errors at the word level:

```text
WER = (Substitutions + Deletions + Insertions) / Number of Reference Words
```

A lower WER indicates fewer word-level transcription errors.

The secondary metric is:

### Character Error Rate (CER)

CER evaluates errors at the character level and is particularly useful for low-resource languages where:

* words may be morphologically complex,
* tokenization can affect WER,
* spacing differences can produce word-level errors.

The repository also includes:

* Example predictions
* Character-level error analysis
* Confusion analysis
* Model-to-model comparisons

---

# 📈 Results

## Best Model per Language

| Language   | Best Model                     |       WER |      CER |
| ---------- | ------------------------------ | --------: | -------: |
| **Nyanja** | Whisper Medium + Shona proxy   | **21.5%** | **5.4%** |
| **Tonga**  | Whisper Medium + Shona proxy   | **31.0%** | **5.8%** |
| **Bemba**  | Whisper Medium + Swahili proxy | **35.6%** | **6.4%** |

These results represent the best-performing configurations in the evaluated experiments.

### Important interpretation

CER is considerably lower than WER for the best-performing models. This indicates that although some predictions contain word-level errors, many of the predicted characters remain close to the reference transcription.

However, **CER should not be interpreted as meaning that the model makes only a small number of complete transcription errors**. WER and CER measure different types of errors and should be considered together.

---

# 📁 Repository Structure

The repository is organized by language and experiment type.

```text
Low-Resource-Multilingual-ASR-for-Zambian-Languages/
│
├── README.md
│
├── Bemba/
│   ├── ASR_BEMBA_WHISPER_medium_proxy_swahili_clean.ipynb
│   ├── ...
│   └── results/
│       ├── ...
│       └── ...
│
├── Nyanja/
│   ├── ASR_NYANJA_WHISPER_medium_proxy_shona_clean.ipynb
│   ├── ...
│   └── results/
│       ├── ...
│       └── ...
│
├── Tonga/
│   ├── ASR_TONGA_WHISPER_medium_proxy_shona_clean.ipynb
│   ├── ...
│   └── results/
│       ├── ...
│       └── ...
│
├── Evaluation/
│   └── Evaluation_all_clean.ipynb
│
└── ...
```

> **Note:** The structure above describes the organization of the research code. Dataset files and model checkpoints are not redistributed in this repository where their respective licences prohibit redistribution.

---

# 📂 Directory Guide

## `Bemba/`

Contains the experiments specific to **Bemba ASR**.

This directory contains the Bemba data-processing and model fine-tuning notebooks, including the Whisper Medium experiment using **Swahili as the proxy language**.

Example:

```text
Bemba/
└── ASR_BEMBA_WHISPER_medium_proxy_swahili_clean.ipynb
```

The notebook covers:

1. Loading the Bemba corpus
2. Cleaning the transcripts
3. Preparing the audio
4. Loading the pre-trained Whisper model
5. Configuring the proxy language
6. Fine-tuning
7. Evaluating the model
8. Saving/publishing the resulting checkpoint

---

## `Nyanja/`

Contains the experiments specific to **Nyanja ASR**.

Example:

```text
Nyanja/
└── ASR_NYANJA_WHISPER_medium_proxy_shona_clean.ipynb
```

The notebook implements the Nyanja Whisper fine-tuning pipeline using **Shona as the proxy language**.

The resulting model is evaluated on held-out Nyanja speech.

---

## `Tonga/`

Contains the experiments specific to **Tonga ASR**.

Example:

```text
Tonga/
└── ASR_TONGA_WHISPER_medium_proxy_shona_clean.ipynb
```

The notebook implements the Tonga Whisper fine-tuning pipeline using **Shona as the proxy language**.

The resulting model is evaluated on held-out Tonga speech.

---

## `Evaluation/`

Contains notebooks used to evaluate and compare models across languages.

The main evaluation notebook:

```text
Evaluation/
└── Evaluation_all_clean.ipynb
```

It is responsible for:

* Loading trained checkpoints
* Running inference on the test sets
* Calculating WER
* Calculating CER
* Generating example predictions
* Performing character-level error analysis
* Comparing models
* Producing research figures

This separates **model training** from **final comparative evaluation**.

---

# 🔄 Experimental Workflow

Each language follows approximately the same pipeline.

```text
                    ┌──────────────────┐
                    │   Speech Corpus  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Transcript       │
                    │ Cleaning         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Audio            │
                    │ Preprocessing    │
                    │ 16 kHz           │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Pre-trained      │
                    │ ASR Model        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Fine-tuning on   │
                    │ Target Language  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Held-out Test    │
                    │ Set              │
                    └────────┬─────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ WER / CER / Error Analysis │
              └─────────────────────────────┘
```

---

# 🧪 Experimental Comparison

The project does not evaluate only one model.

Multiple architectures and model sizes are tested so that their performance can be compared under the same low-resource conditions.

For example, Nyanja experiments include:

| Model                        |       WER |      CER |
| ---------------------------- | --------: | -------: |
| Whisper Medium + Shona proxy | **21.5%** | **5.4%** |
| Whisper Small                |     29.1% |     8.3% |
| Whisper Base                 |     29.7% |     8.8% |
| WavLM Base+                  |     51.2% |    11.5% |
| Whisper Tiny                 |     69.6% |    31.2% |

Similar comparisons are performed for Bemba and Tonga.

The purpose is to investigate **how model architecture, model capacity, and language conditioning affect ASR performance in low-resource Zambian languages**.

---

# 📋 Full Results

### Nyanja

| Model                        |       WER |      CER |
| ---------------------------- | --------: | -------: |
| Whisper Medium + Shona proxy | **21.5%** | **5.4%** |
| Whisper Small                |     29.1% |     8.3% |
| Whisper Base                 |     29.7% |     8.8% |
| WavLM Base+                  |     51.2% |    11.5% |
| Whisper Tiny                 |     69.6% |    31.2% |

### Tonga

| Model                        |       WER |      CER |
| ---------------------------- | --------: | -------: |
| Whisper Medium + Shona proxy | **31.0%** | **5.8%** |
| MMS-300M                     |     34.1% |     6.5% |
| XLS-R-300M                   |     41.3% |     7.7% |
| Whisper Small                |     44.3% |    11.3% |
| WavLM Base+                  |     45.4% |     7.8% |
| Whisper Base + Shona proxy   |     46.9% |    14.9% |
| Whisper Base                 |     51.1% |    14.3% |
| Whisper Tiny + Shona proxy   |     60.5% |    22.9% |
| Whisper Tiny                 |     61.4% |    26.4% |

### Bemba

| Model                          |       WER |      CER |
| ------------------------------ | --------: | -------: |
| Whisper Medium + Swahili proxy | **35.6%** |     6.4% |
| MMS-300M                       |     36.8% | **6.3%** |
| Whisper Small                  |     48.8% |     9.6% |
| Whisper Base                   |     52.6% |    12.2% |
| Whisper Tiny                   |     56.9% |    14.2% |
| XLS-R-300M*                    |     99.9% |    22.4% |

*The Bemba XLS-R result is treated as a failed experimental run and is excluded from model-ranking conclusions.

---

# 🔎 Main Findings

The experiments indicate several patterns:

### 1. Whisper Medium performed strongly

Whisper Medium produced the lowest WER among the evaluated configurations for all three languages.

### 2. Proxy-language conditioning affected Whisper performance

Using a related proxy language during Whisper decoding/conditioning produced substantial differences in the evaluated experiments.

* Shona was used for Nyanja and Tonga.
* Swahili was used for Bemba.

### 3. Model capacity matters

Within the Whisper family, the larger models generally produced lower WER and CER than the smaller models.

### 4. MMS provides a competitive CTC alternative

MMS-300M performed competitively with Whisper Medium, particularly for Bemba and Tonga.

### 5. WER and CER provide complementary information

WER and CER can show different aspects of ASR quality. Therefore, both metrics are reported rather than relying on a single metric.

---

# 🤖 ZamVoice

The research models are also used in **ZamVoice**, an interactive demonstration of local-language speech technology.

ZamVoice demonstrates a pipeline where:

```text
Bemba / Nyanja / Tonga speech
             │
             ▼
       ASR Model
             │
             ▼
       Transcription
             │
             ▼
    English Translation
             │
             ▼
      Speech Synthesis
```

The demonstration combines the ASR models with translation and speech synthesis technologies to show a potential local-language voice interface.

### Live Demo

[ZamVoice — Hugging Face Space](https://huggingface.co/spaces/buumba641/ZamVoice?utm_source=chatgpt.com)

Users can upload or record speech in Bemba, Nyanja, or Tonga and obtain a transcription, with optional English translation.

---

# 🤗 Models and Dataset

### Fine-tuned Models

The trained checkpoints are available on:

[Buumba Chinjila — Hugging Face Models](https://huggingface.co/buumba641?utm_source=chatgpt.com)

Best-performing checkpoints:

| Language | Model                          | Hugging Face                                    |
| -------- | ------------------------------ | ----------------------------------------------- |
| Nyanja   | Whisper Medium + Shona proxy   | `buumba641/Nyanja-whisper-medium-shona-proxy`   |
| Tonga    | Whisper Medium + Shona proxy   | `buumba641/tonga-whisper-medium-shona-proxy`    |
| Bemba    | Whisper Medium + Swahili proxy | `buumba641/bemba-whisper-medium-siwahili-final` |

### Dataset

The prepared multilingual dataset is available on Hugging Face:

[Zambia MultiLingual ASR Dataset](https://huggingface.co/datasets/buumba641/Zambia-MultiLingual-ASR-Dataset?utm_source=chatgpt.com)

The original speech corpora are maintained by the UNZA Speech Lab.

---

# 📚 Datasets and Sources

| Language | Corpus               | Source          |
| -------- | -------------------- | --------------- |
| Bemba    | BembaSpeech          | UNZA Speech Lab |
| Nyanja   | Zambezi Voice Nyanja | UNZA Speech Lab |
| Tonga    | Zambezi Voice        | UNZA Speech Lab |

Research paper:

[Zambezi Voice: A Multilingual Speech Corpus for Zambian Languages](https://arxiv.org/abs/2306.04428?utm_source=chatgpt.com)

The repository does **not** redistribute the original corpus files.

Users should obtain the datasets from their official sources and comply with their respective licences.

---

# 🛠️ Environment

Recommended environment:

* Python 3.10+
* PyTorch
* CUDA-enabled GPU where available
* Hugging Face Transformers
* Hugging Face Datasets
* Evaluate
* Accelerate
* JiWER
* Jupyter / Google Colab

Install the main dependencies:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers datasets evaluate accelerate jiwer huggingface_hub
```


# ▶️ Running the Experiments

Clone the repository:

```bash
git clone https://github.com/buumba641/Low-Resource-Multilingual-ASR-for-Zambian-Languages-Nyanja-Tonga-and-Bemba.git
cd Low-Resource-Multilingual-ASR-for-Zambian-Languages-Nyanja-Tonga-and-Bemba
```

Then select the language you want to investigate.

For example:

```text
Nyanja/
    ↓
ASR_NYANJA_WHISPER_medium_proxy_shona_clean.ipynb
```

or:

```text
Tonga/
    ↓
ASR_TONGA_WHISPER_medium_proxy_shona_clean.ipynb
```

or:

```text
Bemba/
    ↓
ASR_BEMBA_WHISPER_medium_proxy_swahili_clean.ipynb
```

Open the notebook using Jupyter or Google Colab and follow the cells sequentially.

For comparative evaluation, use:

```text
Evaluation/
    ↓
Evaluation_all.ipynb
```

---

# ⚠️ Computational Requirements

Fine-tuning modern speech models is computationally expensive.

The experiments were conducted under student-level computing constraints, including limited GPU availability.

Consequently:

* Model sizes were selected with available GPU memory in mind.
* Training configurations may need to be adjusted for different hardware.
* Google Colab or another CUDA-enabled environment is recommended.
* Large models may require reduced batch sizes or gradient accumulation.

The repository is therefore intended primarily as a **research implementation and reproducibility resource**, rather than a turnkey production training system.

---

# 🔬 Research Scope

The project focuses on:

* Monolingual ASR
* Bemba, Nyanja, and Tonga
* Transfer learning
* Pre-trained speech models
* Low-resource speech recognition
* WER and CER evaluation
* Character-level error analysis
* Proxy-language conditioning for Whisper

The project does **not** attempt to:

* Train a large ASR model from scratch
* Build a universal ASR model for all 72+ Zambian languages
* Provide production-level speech recognition for every Zambian language
* Treat the experimental results as representative of all speakers or dialects

---

# 🔒 Ethics, Privacy, and Licensing

The research uses publicly available research speech corpora.

The repository does not intentionally redistribute the original dataset files.

Users should follow the licence and usage conditions of:

* Zambezi Voice
* BembaSpeech
* Whisper
* MMS
* Wav2Vec2 XLS-R
* WavLM

When using or redistributing derived models, users should also check the licensing requirements of the corresponding base model and training data.

---

# 🙏 Acknowledgements

This project builds on the work of:

* **UNZA Speech Lab** — Zambezi Voice and BembaSpeech
* **OpenAI** — Whisper
* **Meta** — MMS and Wav2Vec2 XLS-R
* **Microsoft** — WavLM
* **Hugging Face** — Transformers, Datasets, Hub, and Spaces

---

# 👨‍💻 Project Information

| Field             | Details                                 |
| ----------------- | --------------------------------------- |
| **Student**       | Buumba Chinjila                         |
| **Institution**   | University of Zambia                    |
| **School**        | School of Natural and Applied Sciences  |
| **Department**    | Department of Computing and Informatics |
| **Project**       | Final Year Research Project             |
| **Academic Year** | 2026                                    |
| **Proposal Date** | 20 March 2026                           |
| **Hugging Face**  | `buumba641`                             |
| **Contact**       | `buumbachinjla@gmail.com`               |

---

# 🔗 Links

* **Source Code:** `github.com/buumba641/Low-Resource-Multilingual-ASR-for-Zambian-Languages-Nyanja-Tonga-and-Bemba`
* **Demo:** [ZamVoice](https://huggingface.co/spaces/buumba641/ZamVoice?utm_source=chatgpt.com)
* **Models:** [Hugging Face — buumba641](https://huggingface.co/buumba641?utm_source=chatgpt.com)
* **Dataset:** [Zambia MultiLingual ASR Dataset](https://huggingface.co/datasets/buumba641/Zambia-MultiLingual-ASR-Dataset?utm_source=chatgpt.com)
* **UNZA Speech Lab:** [UNZA Speech Lab on GitHub](https://github.com/unza-speech-lab?utm_source=chatgpt.com)
* **Zambezi Voice Paper:** [Zambezi Voice paper](https://arxiv.org/abs/2306.04428?utm_source=chatgpt.com)

---

## 📄 Citation

If you use this repository or its experimental results in your research, please cite the project and the original datasets/models used.

**Project:**

```text
Low-Resource Automatic Speech Recognition for Zambian Languages:
A Comparative Analysis of Pre-Trained Models on Bemba, Nyanja, and Tonga.
University of Zambia.
```

---

## 📬 Contact

For questions, collaboration, or research discussions:

**Buumba Chinjila**
University of Zambia
`buumbachinjla@gmail.com`

You can also open an issue in this repository for technical questions or reproducibility issues.
