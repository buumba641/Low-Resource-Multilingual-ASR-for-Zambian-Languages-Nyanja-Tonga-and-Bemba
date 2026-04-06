# Data Directory

Place raw audio and transcription files here under the appropriate language sub-directory.

## Expected Layout

```
data/
├── nyanja/
│   ├── audio/           ← .wav / .flac / .mp3 files
│   └── transcriptions/  ← .txt files with the same stem as the audio files
├── tonga/
│   ├── audio/
│   └── transcriptions/
└── bemba/
    ├── audio/
    └── transcriptions/
```

Each `.txt` file must contain a single transcription line matching the corresponding audio file.

## Alternative: Manifest CSV

You can also place a `manifest.csv` file directly inside the language directory:

```
data/nyanja/manifest.csv
```

with columns:

```
audio_path,transcription
data/nyanja/audio/utt001.wav,ndiyo bwino
```

## Recommended Audio Format

- Format: WAV (16-bit PCM)
- Sample rate: 16,000 Hz (mono)
- Duration: 0.5–20 seconds per utterance

## Data Sources

Zambian language speech data can be obtained from:
- [Mozilla Common Voice](https://commonvoice.mozilla.org/) (check for available Nyanja/Bemba/Tonga data)
- [ALFFA Project](https://github.com/besacier/ALFFA_PUBLIC)
- Local university language labs and community recording sessions
