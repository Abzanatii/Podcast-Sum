# Podcast-Sum

Abstractive podcast summarization using Whisper for transcription and a fine-tuned BART model for generating summaries.

Podcast episodes are long and often lack a useful description. This project transcribes the audio, selects the most informative parts of the transcript with a learned chunk classifier, and generates a short summary with BART fine-tuned on the [TREC Spotify Podcast Dataset](https://podcastsdataset.byspotify.com/).

> Project by Abdelrahman Zanatii, supervised by Prof. Wael Hassan.

## Features

- Speech-to-text with OpenAI Whisper (`base.en`)
- Semantic chunking of transcripts with Sentence-BERT
- Neural chunk classifier to keep only the most relevant transcript segments
- BART (`facebook/bart-large-cnn`) fine-tuned for podcast summarization
- Evaluation with BERTScore and ROUGE
- Gradio demo: upload audio, upload a text file, or paste text

## How It Works

```
Audio → Whisper (ASR) → Transcript → Semantic chunking (SBERT)
      → Chunk classifier (keep relevant chunks, ~1024 tokens)
      → Fine-tuned BART → Summary
```

**Training data preparation.** Episode descriptions are used as reference summaries, so they are cleaned first:

- Dropped descriptions that are missing, shorter than 50 or longer than 700 characters, or not in English
- Removed sponsor, social media, link, and emoji sentences
- Removed low-salience sentences using an average-IDF score
- Removed episodes whose description is too similar to the show description (> 60%) or to another episode's description in the same show (> 70%)
- Replaced descriptions with the best-rated summary from the 150-episode gold set where available

**Chunk selection.** Transcripts are split into semantic chunks. A chunk is labeled "useful" if its ROUGE-L against the reference summary is at least 0.20. A small Keras MLP is trained on the chunk embeddings to predict this label, and the top-ranked chunks are packed into BART's 1024-token input window.

**Summarization.** BART is fine-tuned for 3 epochs (learning rate 2e-5, batch size 4, max input 1024 / max target 256 tokens). Generation uses 4 beams, a length penalty of 2.0, and no repeated 3-grams.

## Results

Evaluated on the Spotify podcast test set (1,025 episodes after filtering), against the original episode descriptions.

| Metric       | Fine-tuned BART | Pre-trained BART |
|--------------|-----------------|------------------|
| BERTScore F1 | **0.8426**      | 0.8192           |
| BERTScore P  | **0.8611**      | 0.8297           |
| BERTScore R  | **0.8257**      | 0.8096           |
| ROUGE-1      | **0.2370**      | 0.1917           |
| ROUGE-2      | **0.0669**      | 0.0377           |
| ROUGE-L      | **0.1621**      | 0.1223           |

ROUGE relies on n-gram overlap and is not ideal for abstractive summaries, so BERTScore is the main metric.

## Repository Structure

```
Podcast-Sum/
├── Podcast_Sum.ipynb    # Data cleaning, transcript filtering, BART fine-tuning, evaluation
├── ChunkModel.ipynb     # Trains the chunk relevance classifier
├── Prez.pptx            # Project presentation
└── Demo/
    ├── DemoRUN.ipynb    # Gradio demo (audio / text → summary)
    ├── Thegodlydating(podcast episode).mp3
    ├── transcript.txt
    ├── episode description.txt
    └── generated summary.txt
```

## Getting Started

### Requirements

- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (required by Whisper)
- A GPU is recommended for fine-tuning and inference

### Installation

```bash
git clone https://github.com/Abzanatii/Podcast-Sum.git
cd Podcast-Sum
pip install tensorflow transformers sentence-transformers datasets evaluate \
            bert-score rouge nltk pysbd regex scikit-learn pandas numpy \
            matplotlib tqdm openai-whisper gradio
```

### Required Files (not included in this repository)

The notebooks expect the following to be available locally:

- `get_transcripts.py` — helper module providing `get_transcription`, `semantic_segmentation`, and `extract_features`
- The TREC Spotify Podcast Dataset (metadata, transcripts, and the 150-episode gold set), available for research use from [Spotify](https://podcastsdataset.byspotify.com/)
- `ChunkModel.h5` — produced by running `ChunkModel.ipynb`
- `FT_model/` — the fine-tuned BART model and tokenizer, produced by running `Podcast_Sum.ipynb`

## Usage

### Training

1. Run `ChunkModel.ipynb` to train the chunk classifier and save `ChunkModel.h5`.
2. Run `Podcast_Sum.ipynb` to clean the data, filter transcripts, fine-tune BART, and evaluate.

### Demo

With `FT_model/` in the `Demo/` directory, open `Demo/DemoRUN.ipynb` and run all cells to launch the Gradio app. It provides three tabs:

- **Upload audio**: transcribes with Whisper, then summarizes
- **Upload text**: summarizes a transcript from a `.txt` file
- **Input text**: summarizes pasted text

A sample episode and its generated summary are included in `Demo/`.

## Limitations and Future Work

- Reference summaries are creator-written episode descriptions, which are noisy and often promotional
- The demo truncates input to the first 1024 tokens and does not apply the chunk classifier
- Longer training could improve performance
- Named entity recognition could be used to surface key people, places, and topics

## References

- Clifton, A. et al. (2020). *100,000 Podcasts: A Spoken English Document Corpus.* COLING 2020.
- Zheng, C., Zhang, K., Wang, H. J., & Fan, L. (2020). *A Two-Phase Approach for Abstractive Podcast Summarization.* arXiv:2011.08291.
- Saxena, P., & El-Haj, M. (2023). *Exploring Abstractive Text Summarisation for Podcasts: A Comparative Study of BART and T5 Models.* RANLP 2023.
- Lewis, M. et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.* ACL 2020.
- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). arXiv:2212.04356.
