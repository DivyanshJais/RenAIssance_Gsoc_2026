# RenAIssance GSoC 2026 — Test II
## Handwritten Historical Document Transcription Pipeline

**Author:** Divyansh Jaiswal  
**Test:** Test II — Text Recognition of Handwritten Sources  
**Organization:** HumanAI 

---

## Overview

An end-to-end OCR pipeline for transcribing early modern handwritten Spanish manuscripts. The pipeline uses CRAFT for layout-aware word detection and TrOCR (a Vision Encoder-Decoder transformer) as the core VLM for handwritten text recognition — embedded at every stage of the pipeline, not just as a post-processing step.

**Results on 5 handwritten sources:**

| Source | Period | CER | WER |
|--------|--------|-----|-----|
| AHPG-GPAH 1.1716 | 1744 | 3.2% | 3.8% |
| AHPG-GPAH AU61.2 | 1606 | 8.4% | 10.3% |
| ES.28079.AHN.INQUISICIÓN | 1640 | 9.2% | 11.5% |
| PT3279.146.342 | 1857 | 10.0% | 10.7% |
| Pleito entre el Marqués de Viana | ~1600s | 10.1% | 12.7% |
| **Average** | | **8.2%** | **9.8%** |

---

## Pipeline Architecture

![Pipeline Architecture](images/TrOCR_Architecture.png)

The pipeline consists of 11 stages:

1. **PDF → Images** — PyMuPDF converts each PDF page to PNG
2. **Two-page spread detection** — aspect ratio > 1.3 triggers mid-split into two single pages
3. **Preprocessing** — denoise (fast NL means) + background removal (median blur division) + CLAHE contrast enhancement
4. **CRAFT word detection** — detects word-level bounding boxes per page
5. **Bounding box sorting** — auto-computed resolution-aware grouping threshold from median Y-gap between word centers
6. **GT extraction** — reads `.docx` ground truth directly, collecting lines after `PDF p1` marker
7. **Band-based line crop extraction** — divides text area into N bands (N = GT line count), merges all word boxes per band into one line crop, pairs band i with GT line i
8. **Data augmentation** — 9x per crop (rotation, brightness, noise, blur, elastic distortion) expanding ~120 pairs to ~1200
9. **Pad and resize** — normalize all crops to 384×96 preserving full line content without cutting off text
10. **TrOCR finetuning** — `microsoft/trocr-base-handwritten` finetuned on manuscript line crops, lazy loading to prevent RAM crashes
11. **CER/WER evaluation** — page-level evaluation against GT page 1 per source

---

## Training Results

### Learning Curve

![Learning Curve](images/training_curve.png)

| Epoch | Train Loss | Val Loss | Val CER |
|-------|-----------|----------|---------|
| 1 | 1.948 | 1.329 | 75.3% |
| 3 | 0.133 | 0.089 | 26.1% |
| 5 | 0.023 | 0.041 | 16.3% |
| 7 | 0.001 | 0.019 | 11.9% |
| **8** | **0.001** | **0.015** | **10.5% ← best saved** |
| 10 | 0.0001 | 0.014 | 11.5% |

Training setup: 10 epochs | Batch size 4 | Learning rate 5e-5 | 1020 train / 180 eval samples

The model is saved only when eval CER improves — epoch 8 produced the best checkpoint at 10.5% eval CER.

---

## Evaluation Results

### CER and WER Per Source

![Evaluation Chart](images/evaluation_chart.png)

All results computed on page 1 GT only — the only annotated pages available per source.

## Repository Structure

```
├── approach1_craft_trocr_pipeline.ipynb   # Main notebook
├── README.md
│
├── images/                                # Images used in README
│   ├── TrOCR_Architecture.png             # Pipeline diagram
│   ├── training_curve.png                 # Loss + CER per epoch
│   └── evaluation_chart.png              # CER/WER bar chart per source
│
├── dataset/                               # 5 handwritten PDF sources
│   ├── AHPG-GPAH 1.1716,A.35 – 1744.pdf
│   ├── AHPG-GPAH AU61.2 – 1606.pdf
│   ├── ES.28079.AHN.INQUISICIÓN,1667,Exp.12 – 1640.pdf
│   ├── PT3279.146.342 – 1857.pdf
│   └── Pleito entre el Marqués de Viana.pdf
│
├── ground_truth/                          # GT transcriptions (.docx)
│   ├── AHPG-GPAH 1.1716,A.35 – 1744_transcription.docx
│   ├── AHPG-GPAH AU61.2 – 1606_transcription.docx
│   ├── ES.28079.AHN.INQUISICIÓN,1667,Exp.12 – 1640_transcription.docx
│   ├── PT3279.146.342 – 1857_transcription.docx
│   └── Pleito entre el Marqués de Viana_transcription.docx
│
└── renaissance/outputs/                               # Saved pipeline outputs
    ├── trocr_finetuned/                   # Finetuned model weights
    ├── boundbox_sorted/                   # CRAFT bounding box files per page
    ├── split_images/                      # Single-page PNGs per source
    ├── word_crops/                        # Line crop images + GT label .txt files
    ├── augmented/                         # Augmented training crops
    └── results/                           # CER/WER evaluation CSVs + charts
```

---

## Key Technical Decisions

**Why band-based line extraction instead of CRAFT grouping?**  
CRAFT detects ~200 word boxes per page but GT has ~24 lines. CRAFT grouping errors (splitting one line into multiple groups) make direct group→GT-line pairing unreliable. The band approach divides the text area into exactly N bands (N = GT line count) and assigns word boxes by Y-position — guaranteeing alignment regardless of CRAFT grouping quality. Validated diagnostically before running the full extraction — all 24 bands for source1 showed correct word counts and GT alignment.

**Why lazy loading dataset?**  
Loading 1200 augmented line crops into RAM simultaneously before training crashes Colab T4 (1200 × 1.7MB ≈ 2GB before model weights). `LazyLineDataset` loads each image only during training batch retrieval — no accuracy cost, no memory crash.

**Resolution-aware bounding box grouping**  
Sources span two resolution groups: 6300×8400px (source1/2/4) and 1300×1800px (source3/5). A fixed pixel threshold fails for both. The auto-computed threshold `median_gap * 0.4` gives ~267px for high-res and ~42px for low-res sources automatically.

**Pad and resize without cropping**  
Early resizing used 200×40 (word-sized) and cropped wide lines — cutting off text. Fixed using `ratio = min(target_width/w, target_height/h)` so the full line fits within 384×96 with white padding on the right, never cropping content.

---

## How to Run

### Requirements
```
Google Colab with T4 GPU
Google Drive with datasets at: MyDrive/Gsoc 2026/
```

### First Time Setup
```
1. Open approach1_craft_trocr_pipeline.ipynb in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run Cell 1 — Mount Drive
4. Run Cell 2 — Install dependencies (~3 minutes)
5. Update SOURCES paths in Cell 4 to match your Drive folder names
6. Run all cells sequentially
```

### CRAFT Setup — Run Once
```python
# Downloads CRAFT weights and scripts to Drive permanently
# All future sessions copy from Drive in ~5 seconds
DRIVE_CRAFT_DIR = '/content/drive/MyDrive/Gsoc 2026/craft'
```

### Session Resume After Colab Disconnects
```
1. Run Cell 1 — Mount Drive
2. Run Cell 2 — Install dependencies
3. Run Cell 3 — Imports + function overrides
4. Run Cell 4 — Config
5. Run Session Restore cell — reloads all outputs from Drive
6. Skip to Cell 18 (Inference) if model is already trained
```

---

## Evaluation Methodology

- **Primary metric:** CER (Character Error Rate) — appropriate for manuscripts with archaic spellings and character confusions (f/s, u/v, long-s)
- **Secondary metric:** WER (Word Error Rate) — captures complete word recognition accuracy
- **Scope:** Page 1 GT only — the only annotated pages per source
- **Text normalization:** lowercase, collapsed whitespace, removed page break markers
- **Library:** `jiwer`

---

## Limitations and Known Issues

- **Data scarcity:** Only 1 GT page per source (~24 lines). Augmentation expands visual variety but cannot introduce new vocabulary from unseen pages.
- **Band boundary errors:** Short lines with few CRAFT detections (e.g. "dos y Confirmacion =") may shift adjacent band assignments. Adaptive band spacing from Y-cluster gaps would fix this.
- **False detections:** Aged paper texture and binding shadows are occasionally detected as word boxes. An ink density filter would reduce these.
- **Inference scope:** Inference currently uses page 1 crops only. Full-document transcription of all pages requires running CRAFT extraction on every page.
- **High resolution sources:** Source1/2/4 at 6300×8400px run significantly slower through CRAFT than the 1300×1800px sources.

---

## Dependencies

```bash
!pip install transformers==4.46.3 torch torchvision --quiet
!pip install pymupdf pdf2image Pillow opencv-python-headless --quiet
!pip install python-docx jiwer pandas matplotlib datasets natsort --quiet
!pip install sentencepiece accelerate albumentations --quiet
```

---

## Acknowledgements

- CRAFT preprocessing pipeline from [ML4SCI DeepLearnHackathon](https://github.com/ML4SCI/DeepLearnHackathon/tree/main/NLPRenaissanceChallenge)
- TrOCR model from [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten)
- Dataset and ground truth provided by HumanAI / RenAIssance project
