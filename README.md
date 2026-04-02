# Thesis Paper — Multimodal Sarcasm Detection

## Working Title

**Multimodal Sarcasm Detection with Cue Learning: Reproduction, Adaptation, and Low-Resource Extensions**

---

## Problem Statement

Sarcasm often relies on incongruity between what is said and what is meant. In multimodal contexts (text + image), the sarcasm may arise from the mismatch between the text and the accompanying image. Detecting this requires models that can jointly reason over both modalities. This thesis aims to reproduce and improve upon cue-learning-based multimodal sarcasm detection, with potential extensions to low-resource languages like Bengali.

---

## Possible Contributions

- Reproduce a simplified version of the cue-learning idea from the reference paper
- Compare text-only vs image-only vs multimodal pipelines
- Compare simple CLIP fusion vs prompt/cue-enhanced version
- Provide ablation study and error analysis
- Optionally add a small Bengali/Banglish pilot discussion

---

## Key Concepts to Understand

- Sarcasm as incongruity (text says one thing, meaning is opposite)
- Multimodal learning basics (combining text + image representations)
- CLIP (Contrastive Language-Image Pre-training) and shared embedding spaces
- Feature fusion strategies (concatenation, attention, cross-modal)
- Prompt tuning / cue learning intuition
- Cosine similarity for matching text-image pairs
- Few-shot / low-resource thinking
- Ablation design and error analysis methodology

---

## Reading List

### Core Reference Paper
- [A multi-modal sarcasm detection model based on cue learning](https://www.nature.com/articles/s41598-025-94266-w.pdf) — **Read this thoroughly, especially the experiment section**

### Multimodal Sarcasm Datasets & Benchmarks
- [MMSD2.0 paper (ACL Findings 2023)](https://aclanthology.org/2023.findings-acl.689/)
- [MMSD2.0 repository](https://github.com/joeying1019/mmsd2.0)
- [MMSD3.0](https://arxiv.org/abs/2510.23299)

### CLIP & Multimodal Foundations
- [OpenAI CLIP repository](https://github.com/openai/CLIP)
- [A multimodal world — HuggingFace course](https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/a_multimodal_world)

### Prompt Tuning & Soft Prompts
- [HuggingFace PEFT — Prompt Tuning](https://huggingface.co/docs/peft/main/en/package_reference/prompt_tuning)
- [Soft prompting concepts](https://huggingface.co/docs/peft/main/en/conceptual_guides/prompting)

### Transformer Fundamentals
- [HuggingFace LLM Course — Ch1.2: NLP overview](https://huggingface.co/learn/llm-course/en/chapter1/2)
- [HuggingFace LLM Course — Ch1.3: What can transformers do?](https://huggingface.co/learn/llm-course/chapter1/3)
- [HuggingFace LLM Course — Ch1.4: How transformers work](https://huggingface.co/learn/llm-course/en/chapter1/4)

### Metrics & Evaluation
- [Google ML Crash Course — Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [Accuracy, Precision, Recall](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

### Reproducibility & Writing
- [ACL reproducibility checklist](https://aclanthology.org/attachments/2025.findings-emnlp.1404.checklist.pdf)

---

## Video Resources

### NLP & Embeddings
- [NLP intro](https://youtu.be/fLvJ8VdHLA0?si=QgK5-wWPDIvVBS9o)
- [Tokenization](https://youtu.be/YdreZtH8oWk?si=XnwgJUBV9BCRomva)
- [Vector Embedding (short)](https://youtube.com/shorts/FJtFZwbvkI4?si=_9O9ZUEK5bF_9eHE)
- [Vector Embedding (explained)](https://youtu.be/dN0lsF2cvm4?si=f9oNtbwSHmsMJLSN)

### Sequence Models (background)
- [RNN explained](https://youtu.be/AsNTP8Kwu80?si=HSkM9MOoSNQlpbNT)
- [RNN implementation](https://youtu.be/0_PgWWmauHk?si=xC67ecbKAze2FD13)
- [LSTM](https://youtu.be/b61DPVFX03I?si=2R83brFKALCbcBLZ)

### Prompt / Cue Learning
- [Prompt tuning overview](https://youtu.be/JgnbwKnHMZQ?si=VGkSGGG79Fq4R88F)
- [Related video 1](https://youtu.be/5HQCNAsSO-s?si=pzWjxyfnoiplFPse)
- [Related video 2](https://youtu.be/4YGkfAd2iXM?si=_RFMwVQvnLreFPbg)

---

## Tech Stack

- Python, PyTorch
- HuggingFace Transformers + PEFT
- CLIP / OpenCLIP
- scikit-learn
- pandas, numpy, matplotlib
- Weights & Biases (optional, for experiment tracking): [quickstart](https://docs.wandb.ai/get-started)

---

## Minimum Model Targets

1. **Text-only baseline** — CLIP text encoder + classifier head
2. **Image-only baseline** — CLIP image encoder + classifier head
3. **Simple multimodal baseline** — concatenated CLIP text + image features + MLP
4. **Improved cue/prompt-inspired model** — class-specific prompt templates, similarity-based scoring, or learned cue vectors on top of CLIP embeddings

---

## Improvement Ideas (pick one or combine)

- Prompt-like textual cues for sarcasm / non-sarcasm classes
- Handcrafted sarcasm cue templates with cosine similarity scoring
- Similarity-based selection of prompt candidates
- Multiple text prompts with ensemble scores
- Class-specific learned vectors on top of CLIP text embeddings

---

## Ablation Experiments to Run

- Text-only vs image-only vs multimodal
- Simple CLIP fusion vs cue/prompt-enhanced version
- Different numbers of prompt templates
- Frozen CLIP features vs trainable classifier head
- With/without cue learning component

---

## Error Analysis Checklist

- Where does the image help that text alone misses?
- Where is text alone sufficient?
- Where is sarcasm culturally dependent or ambiguous?
- Label ambiguity cases (annotator disagreement)
- Failure modes of the multimodal model

---

## Paper Draft Structure

1. Title
2. Abstract
3. Introduction
4. Related Work
5. Methodology (architecture, cue learning, fusion)
6. Experimental Setup (dataset, splits, hyperparameters)
7. Results (baseline comparison table)
8. Ablation Study
9. Error Analysis
10. Limitations
11. Conclusion and Future Work (Bengali/Banglish extension)

---

## What NOT to Waste Time On

- Completing entire ML playlists end-to-end
- Advanced theory not needed for implementation
- Building a new Bengali sarcasm dataset from scratch (out of scope for now)
- Trying too many model variants without documenting results
- Rewriting code repeatedly without saving experiments
