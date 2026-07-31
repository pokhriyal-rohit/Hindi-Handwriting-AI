# Project Benchmarks

This document serves as the permanent, empirical history of our architectural experiments. 
Every model improvement will be justified by appending results to this table.

## Terminology
- **Synthetic**: Generated entirely from fonts or procedural methods (e.g. sine waves).
- **Augmented**: Derived from an existing trajectory through transformations (noise, scaling, timing jitter).
- **Human**: Recorded from an actual person using the collector or an online handwriting dataset.

---

## 1. Synthetic Benchmarks

| Model         | Dataset               | Params | DTW | Fréchet | Train Loss | Inference Time | Notes    |
| ------------- | --------------------- | -----: | --: | ------: | ---------: | -------------: | -------- |
| Baseline LSTM | Synthetic-5           |  ~300k | N/A |     N/A |      4.314 |       ~150ms   | Milestone A (Overfit baseline) |

---

## 2. Augmented Benchmarks

| Model         | Dataset               | Params | DTW | Fréchet | Train Loss | Inference Time | Notes    |
| ------------- | --------------------- | -----: | --: | ------: | ---------: | -------------: | -------- |
| Baseline LSTM | Augmented-Pilot (50)  |  ~300k | N/A |     N/A |      1.695 |       ~155ms   | Milestone B Smoke Test |

---

## 3. Real Human Benchmarks

| Model         | Dataset               | Params | DTW | Fréchet | Train Loss | Inference Time | Notes    |
| ------------- | --------------------- | -----: | --: | ------: | ---------: | -------------: | -------- |
| (Pending)     |                       |        |     |         |            |                |          |
