# Project Benchmarks

This document serves as the permanent, empirical history of our architectural experiments. 
Every model improvement will be justified by appending results to this table.

| Model         | Dataset               | Params | DTW | Fréchet | Train Loss | Inference Time | Notes    |
| ------------- | --------------------- | -----: | --: | ------: | ---------: | -------------: | -------- |
| Baseline LSTM | Synthetic-5           |  ~300k | N/A |     N/A |      4.314 |       ~150ms   | Milestone A (Overfit baseline) |
