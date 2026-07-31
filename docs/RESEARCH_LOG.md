# Research Log

This document records the scientific narrative of the Hindi Handwriting AI project. 
Every experiment follows the scientific method: **Observation → Hypothesis → Experiment → Benchmark → Conclusion → Next Experiment**.

*Note: Maintain isolated variables across experiments. Do not change the dataset, model, and optimizer simultaneously.*

---

## Experiment Template

```markdown
# Experiment XXX

## Question
[What specific question are we trying to answer?]

## Motivation
[Why are we running this experiment? What observation led to it?]

## Hypothesis
[What do we expect to happen?]

## Dataset
- Manifest version: 
- Number of writers: 
- Number of samples: 

## Model
- Architecture: 
- Parameters: 

## Configuration
- Optimizer: 
- Learning rate: 
- Batch size: 
- Epochs: 

## Results
- Train Loss: 
- Validation Loss: 
- DTW: 
- Fréchet: 
- Endpoint Error: 

## Qualitative Observations
[What do the rendered SVGs show? Where does the model struggle?]

## Conclusion
[Did the results support the hypothesis?]

## Next Experiment
[What should we isolate and test next based on these findings?]
```

---

## Experiment 000 (Milestone A)

### Question
Can the baseline LSTM learn a deterministic end-to-end mapping from text to canonical trajectories?

### Motivation
To validate that the end-to-end rendering and evaluation pipelines correctly backpropagate gradients.

### Hypothesis
Yes, a standard Encoder-Decoder LSTM using MSE and BCE loss should overfit a synthetic, noise-free dataset.

### Dataset
- Manifest version: Synthetic-5 (Deterministic sine waves)
- Number of writers: 0 (Algorithmic)
- Number of samples: 5

### Model
- Architecture: Baseline S2S LSTM (No MDN)
- Parameters: ~300k

### Configuration
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 5
- Epochs: 2

### Results
- Train Loss: 4.314
- Validation Loss: N/A
- DTW: N/A
- Fréchet: N/A
- Endpoint Error: N/A

### Qualitative Observations
Training converged rapidly. Rendered SVGs matched the mathematical ground truth.

### Conclusion
The pipeline is functionally sound. The LSTM successfully maps tokens to geometry.

### Next Experiment
Collect genuine human data and test if the same architecture can overfit real biological variance.
