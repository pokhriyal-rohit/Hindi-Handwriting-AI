# Dataset Compatibility Report

This report evaluates the compatibility of the currently downloaded/available datasets in our repository against the strict architectural requirements of the **Hindi Handwriting AI** generative framework.

## Evaluated Datasets
1. `data/raw/IIIT-HW-Hindi_v1`
2. `data/raw/devanagari+handwritten+character+dataset`
3. `data/raw/Dataset_hindi_character-20240412T210635Z-001`

## Compatibility Checklist

| Requirement | Supported? | Details |
| :--- | :---: | :--- |
| **Contains Online Trajectories? (X,Y,T)** | ❌ | **FAIL**. The dataset analysis module confirmed 100% of the files are offline `.jpg` images. |
| **Contains Writer IDs?** | ✅ | PASS. The `IIIT-HW-Hindi_v1` dataset structure organizes images by writer ID folders. |
| **Pen Pressure Available?** | ❌ | FAIL. Offline images do not capture pressure data natively without highly complex inverse rendering. |
| **Time Stamps Available?** | ❌ | FAIL. No temporal data exists in static images. |
| **Script Labels?** | ✅ | PASS. Text transcripts and vocab mappings are provided in `hindi_vocab.txt`. |
| **Sequence Statistics?** | ❌ | FAIL. Impossible to extract sequence lengths, strokes-per-character, or delta distributions from images directly. |
| **Suitable for Primary Generative Training?** | ❌ | **CRITICAL FAIL**. Trajectory generation requires trajectory ground truth. |

## Conclusion
The currently downloaded datasets are **incompatible** with Phase 3 (Coordinate Representation) and beyond. 

Because we do not have a sufficiently large, accessible online Devanagari dataset at this exact moment, we must proceed with **Stage 1: Bootstrap (Synthetic)**. We will build a temporary synthetic trajectory generator derived from Devanagari TrueType fonts to unblock the engineering pipeline.
