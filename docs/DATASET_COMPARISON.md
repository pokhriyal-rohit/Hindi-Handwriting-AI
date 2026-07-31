# Dataset Comparison & Research

This document tracks the research and comparison of available handwriting datasets suitable for Hindi (Devanagari) and other Indic scripts. The primary goal is to find high-quality **online (trajectory-based)** datasets required for a generative handwriting model.

## Comparison Matrix

| Dataset | Script(s) | Online (Trajectories) | Offline (Images) | Writer IDs | Pressure | License | Recommended |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **HP Labs / MILE Lab Indic** | Devanagari, Tamil, Telugu | ✅ | ❌ | ✅ | ❌ | Academic/Non-commercial | ⭐⭐⭐⭐☆ |
| **LipiTK Framework Datasets** | Multi-Indic (incl. Hindi) | ✅ | ❌ | ✅ | ❌ | Open Source | ⭐⭐⭐⭐☆ |
| **IIIT-INDIC-HW-WORDS** | 10 Indic Scripts | ❌ | ✅ | ✅ | ❌ | Non-commercial | ⭐⭐☆☆☆ |
| **DohaScript** | Devanagari | ❌ | ✅ | ✅ | ❌ | Research | ⭐⭐☆☆☆ |
| **DHCD (Devanagari Char)** | Devanagari | ❌ | ✅ | ❌ | ❌ | Open | ⭐☆☆☆☆ |
| **AnciDev** | Devanagari (Historical) | ❌ | ✅ | ❌ | ❌ | Research | ⭐☆☆☆☆ |

## Findings

1. **Scarcity of Large-Scale Online Data:** Unlike English (which has the massive IAM-OnDB dataset for continuous sentence trajectories), true online datasets for Devanagari and other Indic scripts are historically scarce. 
2. **Isolated Characters/Words vs. Sentences:** The available online datasets (like HP Labs/MILE Lab) primarily consist of isolated characters and isolated words, rather than continuous, multi-line sentences. This is sufficient to learn strokes and characters, but limits the ability to learn natural word-spacing and line-spacing.
3. **Offline Dominance:** Modern large-scale datasets (like IIIT-INDIC-HW-WORDS with 872K instances) are strictly offline (image-based) because they are easier to crowdsource.

## Conclusion & Recommendation

The **HP Labs / MILE Lab Indic Handwriting Datasets** (or those managed within the **LipiTK** framework) represent the best available true online trajectory data for Devanagari. 

However, they are not immediately accessible via open URLs (often requiring academic request forms) and they focus on isolated characters/words. 

**Recommendation:** We should attempt to acquire the MILE Lab / LipiTK data. If it cannot be acquired rapidly for Phase 3 engineering, we must implement **Stage 1: Synthetic Bootstrap**, using a temporary synthetic trajectory generator derived from Devanagari TTF fonts. This will allow the engineering pipeline (tokenizers, renderers, models) to be fully built and validated while the real dataset acquisition is pending.
