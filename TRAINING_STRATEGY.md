# Advanced Training Strategy

To effectively teach the Generator both the fundamental structure of Devanagari characters and the nuances of human handwriting style, training must be split into two distinct stages.

## Stage A: Pretraining (Character Geometry)
**Dataset:** Writer-independent character dataset (Massive synthetic or generic font-based trajectories).
**Objective:** Teach the sequence-to-sequence model the geometric rules, stroke order, and topology of the Devanagari script.

- **Process:** The model learns a direct mapping from `Text Sequence → Trajectory`.
- **Identity Agnostic:** This dataset explicitly contains NO writer identity.
- **Result:** The model can generate perfect, rigid, "standard" Hindi handwriting, but it lacks human flair and variance.

## Stage B: Fine-tuning (Writer Style)
**Dataset:** Writer-specific online trajectories (Requires massive collection of distinct human identities).
**Objective:** Teach the model handwriting style without destroying its fundamental knowledge of character geometry.

- **Process:** The model learns a conditional mapping: `Text Sequence + Style Embedding → Trajectory`.
- **Mechanism:**
  1. The Style Encoder produces a `Style Embedding` from a reference input.
  2. The `Style Embedding` is concatenated or cross-attended with the `Text Sequence` embeddings at every timestep of the Generator.
  3. **Crucial:** Early layers of the Generator (which learned the raw geometric rules in Stage A) must be heavily regularized or frozen. Only the deeper layers or style-conditioning blocks are fine-tuned.
- **Result:** The model outputs legible Devanagari characters that perfectly mimic the specific slant, thickness, and stylistic choices of the reference writer.

By isolating the "What to write" (Stage A) from the "How to write it" (Stage B), the network avoids mode collapse and content/style entanglement.
