# Next Task

**Target:** Phase B3 - Multi-writer Expansion

## Objectives
1. **Data Collection Campaign**: The fundamental blocking issue for Phase 3 (Style Encoding) is extreme dataset sparsity (only 13 distinct writers total, with 0 overlap between online and offline domains).
2. **Action Plan**:
   - Collect paired online and offline writing samples from a minimum of 200–500 distinct Devanagari writers.
   - Utilize standard digital tablets for trajectory (online) logging.
   - Utilize standard paper+pen (offline) scanning.
3. **Integration**: Ingest the new massive paired datasets via existing `scripts/build_canonical_dataset.py` and `scripts/build_offline_dataset.py` routines.
4. **Resumption**: Once the massive multi-writer dataset is fully ingested and verified, Phase 3 Software Engineering (Style Encoder) can officially commence.
