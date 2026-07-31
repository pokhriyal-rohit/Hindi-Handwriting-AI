# Archived Datasets

The following datasets are deliberately excluded from the canonical `data/canonical/` folder and the `v1.1.0-colab-ready` release archive to conserve upload bandwidth to Colab. These datasets contain offline (static image) handwriting samples, which are fundamentally incompatible with this project's online (temporal sequence) trajectory-based architecture.

They are recorded here so their existence and provenance are not lost to history.

## 1. IIIT-HW-Hindi
- **Source**: CVIT (Center for Visual Information Technology) at IIIT Hyderabad
- **Description**: A massive offline dataset containing cropped images of handwritten Hindi words.
- **Reason Excluded**: Purely image-based (offline). Lacks `(x, y, t)` trajectory points and pen states.
- **Reference URL**: [http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data](http://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data)

## 2. Kaggle Devanagari Character Dataset
- **Source**: Kaggle (various contributors)
- **Description**: Offline datasets of isolated Devanagari characters and digits stored as PNG images.
- **Reason Excluded**: Image-based (offline) and isolated characters, rather than continuous cursive/connected trajectories.
- **Reference URL**: Assorted Kaggle sources for Devanagari character recognition.

## Data Restoration
If future phases of this project incorporate offline-to-online trajectory reconstruction models (e.g., extracting synthetic trajectories from static images), these datasets can be re-downloaded into the `data/archive/` or `data/raw/offline/` directory. For now, the canonical pipeline strictly relies on the online data collected via the `custom_hindi` collector.
