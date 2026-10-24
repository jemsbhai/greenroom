"""greenroom: staff-reviewed indexing of long-form youth arts video.

AI4Good Hackathon 2026, Problem Case 1 (The Performance Academy), Track A.

Design rules enforced by this package:
- all media and derived artifacts live outside the repository (see config.data_dir)
- no face crops, face embeddings, voiceprints or identity structures are ever produced
- model outputs are suggestions with explicit uncertainty; staff decisions are the record
"""

__version__ = "0.1.0"
