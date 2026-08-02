# Status
- OpenReview ID: `94FOsjgeHK`; 6 anchored live claims / 12 maximum points.
- Phase: Claim 1 source/table CPU audit complete, **inconclusive**.
- Source: pinned arXiv 2406.12915 PDF/source; manifest in `evidence/source/SHA256SUMS`.
- Claim 1: Table-1 values 21.97% and 0.12% produce 21.85 percentage-point arithmetic reduction. This is source transcription only, not a reproduction or verification. Evidence: `outputs/claim1_source_audit/`.
- Compute: local CPU/local GTX 1050 only; no HF cpu-upgrade, Jobs, paid, or remote compute.
- Next: source-audit feasibility of a small local OOD synthetic method fixture; ImageNet/BERT claims require resource audit.
- Claim 1 attempt 2: **toy** local CPU 2-D feature-space GROD-style fixture preserves outward synthetic-OOD center generation, Mahalanobis ID-like filtering, and binary ID/OOD loss. It reports held-out synthetic AUROC/FPR@95 for seeds 17/23/29 against a nearest-ID Mahalanobis control, with raw arrays and hashes retained. It is not ViT, CIFAR, ImageNet, or a Table-1 reproduction; no reported 21.97%/0.12% claim is verified. Evidence: `outputs/claim1_synthetic_grod_toy/`.
