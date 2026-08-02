# Claim 1 — GROD Table-1 FPR@95

**Outcome: toy.** The anchored claim concerns Table-1 ViT image benchmarks (21.97% to 0.12% FPR@95). The pinned source has no author implementation/checkpoint, and the literal ViT/CIFAR/ImageNet setup is not locally available. We therefore do **not** verify the reported reduction.

A local CPU 2-D feature-space fixture retains source-inspired outward PCA/LDA-style synthetic OOD centers, Mahalanobis filtering, and an ID/OOD binary loss. It evaluates held-out synthetic OOD with AUROC and FPR@95 for seeds 17, 23, and 29, versus a nearest-ID Mahalanobis baseline. Held-out OOD is never used for generated-outlier training. Raw test points, kept synthetic points, result table, protocol, and hashes are in `outputs/claim1_synthetic_grod_toy/`.

This is neither a ViT run nor a CIFAR/ImageNet/Table-1 reproduction; its metrics cannot be compared to 21.97% or 0.12%.
