"""Zero-shot classification by comparing image embeddings with text embeddings.

Text prompts are encoded once at start-up. At runtime, classifying an image is
one image embedding plus a matrix multiply against the cached label matrix.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .clip_model import Encoder, l2_normalize

LOGIT_SCALE = 100.0  # CLIP's learned temperature is ~100 for most checkpoints


class ZeroShotClassifier:
    def __init__(
        self,
        encoder: Encoder,
        label_prompts: Mapping[str, Sequence[str]],
        template: str = "{}",
    ):
        if not label_prompts:
            raise ValueError("need at least one label")
        self.encoder = encoder
        self.labels: list[str] = list(label_prompts)
        rows = []
        for name in self.labels:
            prompts = [template.format(p) for p in label_prompts[name]]
            emb = encoder.encode_text(prompts)
            # Prompt ensembling: average the normalised prompt embeddings.
            rows.append(l2_normalize(emb.mean(axis=0)))
        self.text_matrix = np.stack(rows).astype(np.float32)  # (L, D)

    def logits(self, image_embeddings: np.ndarray) -> np.ndarray:
        return LOGIT_SCALE * image_embeddings @ self.text_matrix.T

    def classify_embeddings(self, image_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (label index, softmax probability) for each embedding."""
        if len(image_embeddings) == 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        z = self.logits(image_embeddings)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        idx = p.argmax(axis=1)
        return idx.astype(np.int32), p[np.arange(len(idx)), idx].astype(np.float32)

    def classify(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.classify_embeddings(self.encoder.encode_images(images))
