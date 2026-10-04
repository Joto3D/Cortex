"""Contrastive vision-language encoders.

Everything downstream only needs an object with two methods, so the CLIP
backend can be swapped (torch/MPS today, Core ML or a distilled CNN later):

    encode_images(images: uint8 array (N, H, W, 3)) -> float32 (N, D), L2-normalised
    encode_text(texts: list[str])                   -> float32 (N, D), L2-normalised
"""
from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np

log = logging.getLogger(__name__)

# OpenAI CLIP normalisation, used if the model's preprocess can't be introspected.
_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


class Encoder(Protocol):
    def encode_images(self, images: np.ndarray) -> np.ndarray: ...
    def encode_text(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, 1e-12)


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClipEncoder:
    """open_clip model (MobileCLIP by default) running on MPS / CUDA / CPU.

    Images are resized with nearest-neighbour interpolation, which keeps pixel-art
    edges crisp, and normalised on the GPU. This skips the PIL preprocessing
    pipeline, which is far too slow to run per tile.
    """

    def __init__(
        self,
        model_name: str = "MobileCLIP-S1",
        pretrained: str = "datacompdr",
        device: str | None = None,
        half: bool | None = None,
        batch_size: int = 256,
        fallback: tuple[str, str] | None = ("ViT-B-32", "laion2b_s34b_b79k"),
    ):
        import open_clip
        import torch

        self._torch = torch
        self.device = device or pick_device()
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        except Exception as e:  # unknown model tag, no network, etc.
            if not fallback:
                raise
            log.warning("could not load %s/%s (%s); falling back to %s/%s", model_name, pretrained, e, *fallback)
            model_name, pretrained = fallback
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model_name = model_name
        self.tokenizer = open_clip.get_tokenizer(model_name)
        model = _maybe_reparameterize(model.eval())

        self.half = (self.device != "cpu") if half is None else half
        self.dtype = torch.float16 if self.half else torch.float32
        self.model = model.to(self.device, dtype=self.dtype)
        self.batch_size = batch_size

        size, mean, std = _preprocess_params(preprocess, model)
        self.image_size = size
        self._mean = torch.tensor(mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        log.info("loaded %s on %s (input %s, %s)", model_name, self.device, size, self.dtype)

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        torch = self._torch
        F = torch.nn.functional
        if images.ndim == 3:
            images = images[None]
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                chunk = np.ascontiguousarray(images[i : i + self.batch_size, :, :, :3])
                x = torch.from_numpy(chunk).to(self.device).permute(0, 3, 1, 2).to(self.dtype) / 255.0
                x = F.interpolate(x, size=self.image_size, mode="nearest")
                x = (x - self._mean) / self._std
                f = self.model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                out.append(f.float().cpu().numpy())
        if not out:
            return np.zeros((0, 0), np.float32)
        return np.concatenate(out)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            tokens = self.tokenizer(list(texts)).to(self.device)
            f = self.model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()


def _maybe_reparameterize(model):
    """MobileCLIP uses re-parameterisable blocks; fuse them for faster inference."""
    try:
        from timm.utils.model import reparameterize_model  # type: ignore
    except Exception:
        return model
    try:
        return reparameterize_model(model)
    except Exception as e:  # pragma: no cover - best effort
        log.debug("reparameterize_model failed: %s", e)
        return model


def _preprocess_params(preprocess, model) -> tuple[tuple[int, int], tuple, tuple]:
    size = None
    mean, std = _DEFAULT_MEAN, _DEFAULT_STD
    for t in getattr(preprocess, "transforms", []):
        name = type(t).__name__
        if name == "Normalize":
            mean, std = tuple(t.mean), tuple(t.std)
        elif name == "CenterCrop" or (name == "Resize" and size is None):
            # CenterCrop (the final model input size) wins over Resize.
            s = getattr(t, "size", None)
            if s is not None:
                size = (s, s) if isinstance(s, int) else tuple(s)
    if size is None:
        s = getattr(model.visual, "image_size", 224)
        size = (s, s) if isinstance(s, int) else tuple(s)
    return size, mean, std
