"""
tf_restormer.utils.metrics.semantic
=====================================
Semantic speech metrics: SpeechBLEU, SpeechBERTScore, SpeechTokenDistance.

- ApplyKmeans and int_array_to_chinese_unicode are private helpers
  (previously duplicated in util_speechbleu.py and util_speechtokendistance.py).
- km/ binary files are resolved via pathlib relative to this module's location:
  pathlib.Path(__file__).parent.parent / "km"  →  tf_restormer/utils/km/
- All scorer classes are cached via _model_cache (one instance per
  (model_type, vocab, layer, device) configuration).
- transformers, nltk, joblib, Levenshtein, jellyfish are lazy-imported.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
from typing import Optional

import numpy as np

from tf_restormer.utils.metrics import _model_cache

# Suppress transformers checkpoint-mismatch warnings (false alarm for PyTorch 2+).
# See: https://github.com/huggingface/transformers/issues/26796
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# km/ path resolution
# ---------------------------------------------------------------------------

# __file__ → .../tf_restormer/utils/metrics/semantic.py
# .parent  → .../tf_restormer/utils/metrics/
# .parent  → .../tf_restormer/utils/
# / "km"   → .../tf_restormer/utils/km/
_KM_DIR = pathlib.Path(__file__).parent.parent / "km"


# ---------------------------------------------------------------------------
# Shared private helpers (deduplicated from speechbleu + speechtokendistance)
# ---------------------------------------------------------------------------

def _int_array_to_chinese_unicode(arr) -> str:
    """Map each integer to a CJK Unified Ideograph (U+4E00 base).

    Unicode CJK region: 4E00–9FFF (20992 characters), sufficient for
    the k-means vocabulary sizes used here (50, 100, 200).
    """
    base = 0x4E00
    return "".join(chr(base + v) for v in arr)


class _ApplyKmeans:
    """Assign speech feature frames to k-means cluster indices.

    Supports both torch.Tensor and np.ndarray inputs (T, D).
    Returns np.ndarray of cluster indices (T,).
    """

    def __init__(self, km_path, device: str):
        import joblib
        import torch

        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        import torch

        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


# ---------------------------------------------------------------------------
# NLTK punkt helper (NLTK 3.7 / 3.9.1 compat)
# ---------------------------------------------------------------------------

def _ensure_nltk_punkt() -> None:
    """Check punkt tokenizer exists, attempt download if missing.

    Handles both NLTK < 3.9.1 ('punkt') and >= 3.9.1 ('punkt_tab').
    Note: NLTK < 3.8 raises OSError (not LookupError) for missing resources.
    """
    import nltk

    # NLTK >= 3.9.1 renamed punkt -> punkt_tab
    # Catch both LookupError (NLTK >= 3.8) and OSError (NLTK < 3.8)
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            return  # found — no download needed
        except (LookupError, OSError):
            continue

    # Neither found — attempt download (try new name first)
    # nltk.download() returns False on failure without raising, so
    # verify success by re-checking with nltk.data.find() after each attempt.
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            continue

    # Verify that at least one resource is now available
    for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
        try:
            nltk.data.find(resource)
            return  # download succeeded and resource is usable
        except (LookupError, OSError):
            continue

    raise RuntimeError(
        "nltk punkt tokenizer not found and download failed. "
        "If in an offline environment, manually install nltk data:\n"
        "  python -c \"import nltk; nltk.download('punkt_tab')\"  # NLTK>=3.9.1\n"
        "  python -c \"import nltk; nltk.download('punkt')\"      # NLTK<3.9.1"
    )


# ---------------------------------------------------------------------------
# km/ file resolution helper
# ---------------------------------------------------------------------------

def _resolve_km_path(vocab: int) -> pathlib.Path:
    """Return path to km{vocab}.bin, downloading if not present."""
    if vocab not in (50, 100, 200):
        raise ValueError(
            f"km vocabularies other than 50, 100, 200 are not supported. Got: {vocab}"
        )
    os.makedirs(_KM_DIR, exist_ok=True)
    km_path = _KM_DIR / f"km{vocab}.bin"
    if not km_path.exists():
        url = (
            f"http://sarulab.sakura.ne.jp/saeki/discrete_speech_metrics/km/km{vocab}.bin"
        )
        subprocess.run(["wget", url, "-O", str(km_path)], check=True)
        print(f"Downloaded file from {url} to {km_path}")
    else:
        print(f"Using a cache at {km_path}")
    return km_path


# ---------------------------------------------------------------------------
# Private input conversion helper
# ---------------------------------------------------------------------------

def _to_numpy(x) -> np.ndarray:
    """Return a 1-D float32 numpy array, accepting Tensor or ndarray."""
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy().squeeze()
    return np.asarray(x, dtype=np.float32).squeeze()


# ---------------------------------------------------------------------------
# SpeechBLEU scorer factory (internal, not exported)
# ---------------------------------------------------------------------------

def _build_speechbleu(sr: int, model_type: str, vocab: int, layer: Optional[int],
                       n_ngram: int, device: str):
    """Construct and return a SpeechBLEU instance."""
    import torch
    import torchaudio
    from transformers import HubertModel

    if model_type == "hubert-base":
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        model.eval()
        model.to(device)
    else:
        raise ValueError(f"Not found the setting for {model_type}.")

    km_path = _resolve_km_path(vocab)
    apply_kmeans = _ApplyKmeans(km_path, device=device)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)

    # Encode all state into a namespace-like object
    class _SpeechBLEU:
        pass

    obj = _SpeechBLEU()
    obj.model = model
    obj.device = device
    obj.sr = sr
    obj.layer = layer
    obj.apply_kmeans = apply_kmeans
    obj.n_ngram = n_ngram
    obj.weights = [1.0 / n_ngram] * n_ngram
    obj.resampler = resampler

    def _decode_label(audio):
        audio = audio.to(device)
        if sr != 16000:
            audio = obj.resampler(audio)
        if obj.layer is None:
            outputs = obj.model(audio)
            feats = outputs.last_hidden_state
        else:
            feats_hiddens = obj.model(audio, output_hidden_states=True).hidden_states
            feats = feats_hiddens[obj.layer]
        return obj.apply_kmeans(feats[0, ...]).tolist()

    def _score(gt_wav_np, gen_wav_np):
        from nltk.translate.bleu_score import sentence_bleu

        gt_wav = torch.from_numpy(gt_wav_np).unsqueeze(0).to(device).float()
        gen_wav = torch.from_numpy(gen_wav_np).unsqueeze(0).to(device).float()
        gt_label = _decode_label(gt_wav)
        gen_label = _decode_label(gen_wav)
        gt_text = _int_array_to_chinese_unicode(gt_label)
        gen_text = _int_array_to_chinese_unicode(gen_label)
        return sentence_bleu([gt_text], gen_text, weights=obj.weights)

    obj.score = _score
    return obj


# ---------------------------------------------------------------------------
# SpeechBERTScore scorer factory (internal, not exported)
# ---------------------------------------------------------------------------

def _build_speechbertscore(sr: int, model_type: str, layer: Optional[int], device: str):
    """Construct and return a SpeechBERTScore instance."""
    import torch
    import torchaudio
    from transformers import HubertModel, Wav2Vec2Model, WavLMModel

    if model_type == "hubert-base":
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    elif model_type == "hubert-large":
        model = HubertModel.from_pretrained("facebook/hubert-large-ll60k")
    elif model_type == "wav2vec2-base":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    elif model_type == "wav2vec2-large":
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
    elif model_type == "wavlm-base":
        model = WavLMModel.from_pretrained("microsoft/wavlm-base")
    elif model_type == "wavlm-base-plus":
        model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    elif model_type == "wavlm-large":
        model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    else:
        raise ValueError(f"Not found the setting for {model_type}.")

    model.eval()
    model.to(device)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)

    class _SpeechBERTScore:
        pass

    obj = _SpeechBERTScore()
    obj.model = model
    obj.device = device
    obj.sr = sr
    obj.layer = layer
    obj.resampler = resampler

    def _process_feats(audio):
        if obj.layer is None:
            return obj.model(audio).last_hidden_state
        else:
            return obj.model(audio, output_hidden_states=True).hidden_states[obj.layer]

    def _bert_score(v_generated, v_reference):
        """Compute cosine-similarity-based precision, recall, F1."""
        sim_matrix = torch.matmul(v_generated, v_reference.T) / (
            torch.norm(v_generated, dim=1, keepdim=True)
            * torch.norm(v_reference, dim=1).unsqueeze(0)
        )
        precision = torch.max(sim_matrix, dim=1)[0].mean().item()
        recall = torch.max(sim_matrix, dim=0)[0].mean().item()
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def _score(gt_wav_np, gen_wav_np):
        gt_wav = torch.from_numpy(gt_wav_np).unsqueeze(0).to(device).float()
        gen_wav = torch.from_numpy(gen_wav_np).unsqueeze(0).to(device).float()
        if sr != 16000:
            gt_wav = obj.resampler(gt_wav)
            gen_wav = obj.resampler(gen_wav)
        v_ref = _process_feats(gt_wav)
        v_gen = _process_feats(gen_wav)
        return _bert_score(v_gen.squeeze(0), v_ref.squeeze(0))

    obj.score = _score
    return obj


# ---------------------------------------------------------------------------
# SpeechTokenDistance scorer factory (internal, not exported)
# ---------------------------------------------------------------------------

def _build_speechtokendist(sr: int, model_type: str, vocab: int, layer: Optional[int],
                            distance_type: str, device: str):
    """Construct and return a SpeechTokenDistance instance."""
    import torch
    import torchaudio
    from transformers import HubertModel

    if model_type == "hubert-base":
        model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        model.eval()
        model.to(device)
    else:
        raise ValueError(f"Not found the setting for {model_type}.")

    km_path = _resolve_km_path(vocab)
    apply_kmeans = _ApplyKmeans(km_path, device=device)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000).to(device)

    class _SpeechTokenDist:
        pass

    obj = _SpeechTokenDist()
    obj.model = model
    obj.device = device
    obj.sr = sr
    obj.layer = layer
    obj.apply_kmeans = apply_kmeans
    obj.distance_type = distance_type
    obj.resampler = resampler

    def _decode_label(audio):
        audio = audio.to(device)
        if sr != 16000:
            audio = obj.resampler(audio)
        if obj.layer is None:
            feats = obj.model(audio).last_hidden_state
        else:
            feats = obj.model(audio, output_hidden_states=True).hidden_states[obj.layer]
        return obj.apply_kmeans(feats[0, ...]).tolist()

    def _score(gt_wav_np, gen_wav_np):
        try:
            from Levenshtein import distance as levenshtein_distance
        except ImportError:
            raise ImportError(
                "SpeechTokenDistance with 'levenshtein' distance requires the "
                "'Levenshtein' package. Install with: pip install tf-restormer[metrics-semantic]"
            )
        try:
            import jellyfish
        except ImportError:
            raise ImportError(
                "SpeechTokenDistance with 'jaro-winkler' distance requires 'jellyfish'. "
                "Install with: pip install tf-restormer[metrics-semantic]"
            )

        gt_wav = torch.from_numpy(gt_wav_np).unsqueeze(0).to(device).float()
        gen_wav = torch.from_numpy(gen_wav_np).unsqueeze(0).to(device).float()
        gt_label = _decode_label(gt_wav)
        gen_label = _decode_label(gen_wav)
        gt_text = _int_array_to_chinese_unicode(gt_label)
        gen_text = _int_array_to_chinese_unicode(gen_label)

        if obj.distance_type == "levenshtein":
            return levenshtein_distance(gen_text, gt_text)
        elif obj.distance_type == "jaro-winkler":
            return jellyfish.jaro_winkler_similarity(gen_text, gt_text)
        else:
            raise ValueError(
                f"Unsupported distance_type: {obj.distance_type!r}. "
                "Choose 'levenshtein' or 'jaro-winkler'."
            )

    obj.score = _score
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_speechbleu(
    estim,
    target,
    fs: int = 16000,
    *,
    model_type: str = "hubert-base",
    vocab: int = 200,
    layer: Optional[int] = 11,
    n_ngram: int = 2,
    device: str = "cuda",
    cache=None,
    **kwargs,
) -> float:
    """Compute SpeechBLEU between estimated and reference waveforms.

    Parameters
    ----------
    estim:      Enhanced waveform (torch.Tensor or np.ndarray, 1-D).
    target:     Reference waveform (torch.Tensor or np.ndarray, 1-D).
    fs:         Sampling rate in Hz (default 16000).
    model_type: HuBERT variant for feature extraction (default "hubert-base").
    vocab:      k-means vocabulary size: 50, 100, or 200 (default 200).
    layer:      Hidden layer index for feature extraction (default 11).
    n_ngram:    N-gram order for BLEU (default 2).
    device:     Torch device string (default ``"cuda"``).
    cache:      _ModelCache instance. Defaults to ``_model_cache``.

    Returns
    -------
    float — BLEU score in [0, 1].
    """
    _ensure_nltk_punkt()

    try:
        from transformers import HubertModel  # noqa: F401 — availability check
    except ImportError:
        raise ImportError(
            "SpeechBLEU requires 'transformers'. "
            "Install with: pip install tf-restormer[metrics-semantic]"
        )
    try:
        import joblib  # noqa: F401 — availability check
    except ImportError:
        raise ImportError(
            "SpeechBLEU requires 'joblib'. "
            "Install with: pip install tf-restormer[metrics-semantic]"
        )

    cache = cache or _model_cache
    scorer = cache.get_or_create(
        f"bleu_{model_type}_v{vocab}_L{layer}",
        device,
        lambda dev: _build_speechbleu(
            sr=fs, model_type=model_type, vocab=vocab,
            layer=layer, n_ngram=n_ngram, device=dev,
        ),
    )
    ref = _to_numpy(target)
    gen = _to_numpy(estim)
    return scorer.score(ref, gen)


def compute_speechbertscore(
    estim,
    target,
    fs: int = 16000,
    *,
    model_type: str = "wavlm-large",
    layer: Optional[int] = 14,
    device: str = "cuda",
    cache=None,
    **kwargs,
) -> tuple:
    """Compute SpeechBERTScore between estimated and reference waveforms.

    Parameters
    ----------
    estim:      Enhanced waveform (torch.Tensor or np.ndarray, 1-D).
    target:     Reference waveform (torch.Tensor or np.ndarray, 1-D).
    fs:         Sampling rate in Hz (default 16000).
    model_type: SSL model variant (default "wavlm-large").
                Options: "hubert-base", "hubert-large", "wav2vec2-base",
                "wav2vec2-large", "wavlm-base", "wavlm-base-plus", "wavlm-large".
    layer:      Hidden layer index for feature extraction (default 14).
    device:     Torch device string (default ``"cuda"``).
    cache:      _ModelCache instance. Defaults to ``_model_cache``.

    Returns
    -------
    tuple[float, float, float] — (precision, recall, f1).
    """
    try:
        from transformers import HubertModel  # noqa: F401 — availability check
    except ImportError:
        raise ImportError(
            "SpeechBERTScore requires 'transformers'. "
            "Install with: pip install tf-restormer[metrics-semantic]"
        )

    cache = cache or _model_cache
    scorer = cache.get_or_create(
        f"bertscore_{model_type}_L{layer}",
        device,
        lambda dev: _build_speechbertscore(
            sr=fs, model_type=model_type, layer=layer, device=dev,
        ),
    )
    ref = _to_numpy(target)
    gen = _to_numpy(estim)
    return scorer.score(ref, gen)


def compute_speechtokendist(
    estim,
    target,
    fs: int = 16000,
    *,
    model_type: str = "hubert-base",
    vocab: int = 200,
    layer: Optional[int] = 6,
    distance_type: str = "jaro-winkler",
    device: str = "cuda",
    cache=None,
    **kwargs,
) -> float:
    """Compute SpeechTokenDistance between estimated and reference waveforms.

    Parameters
    ----------
    estim:         Enhanced waveform (torch.Tensor or np.ndarray, 1-D).
    target:        Reference waveform (torch.Tensor or np.ndarray, 1-D).
    fs:            Sampling rate in Hz (default 16000).
    model_type:    HuBERT variant for feature extraction (default "hubert-base").
    vocab:         k-means vocabulary size: 50, 100, or 200 (default 200).
    layer:         Hidden layer index for feature extraction (default 6).
    distance_type: Distance metric: ``"levenshtein"`` or ``"jaro-winkler"``
                   (default ``"jaro-winkler"``).
    device:        Torch device string (default ``"cuda"``).
    cache:         _ModelCache instance. Defaults to ``_model_cache``.

    Returns
    -------
    float — distance score.
    """
    try:
        from transformers import HubertModel  # noqa: F401 — availability check
    except ImportError:
        raise ImportError(
            "SpeechTokenDistance requires 'transformers'. "
            "Install with: pip install tf-restormer[metrics-semantic]"
        )
    try:
        import joblib  # noqa: F401 — availability check
    except ImportError:
        raise ImportError(
            "SpeechTokenDistance requires 'joblib'. "
            "Install with: pip install tf-restormer[metrics-semantic]"
        )

    cache = cache or _model_cache
    scorer = cache.get_or_create(
        f"tokendist_{model_type}_v{vocab}_L{layer}_{distance_type}",
        device,
        lambda dev: _build_speechtokendist(
            sr=fs, model_type=model_type, vocab=vocab,
            layer=layer, distance_type=distance_type, device=dev,
        ),
    )
    ref = _to_numpy(target)
    gen = _to_numpy(estim)
    return scorer.score(ref, gen)
