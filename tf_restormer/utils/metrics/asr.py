"""
tf_restormer.utils.metrics.asr
================================
ASR-based error-rate metrics: WER (Whisper), WER (Wav2Vec2 CTC), CER (Whisper).

Merges WhisperASR (ASR_whisper.py) and SimpleCTCASR (ASR_w2v.py) into a single
module with top-level ``compute_*`` functions.

All functions return ``tuple[int, int]`` → ``(err, ref_len)`` for accumulation:
  - ``err``     : edit distance between reference and hypothesis token sequences
  - ``ref_len`` : number of tokens in the reference sequence (denominator)
  Callers accumulate (err, ref_len) across utterances, then compute rate =
  sum(err) / sum(ref_len).  When ref_len == 0, (0, 0) is returned so that
  external accumulators do not produce NaN.

Lazy imports: ``transformers`` is imported inside each ``compute_*`` call so
that this module can be imported without transformers being installed.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

from tf_restormer.utils.metrics import _model_cache


# ---------------------------------------------------------------------------
# Private: edit distance
# ---------------------------------------------------------------------------

def _edit_distance(ref_units: List[str], hyp_units: List[str]) -> int:
    """Standard dynamic-programming edit distance (Levenshtein)."""
    R, H = len(ref_units), len(hyp_units)
    dp = [[0] * (H + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        dp[i][0] = i
    for j in range(H + 1):
        dp[0][j] = j
    for i in range(1, R + 1):
        ri = ref_units[i - 1]
        for j in range(1, H + 1):
            hj = hyp_units[j - 1]
            cost = 0 if ri == hj else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[R][H]


# ---------------------------------------------------------------------------
# Private: text normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_wer_whisper(s: str) -> str:
    """WER normalisation for Whisper output (multilingual: Korean + English)."""
    s = s.lower()
    # Keep Korean (ㄱ-ㅎ, ㅏ-ㅣ, 가-힣), ASCII letters, digits, apostrophe, spaces.
    s = re.sub(r"[^a-z0-9'\sㄱ-ㅎㅏ-ㅣ가-힣]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_wer_w2v(s: str) -> str:
    """WER normalisation for Wav2Vec2 output (English CTC)."""
    s = s.lower()
    # Keep ASCII letters, digits, apostrophe, spaces.
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_cer(s: str, keep_space: bool = False) -> str:
    """CER normalisation: NFKC → lowercase → strip punctuation."""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    # Keep Korean, ASCII letters, digits, whitespace.
    s = re.sub(r"[^a-z0-9\sㄱ-ㅎㅏ-ㅣ가-힣]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not keep_space:
        s = s.replace(" ", "")
    return s


# ---------------------------------------------------------------------------
# Private: TIMIT-39 phoneme mapping / post-processing
# TODO: PhER (Phoneme Error Rate) metric is not yet registered in the registry.
#       The helpers below are reserved for future PhER metric addition.
# ---------------------------------------------------------------------------

_TIMIT39_MAP: Dict[str, str] = {
    "ax": "ah", "ix": "ih", "ax-h": "ah", "axr": "er",
    "em": "m",  "en": "n",  "eng": "ng", "el": "l",
    "ux": "uw",
    "dx": "dx",
}

_SILENCE_TOKENS = {"sil", "sp", "spn", "nsn", "pau"}


def _normalize_pher(s: str) -> str:
    """PhER normalisation for phoneme sequences."""
    s = s.lower()
    # Remove stress digits (aa0/eh1/er2 → aa/eh/er)
    s = re.sub(r"\b([a-z]{2,})([0-2])\b", r"\1", s)
    # Remove non-alpha (keep word boundaries)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _postprocess_phonemes(
    toks: List[str],
    phoneme_set: str = "arpabet",
    collapse_repeats: bool = False,
) -> List[str]:
    """Remove silence tokens, optionally collapse repeats, apply phoneme mapping."""
    out: List[str] = []
    prev: Optional[str] = None
    for t in toks:
        if t in _SILENCE_TOKENS or not t:
            continue
        if phoneme_set == "timit-39":
            t = _TIMIT39_MAP.get(t, t)
        if not t:
            continue
        if collapse_repeats and prev is not None and t == prev:
            continue
        out.append(t)
        prev = t
    return out


# ---------------------------------------------------------------------------
# Private: Whisper model factory and inference
# ---------------------------------------------------------------------------

def _build_whisper(device: str):
    """Load WhisperForConditionalGeneration + processor onto device."""
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        raise ImportError(
            "ASR metrics require 'transformers'. "
            "Install with: pip install tf-restormer[metrics-neural]"
        )
    model_id = "openai/whisper-large-v3"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


def _whisper_transcribe(wav_tensor, processor, model, device: str) -> str:
    """Run Whisper inference on a 1-D float32 tensor. Returns transcription str."""
    import torch as th

    if wav_tensor.ndim == 2:
        wav_tensor = wav_tensor.mean(dim=0)
    assert wav_tensor.ndim == 1, "input must be 1D or 2D (C, T) tensor"

    # dtype normalisation
    if wav_tensor.dtype == th.int16:
        wav_tensor = wav_tensor.to(th.float32) / 32768.0
    elif wav_tensor.dtype == th.int32:
        wav_tensor = wav_tensor.to(th.float32) / 2147483648.0
    else:
        wav_tensor = wav_tensor.to(th.float32)
    wav_tensor = wav_tensor.clamp_(-1.0, 1.0)

    input_features = processor(
        wav_tensor.detach().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(device)

    with th.inference_mode():
        predicted_ids = model.generate(
            input_features,
            task="transcribe",
            language="korean",
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Private: Wav2Vec2 CTC model factory and inference
# ---------------------------------------------------------------------------

def _build_w2v_wer(device: str):
    """Load Wav2Vec2ForCTC (WER) + processor onto device."""
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError:
        raise ImportError(
            "ASR metrics require 'transformers'. "
            "Install with: pip install tf-restormer[metrics-neural]"
        )
    model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to(device).eval()
    return processor, model


def _w2v_transcribe(wav_tensor, processor, model, device: str) -> str:
    """Run Wav2Vec2 CTC inference. Returns transcription str."""
    import torch as th

    if wav_tensor.ndim == 2:
        wav_tensor = wav_tensor.mean(dim=0)
    assert wav_tensor.ndim == 1, "input must be 1D or 2D (C, T) tensor"

    # dtype normalisation
    if wav_tensor.dtype == th.int16:
        wav_tensor = wav_tensor.to(th.float32) / 32768.0
    elif wav_tensor.dtype == th.int32:
        wav_tensor = wav_tensor.to(th.float32) / 2147483648.0
    else:
        wav_tensor = wav_tensor.to(th.float32)
    wav_tensor = wav_tensor.clamp_(-1.0, 1.0)

    enc = processor(
        wav_tensor.detach().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with th.inference_mode():
        logits = model(
            enc["input_values"],
            attention_mask=enc.get("attention_mask"),
        ).logits
    ids = logits.argmax(dim=-1)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Public: compute_wer_whisper
# ---------------------------------------------------------------------------

def compute_wer_whisper(
    estim,
    target,
    fs: int = 16000,
    *,
    device: str = "cuda",
    cache=None,
    key: str = "Output",
    remove_punct: Optional[bool] = None,
    **kwargs,
) -> tuple:
    """Compute WER using Whisper large-v3.

    Parameters
    ----------
    estim:       Hypothesis (enhanced) waveform — torch.Tensor (T,) or (C, T).
                 Dispatcher 1st positional arg.
    target:      Reference (clean) waveform — same shape convention.
                 Dispatcher 2nd positional arg.
    fs:          Sampling rate (default 16000). Whisper expects 16 kHz.
    device:      Torch device string (default ``"cuda"``).
    cache:       _ModelCache instance. Defaults to ``_model_cache``.
    key:         Label printed next to the hypothesis transcript. Useful for
                 debug logging. Default ``"Output"``.
    remove_punct: Whether to strip punctuation before WER computation.
                  ``None`` (default) applies the WER convention (True).

    Returns
    -------
    tuple[int, int] — ``(err, ref_len)``.
    ``(0, 0)`` when reference is empty (avoids divide-by-zero at caller).
    """
    import torch as th

    cache = cache or _model_cache

    processor, model = cache.get_or_create(
        "whisper_large_v3", device, _build_whisper
    )

    # estim = hypothesis (enhanced), target = reference (clean)
    hyp_wav = estim
    ref_wav = target

    if not isinstance(ref_wav, th.Tensor):
        ref_wav = th.from_numpy(ref_wav)
    if not isinstance(hyp_wav, th.Tensor):
        hyp_wav = th.from_numpy(hyp_wav)

    ref_text = _whisper_transcribe(ref_wav, processor, model, device)
    hyp_text = _whisper_transcribe(hyp_wav, processor, model, device)

    if key != "Output":
        print(f"clean: \n{ref_text}")
    print(f"{key}: \n{hyp_text}")

    # normalise
    if remove_punct is None:
        remove_punct = True
    ref_n = _normalize_wer_whisper(ref_text) if remove_punct else re.sub(r"\s+", " ", ref_text).strip()
    hyp_n = _normalize_wer_whisper(hyp_text) if remove_punct else re.sub(r"\s+", " ", hyp_text).strip()

    ref_tok = ref_n.split()
    hyp_tok = hyp_n.split()

    if len(ref_tok) == 0:
        return (0, 0)

    err = _edit_distance(ref_tok, hyp_tok)
    return (err, len(ref_tok))


# ---------------------------------------------------------------------------
# Public: compute_wer_w2v
# ---------------------------------------------------------------------------

def compute_wer_w2v(
    estim,
    target,
    fs: int = 16000,
    *,
    device: str = "cuda",
    cache=None,
    key: str = "Output",
    remove_punct: Optional[bool] = None,
    **kwargs,
) -> tuple:
    """Compute WER using Wav2Vec2 CTC (jonatasgrosman/wav2vec2-large-xlsr-53-english).

    Parameters
    ----------
    estim:        Hypothesis (enhanced) waveform — torch.Tensor (T,) or (C, T).
                  Dispatcher 1st positional arg.
    target:       Reference (clean) waveform — same shape convention.
                  Dispatcher 2nd positional arg.
    fs:           Sampling rate (default 16000). Model expects 16 kHz.
    device:       Torch device string (default ``"cuda"``).
    cache:        _ModelCache instance. Defaults to ``_model_cache``.
    key:          Label printed next to hypothesis transcript (debug logging).
    remove_punct: Whether to strip punctuation. ``None`` applies WER default (True).

    Returns
    -------
    tuple[int, int] — ``(err, ref_len)``.
    ``(0, 0)`` when reference is empty.
    """
    import torch as th

    cache = cache or _model_cache

    processor, model = cache.get_or_create(
        "w2v_wer_xlsr53en", device, _build_w2v_wer
    )

    # estim = hypothesis (enhanced), target = reference (clean)
    hyp_wav = estim
    ref_wav = target

    if not isinstance(ref_wav, th.Tensor):
        ref_wav = th.from_numpy(ref_wav)
    if not isinstance(hyp_wav, th.Tensor):
        hyp_wav = th.from_numpy(hyp_wav)

    ref_text = _w2v_transcribe(ref_wav, processor, model, device)
    hyp_text = _w2v_transcribe(hyp_wav, processor, model, device)

    if key != "Output":
        print(f"clean: \n{ref_text}")
    print(f"{key}: \n{hyp_text}")

    # normalise
    if remove_punct is None:
        remove_punct = True
    ref_n = _normalize_wer_w2v(ref_text) if remove_punct else ref_text.lower().strip()
    hyp_n = _normalize_wer_w2v(hyp_text) if remove_punct else hyp_text.lower().strip()

    ref_tok = ref_n.split()
    hyp_tok = hyp_n.split()

    if len(ref_tok) == 0:
        return (0, 0)

    err = _edit_distance(ref_tok, hyp_tok)
    return (err, len(ref_tok))


# ---------------------------------------------------------------------------
# Public: compute_cer_whisper
# ---------------------------------------------------------------------------

def compute_cer_whisper(
    estim,
    target,
    fs: int = 16000,
    *,
    device: str = "cuda",
    cache=None,
    key: str = "Output",
    keep_space: bool = False,
    **kwargs,
) -> tuple:
    """Compute CER using Whisper large-v3.

    Character Error Rate is computed by treating each character as a token.
    Korean text (including mixed Korean+English) is handled via NFKC normalisation.

    Parameters
    ----------
    estim:      Hypothesis (enhanced) waveform — torch.Tensor (T,) or (C, T).
                Dispatcher 1st positional arg.
    target:     Reference (clean) waveform — same shape convention.
                Dispatcher 2nd positional arg.
    fs:         Sampling rate (default 16000). Whisper expects 16 kHz.
    device:     Torch device string (default ``"cuda"``).
    cache:      _ModelCache instance. Defaults to ``_model_cache``.
    key:        Label printed next to hypothesis transcript (debug logging).
    keep_space: Whether to retain whitespace characters for CER (default False).

    Returns
    -------
    tuple[int, int] — ``(err, ref_len)``.
    ``(0, 0)`` when reference is empty.
    """
    import torch as th

    cache = cache or _model_cache

    processor, model = cache.get_or_create(
        "whisper_large_v3", device, _build_whisper
    )

    # estim = hypothesis (enhanced), target = reference (clean)
    hyp_wav = estim
    ref_wav = target

    if not isinstance(ref_wav, th.Tensor):
        ref_wav = th.from_numpy(ref_wav)
    if not isinstance(hyp_wav, th.Tensor):
        hyp_wav = th.from_numpy(hyp_wav)

    ref_text = _whisper_transcribe(ref_wav, processor, model, device)
    hyp_text = _whisper_transcribe(hyp_wav, processor, model, device)

    if key != "Output":
        print(f"clean: \n{ref_text}")
    print(f"{key}: \n{hyp_text}")

    ref_n = _normalize_cer(ref_text, keep_space=keep_space)
    hyp_n = _normalize_cer(hyp_text, keep_space=keep_space)

    # Character-level tokenisation
    ref_tok = list(ref_n)
    hyp_tok = list(hyp_n)

    if len(ref_tok) == 0:
        return (0, 0)

    err = _edit_distance(ref_tok, hyp_tok)
    return (err, len(ref_tok))
