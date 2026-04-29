"""A/B comparison: old torchaudio.io.AudioEffector vs new ffmpeg subprocess.

Runs both paths with identical input and fixed random parameters,
then compares outputs with np.allclose.
"""
import random
import numpy as np
import torch
import soundfile as sf
from torchaudio.io import AudioEffector, CodecConfig

# Import new dataset-side implementation
from tf_restormer.models.TF_Restormer.dataset import SynthesisDataset
import types
from tf_restormer.models.TF_Restormer.dataset import (
    _check_ffmpeg, _check_ffmpeg_filters, _check_ffmpeg_encoders,
)

SR = 16000
DURATION = 1.0

def _sine_wave(freq=440, sr=SR, dur=DURATION):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _make_stub(prob_effect):
    real_cls = getattr(SynthesisDataset, '__wrapped__', SynthesisDataset)
    obj = types.SimpleNamespace()
    obj.prob_effect = prob_effect
    obj.has_ffmpeg = _check_ffmpeg()
    if prob_effect and obj.has_ffmpeg:
        obj.available_filters = _check_ffmpeg_filters(('crystalizer', 'flanger', 'acrusher'))
        obj.available_encoders = _check_ffmpeg_encoders(('libmp3lame', 'libvorbis', 'libopus'))
    else:
        obj.available_filters = frozenset()
        obj.available_encoders = frozenset()
    obj._apply_ffmpeg_effect = real_cls._apply_ffmpeg_effect.__get__(obj)
    obj._apply_ffmpeg_codec = real_cls._apply_ffmpeg_codec.__get__(obj)
    obj.audio_effecter = real_cls.audio_effecter.__get__(obj)
    return obj


# ---- OLD path: torchaudio.io.AudioEffector (from engine.py @ f11a0ac) ----

def old_audio_effecter_single_sample(audio_np, prob):
    """Replicate old engine.py audio_effecter for a single sample.

    Input: numpy float32 array (mono, 16kHz)
    Output: numpy float32 array
    """
    audio_tensor = torch.from_numpy(audio_np).unsqueeze(-1)  # (L, 1) as AudioEffector expects

    effects = []
    if random.random() < prob.get('crystalizer', 0.0):
        intensity = random.uniform(1, 4)
        effects.append(AudioEffector(effect=f'crystalizer=i={intensity}'))
    if random.random() < prob.get('flanger', 0.0):
        depth = random.uniform(1, 5)
        effects.append(AudioEffector(effect=f'flanger=depth={depth}'))
    if random.random() < prob.get('crusher', 0.0):
        bits = random.randint(1, 9)
        effects.append(AudioEffector(effect=f'acrusher=bits={bits}'))
    if random.random() < prob.get('codec', 0.0):
        codec_choice = random.choice(['mp3', 'ogg'])
        if codec_choice == 'mp3':
            bit_rate = int(random.randint(4, 16) * 1000)
            effects.append(AudioEffector(format='mp3', codec_config=CodecConfig(bit_rate=bit_rate)))
        else:
            encoder = random.choice(['vorbis', 'opus'])
            effects.append(AudioEffector(format='ogg', encoder=encoder))

    # Apply pipeline
    x = audio_tensor
    for eff in effects:
        x = eff.apply(x, SR)

    x = x[:len(audio_np), :]  # truncate to original length
    return x.squeeze(-1).numpy()


# ---- NEW path: ffmpeg subprocess (from dataset.py) ----

def new_audio_effecter_single_sample(audio_np, prob):
    """Use the new dataset.py audio_effecter."""
    stub = _make_stub(prob)
    return stub.audio_effecter(audio_np.copy(), SR)


# ---- A/B Tests ----

def test_single_effect(effect_name, prob_key, params_desc):
    """Test a single effect type."""
    audio = _sine_wave()
    prob = {prob_key: 1.0}  # force this effect on

    results_match = []
    for seed in range(10):
        random.seed(seed)
        old_out = old_audio_effecter_single_sample(audio.copy(), prob)

        random.seed(seed)
        new_out = new_audio_effecter_single_sample(audio.copy(), prob)

        match = np.allclose(old_out, new_out, atol=1e-4)
        rms_old = np.sqrt(np.mean(old_out**2))
        rms_new = np.sqrt(np.mean(new_out**2))
        rms_diff_db = 20 * np.log10(max(abs(rms_old - rms_new), 1e-10) / max(rms_old, 1e-10))
        max_diff = np.max(np.abs(old_out.astype(np.float64) - new_out.astype(np.float64)))

        results_match.append(match)
        status = "MATCH" if match else f"DIFF (max={max_diff:.6f}, rms_diff={rms_diff_db:.1f}dB)"
        print(f"  seed={seed}: {status}")

    exact_matches = sum(results_match)
    print(f"  => {effect_name}: {exact_matches}/10 exact matches (atol=1e-4)")
    return exact_matches


def test_full_pipeline():
    """Test full pipeline with all effects enabled."""
    audio = _sine_wave()
    prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 1.0}

    results = []
    for seed in range(10):
        random.seed(seed)
        old_out = old_audio_effecter_single_sample(audio.copy(), prob)

        random.seed(seed)
        new_out = new_audio_effecter_single_sample(audio.copy(), prob)

        match = np.allclose(old_out, new_out, atol=1e-4)
        max_diff = np.max(np.abs(old_out.astype(np.float64) - new_out.astype(np.float64)))

        # Energy comparison
        rms_old = np.sqrt(np.mean(old_out**2))
        rms_new = np.sqrt(np.mean(new_out**2))
        if rms_old > 1e-8:
            rms_ratio_db = 20 * np.log10(rms_new / rms_old)
        else:
            rms_ratio_db = 0.0

        results.append({
            'match': match, 'max_diff': max_diff,
            'rms_ratio_db': rms_ratio_db,
            'len_old': len(old_out), 'len_new': len(new_out)
        })

        status = "MATCH" if match else "DIFF"
        print(f"  seed={seed}: {status} max_diff={max_diff:.6f} rms_ratio={rms_ratio_db:+.2f}dB len={len(old_out)}/{len(new_out)}")

    exact = sum(r['match'] for r in results)
    max_diffs = [r['max_diff'] for r in results]
    rms_ratios = [r['rms_ratio_db'] for r in results]
    print(f"  => Full pipeline: {exact}/10 exact, max_diff range=[{min(max_diffs):.6f}, {max(max_diffs):.6f}], rms_ratio range=[{min(rms_ratios):+.2f}, {max(rms_ratios):+.2f}]dB")


if __name__ == "__main__":
    print("=" * 60)
    print("A/B Comparison: torchaudio.io vs ffmpeg subprocess")
    print("=" * 60)

    print("\n--- Crystalizer ---")
    test_single_effect("crystalizer", "crystalizer", "intensity=U(1,4)")

    print("\n--- Flanger ---")
    test_single_effect("flanger", "flanger", "depth=U(1,5)")

    print("\n--- ACrusher ---")
    test_single_effect("crusher", "crusher", "bits=randint(1,9)")

    print("\n--- Codec (MP3 + OGG) ---")
    test_single_effect("codec", "codec", "mp3/ogg random")

    print("\n--- Full Pipeline (all effects) ---")
    test_full_pipeline()

    print("\n" + "=" * 60)
    print("DONE")
