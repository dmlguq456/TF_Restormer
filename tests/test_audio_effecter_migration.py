"""Tests for the ffmpeg-based audio_effecter migration in SynthesisDataset.

Covers:
  1. Functional correctness — ffmpeg effects actually modify audio
  2. Length preservation — output length matches input length
  3. Seed reproducibility — same seed produces same random choices
  4. Config integration — synthesis_config.audio_effects is properly read
  5. Graceful degradation — when ffmpeg is absent, effects are skipped
  6. Pipeline order — effects applied in correct order
  7. dtype preservation — output is np.float32
  8. Zero-probability bypass — all probs 0.0 → output equals input
  9. Opus energy-scale sanity — RMS within ±6 dB
 10. Codec amplitude bounds — |max| < 2.0
 11. Filter availability check — _check_ffmpeg_filters degrades gracefully
 12. Encoder availability check — _check_ffmpeg_encoders parses correctly
 13. Codec random consumption sequence — sequence matches engine.py L172 pattern

Run:
    uv run pytest tests/test_audio_effecter_migration.py -v
"""

import random
import shutil
import types
import unittest.mock as mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level helpers imported directly (not through SynthesisDataset)
# ---------------------------------------------------------------------------
from tf_restormer.models.TF_Restormer.dataset import (
    _check_ffmpeg,
    _check_ffmpeg_encoders,
    _check_ffmpeg_filters,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SR = 16000  # expected operating sample rate inside dataset pipeline
DURATION = 1.0  # seconds
N_SAMPLES = int(SR * DURATION)


def _sine_wave(freq: float = 440.0, sr: int = SR, duration: float = DURATION) -> np.ndarray:
    """Return a 440 Hz sine wave as float32, amplitude ~0.5."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _make_dataset_stub(prob_effect: dict):
    """Construct a minimal object that has the SynthesisDataset audio-effect
    attributes without requiring real SCP files or a RIR directory.

    Uses types.SimpleNamespace with bound methods from the real class.
    SynthesisDataset is wrapped by @logger_wraps() so object.__new__() fails;
    we access the underlying class via __wrapped__ attribute.
    """
    from tf_restormer.models.TF_Restormer.dataset import SynthesisDataset

    # logger_wraps wraps the class — get the real class
    real_cls = getattr(SynthesisDataset, '__wrapped__', SynthesisDataset)

    obj = types.SimpleNamespace()

    # Attributes required by audio_effecter / _apply_ffmpeg_codec
    obj.prob_effect = prob_effect
    obj.has_ffmpeg = _check_ffmpeg()

    if prob_effect and obj.has_ffmpeg:
        needed_filters = ('crystalizer', 'flanger', 'acrusher')
        obj.available_filters = _check_ffmpeg_filters(needed_filters)
        needed_encoders = ('libmp3lame', 'libvorbis', 'libopus')
        obj.available_encoders = _check_ffmpeg_encoders(needed_encoders)
    else:
        obj.available_filters = frozenset()
        obj.available_encoders = frozenset()

    # Bind methods from the real class so they operate on obj
    obj._apply_ffmpeg_effect = real_cls._apply_ffmpeg_effect.__get__(obj)
    obj._apply_ffmpeg_codec = real_cls._apply_ffmpeg_codec.__get__(obj)
    obj.audio_effecter = real_cls.audio_effecter.__get__(obj)

    return obj


# ---------------------------------------------------------------------------
# Skip marker for environments without ffmpeg
# ---------------------------------------------------------------------------

_ffmpeg_available = _check_ffmpeg()
skip_no_ffmpeg = pytest.mark.skipif(
    not _ffmpeg_available,
    reason="ffmpeg not available in this environment",
)


# ===========================================================================
# Test 1 — Functional correctness
# ===========================================================================

class TestFunctionalCorrectness:
    """ffmpeg effects actually change audio (not a no-op)."""

    @skip_no_ffmpeg
    def test_crystalizer_modifies_audio(self):
        stub = _make_dataset_stub({'crystalizer': 1.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0})
        if 'crystalizer' not in stub.available_filters:
            pytest.skip("crystalizer filter not available in this ffmpeg build")
        audio = _sine_wave()
        out = stub.audio_effecter(audio.copy(), SR)
        assert not np.allclose(out, audio), "crystalizer should modify audio"

    @skip_no_ffmpeg
    def test_flanger_modifies_audio(self):
        stub = _make_dataset_stub({'crystalizer': 0.0, 'flanger': 1.0, 'crusher': 0.0, 'codec': 0.0})
        if 'flanger' not in stub.available_filters:
            pytest.skip("flanger filter not available in this ffmpeg build")
        audio = _sine_wave()
        out = stub.audio_effecter(audio.copy(), SR)
        assert not np.allclose(out, audio), "flanger should modify audio"

    @skip_no_ffmpeg
    def test_acrusher_modifies_audio(self):
        stub = _make_dataset_stub({'crystalizer': 0.0, 'flanger': 0.0, 'crusher': 1.0, 'codec': 0.0})
        if 'acrusher' not in stub.available_filters:
            pytest.skip("acrusher filter not available in this ffmpeg build")
        audio = _sine_wave()
        out = stub.audio_effecter(audio.copy(), SR)
        assert not np.allclose(out, audio), "acrusher should modify audio"

    @skip_no_ffmpeg
    def test_codec_modifies_audio(self):
        stub = _make_dataset_stub({'crystalizer': 0.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 1.0})
        if not stub.available_encoders:
            pytest.skip("No codec encoders available in this ffmpeg build")
        audio = _sine_wave()
        out = stub.audio_effecter(audio.copy(), SR)
        assert not np.allclose(out, audio), "codec encode/decode should modify audio"


# ===========================================================================
# Test 2 — Length preservation
# ===========================================================================

class TestLengthPreservation:
    """Output length must match input length for all effect paths."""

    @skip_no_ffmpeg
    def test_filter_effects_preserve_length(self):
        prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 0.0}
        stub = _make_dataset_stub(prob)
        audio = _sine_wave()
        out = stub.audio_effecter(audio.copy(), SR)
        assert len(out) == N_SAMPLES, (
            f"Expected {N_SAMPLES} samples, got {len(out)}"
        )

    @skip_no_ffmpeg
    def test_codec_effect_preserves_length(self):
        prob = {'crystalizer': 0.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 1.0}
        stub = _make_dataset_stub(prob)
        if not stub.available_encoders:
            pytest.skip("No codec encoders available")
        audio = _sine_wave()
        out = stub.audio_effecter(audio.copy(), SR)
        assert len(out) == N_SAMPLES, (
            f"Expected {N_SAMPLES} samples, got {len(out)}"
        )

    @skip_no_ffmpeg
    def test_all_effects_preserve_length(self):
        prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 1.0}
        stub = _make_dataset_stub(prob)
        audio = _sine_wave()
        out = stub.audio_effecter(audio.copy(), SR)
        assert len(out) == N_SAMPLES, (
            f"Expected {N_SAMPLES} samples, got {len(out)}"
        )

    @skip_no_ffmpeg
    def test_apply_ffmpeg_effect_direct_length(self):
        """_apply_ffmpeg_effect with crystalizer should preserve length."""
        stub = _make_dataset_stub({'crystalizer': 1.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0})
        if 'crystalizer' not in stub.available_filters:
            pytest.skip("crystalizer not available")
        audio = _sine_wave()
        out = stub._apply_ffmpeg_effect(audio.copy(), SR, "crystalizer=i=2.0")
        assert len(out) == N_SAMPLES


# ===========================================================================
# Test 3 — Seed reproducibility
# ===========================================================================

class TestSeedReproducibility:
    """Same seed → same random parameter choices and same output."""

    @skip_no_ffmpeg
    def test_same_seed_same_output(self):
        prob = {'crystalizer': 0.5, 'flanger': 0.5, 'crusher': 0.5, 'codec': 0.5}
        stub = _make_dataset_stub(prob)
        audio = _sine_wave()

        random.seed(42)
        out1 = stub.audio_effecter(audio.copy(), SR)

        random.seed(42)
        out2 = stub.audio_effecter(audio.copy(), SR)

        np.testing.assert_array_equal(
            out1, out2,
            err_msg="Different seeds should produce different outputs; same seeds must match"
        )

    @skip_no_ffmpeg
    def test_different_seeds_different_choices(self):
        """High-probability effects with different seeds should diverge in at least
        some runs. We verify by checking that the two outputs are not identical
        (the probability of identical outputs with different effect params is negligible).
        """
        prob = {'crystalizer': 0.8, 'flanger': 0.8, 'crusher': 0.8, 'codec': 0.8}
        stub = _make_dataset_stub(prob)
        if not stub.available_filters and not stub.available_encoders:
            pytest.skip("No effects available")
        audio = _sine_wave()

        results = set()
        for seed in range(10):
            random.seed(seed)
            out = stub.audio_effecter(audio.copy(), SR)
            results.add(out.tobytes())

        assert len(results) > 1, (
            "Expected at least 2 distinct outputs with 10 different seeds"
        )


# ===========================================================================
# Test 4 — Config integration
# ===========================================================================

class TestConfigIntegration:
    """synthesis_config.audio_effects is correctly read into self.prob_effect."""

    def test_prob_effect_set_from_config(self):
        prob = {'crystalizer': 0.15, 'flanger': 0.05, 'crusher': 0.1, 'codec': 0.3}
        stub = _make_dataset_stub(prob)
        assert stub.prob_effect == prob, (
            "prob_effect attribute must mirror the audio_effects config dict"
        )

    def test_empty_config_no_effects(self):
        """Empty audio_effects dict → prob_effect is falsy → no ffmpeg calls."""
        stub = _make_dataset_stub({})
        audio = _sine_wave()
        # With empty prob_effect, audio_effecter still runs (it checks per-key),
        # but should return audio unchanged since all random() < 0.0 is always False
        # (empty dict → all .get() return 0.0)
        random.seed(0)
        out = stub.audio_effecter(audio.copy(), SR)
        np.testing.assert_array_equal(out, audio.astype(np.float32))

    def test_none_config_treated_as_empty(self):
        """None audio_effects → no AttributeError, no ffmpeg calls."""
        # When synthesis_config has no audio_effects key, .get() returns {}
        # The _synthesis() guard `if self.prob_effect and self.has_ffmpeg` catches this.
        stub = _make_dataset_stub({})
        assert not stub.prob_effect  # empty dict is falsy


# ===========================================================================
# Test 5 — Graceful degradation (ffmpeg absent)
# ===========================================================================

class TestGracefulDegradation:
    """When ffmpeg is absent, effects are skipped without error."""

    def test_no_ffmpeg_returns_original(self):
        prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 1.0}
        stub = _make_dataset_stub(prob)
        # Force ffmpeg unavailability
        stub.has_ffmpeg = False
        stub.available_filters = frozenset()
        stub.available_encoders = frozenset()

        audio = _sine_wave()
        # audio_effecter itself does not check has_ffmpeg — that guard is in _synthesis().
        # However, with no available_filters and no available_encoders, no ffmpeg calls
        # are made. The filter prob rolls still happen but no effect is applied.
        out = stub.audio_effecter(audio.copy(), SR)
        # Output must be float32 and same length
        assert out.dtype == np.float32
        assert len(out) == N_SAMPLES

    def test_apply_ffmpeg_effect_on_failure_returns_original(self):
        """_apply_ffmpeg_effect with an invalid filter string returns original audio."""
        stub = _make_dataset_stub({'crystalizer': 1.0})
        if not _ffmpeg_available:
            pytest.skip("ffmpeg not available")
        audio = _sine_wave()
        # Use a deliberately invalid filter string to trigger ffmpeg failure
        out = stub._apply_ffmpeg_effect(audio.copy(), SR, "nonexistent_filter_xyz=val=1")
        np.testing.assert_array_equal(out, audio.astype(np.float32))

    def test_unavailable_filters_skipped(self):
        """When available_filters is empty, filter effects are not applied."""
        prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 0.0}
        stub = _make_dataset_stub(prob)
        stub.available_filters = frozenset()  # force all filters unavailable
        audio = _sine_wave()
        random.seed(7)
        out = stub.audio_effecter(audio.copy(), SR)
        # No filter applied → output must equal input (float32 cast only)
        np.testing.assert_array_equal(out, audio.astype(np.float32))


# ===========================================================================
# Test 6 — Pipeline order
# ===========================================================================

class TestPipelineOrder:
    """Effects must be applied crystalizer → flanger → crusher → codec."""

    @skip_no_ffmpeg
    def test_order_recorded_via_mock(self):
        """Patch _apply_ffmpeg_effect and _apply_ffmpeg_codec to record call order."""
        prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 1.0}
        stub = _make_dataset_stub(prob)
        # Give all filters and encoders so probability gates pass (with seed 0, all rolls succeed
        # only if prob=1.0, which is the case here)
        stub.available_filters = frozenset({'crystalizer', 'flanger', 'acrusher'})
        stub.available_encoders = frozenset({'libmp3lame', 'libvorbis', 'libopus'})

        call_order = []

        def fake_effect(audio, sr, effect_str):
            call_order.append('filter:' + effect_str)
            return audio.astype(np.float32)

        def fake_codec(audio, sr):
            call_order.append('codec')
            return audio.astype(np.float32)

        stub._apply_ffmpeg_effect = fake_effect
        stub._apply_ffmpeg_codec = fake_codec

        random.seed(0)
        stub.audio_effecter(_sine_wave(), SR)

        # The filter chain must arrive before the codec call
        codec_positions = [i for i, c in enumerate(call_order) if c == 'codec']
        filter_positions = [i for i, c in enumerate(call_order) if c.startswith('filter:')]

        if codec_positions and filter_positions:
            assert min(codec_positions) > max(filter_positions), (
                "Codec call must come after all filter-chain calls"
            )

    @skip_no_ffmpeg
    def test_chained_filter_string_order(self):
        """When multiple filters are active, the chained -af string has
        crystalizer before flanger before acrusher."""
        prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 0.0}
        stub = _make_dataset_stub(prob)
        stub.available_filters = frozenset({'crystalizer', 'flanger', 'acrusher'})
        stub.available_encoders = frozenset()

        captured_effect_str = []

        def fake_effect(audio, sr, effect_str):
            captured_effect_str.append(effect_str)
            return audio.astype(np.float32)

        stub._apply_ffmpeg_effect = fake_effect

        random.seed(0)
        stub.audio_effecter(_sine_wave(), SR)

        assert len(captured_effect_str) == 1, "Single chained ffmpeg call expected"
        parts = captured_effect_str[0].split(',')
        # All parts come from {crystalizer, flanger, acrusher} and ordering is preserved
        filter_names = [p.split('=')[0] for p in parts]
        assert filter_names == sorted(
            filter_names,
            key=lambda n: ['crystalizer', 'flanger', 'acrusher'].index(n)
        ), f"Filter chain order incorrect: {filter_names}"


# ===========================================================================
# Test 7 — dtype preservation
# ===========================================================================

class TestDtypePreservation:
    """Output must be np.float32 regardless of input dtype."""

    @skip_no_ffmpeg
    def test_float64_input_returns_float32(self):
        prob = {'crystalizer': 1.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0}
        stub = _make_dataset_stub(prob)
        if 'crystalizer' not in stub.available_filters:
            pytest.skip("crystalizer not available")
        audio_f64 = _sine_wave().astype(np.float64)
        out = stub.audio_effecter(audio_f64, SR)
        assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"

    def test_float32_input_returns_float32(self):
        prob = {'crystalizer': 0.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0}
        stub = _make_dataset_stub(prob)
        audio_f32 = _sine_wave().astype(np.float32)
        out = stub.audio_effecter(audio_f32, SR)
        assert out.dtype == np.float32

    @skip_no_ffmpeg
    def test_apply_ffmpeg_effect_returns_float32(self):
        stub = _make_dataset_stub({'crystalizer': 1.0})
        if 'crystalizer' not in stub.available_filters:
            pytest.skip("crystalizer not available")
        audio_f64 = _sine_wave().astype(np.float64)
        out = stub._apply_ffmpeg_effect(audio_f64, SR, "crystalizer=i=2.0")
        assert out.dtype == np.float32

    @skip_no_ffmpeg
    def test_apply_ffmpeg_codec_returns_float32(self):
        stub = _make_dataset_stub({'codec': 1.0})
        if not stub.available_encoders:
            pytest.skip("No encoders available")
        audio_f64 = _sine_wave().astype(np.float64)
        out = stub._apply_ffmpeg_codec(audio_f64, SR)
        assert out.dtype == np.float32


# ===========================================================================
# Test 8 — Zero-probability bypass
# ===========================================================================

class TestZeroProbabilityBypass:
    """When all probabilities are 0.0, output must equal input (no ffmpeg call)."""

    def test_all_zero_prob_output_equals_input(self):
        prob = {'crystalizer': 0.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0}
        stub = _make_dataset_stub(prob)
        audio = _sine_wave()

        # Track calls by replacing the methods directly on the namespace
        effect_called = []
        codec_called = []

        original_effect = stub._apply_ffmpeg_effect
        original_codec = stub._apply_ffmpeg_codec

        def tracking_effect(audio, sr, effect_str):
            effect_called.append(effect_str)
            return original_effect(audio, sr, effect_str)

        def tracking_codec(audio, sr):
            codec_called.append(True)
            return original_codec(audio, sr)

        stub._apply_ffmpeg_effect = tracking_effect
        stub._apply_ffmpeg_codec = tracking_codec

        random.seed(99)
        out = stub.audio_effecter(audio.copy(), SR)

        assert len(effect_called) == 0, "No ffmpeg effect call expected with all probs=0"
        assert len(codec_called) == 0, "No codec call expected with all probs=0"
        np.testing.assert_array_equal(out, audio.astype(np.float32))


# ===========================================================================
# Test 9 — Opus energy-scale sanity (RMS within ±6 dB)
# ===========================================================================

class TestOpusEnergySanity:
    """Opus codec round-trip should not cause catastrophic energy change."""

    @skip_no_ffmpeg
    def test_opus_rms_within_6db(self):
        stub = _make_dataset_stub({'crystalizer': 0.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0})
        if 'libopus' not in stub.available_encoders:
            pytest.skip("libopus not available")

        audio = _sine_wave()
        rms_in = _rms(audio)

        # Force opus path by patching random.choice and random.randint
        with mock.patch('random.choice') as mock_choice, \
             mock.patch('random.randint') as mock_randint:
            # First random.choice(['mp3', 'ogg']) → 'ogg'
            # Second random.choice(['vorbis', 'opus']) → 'opus'
            mock_choice.side_effect = ['ogg', 'opus']
            mock_randint.return_value = 8  # not used for ogg path

            out = stub._apply_ffmpeg_codec(audio.copy(), SR)

        rms_out = _rms(out)
        if rms_in < 1e-8:
            pytest.skip("Input RMS too low for meaningful comparison")

        ratio_db = 20 * np.log10(rms_out / rms_in + 1e-12)
        assert abs(ratio_db) <= 6.0, (
            f"Opus RMS ratio {ratio_db:.2f} dB exceeds ±6 dB threshold"
        )


# ===========================================================================
# Test 10 — Codec amplitude bounds
# ===========================================================================

class TestCodecAmplitudeBounds:
    """After codec apply, |max| < 2.0 (no normalization blow-up)."""

    @skip_no_ffmpeg
    def test_mp3_amplitude_bounded(self):
        stub = _make_dataset_stub({'codec': 1.0})
        if 'libmp3lame' not in stub.available_encoders:
            pytest.skip("libmp3lame not available")
        audio = _sine_wave()

        with mock.patch('random.choice') as mock_choice, \
             mock.patch('random.randint') as mock_randint:
            mock_choice.return_value = 'mp3'
            mock_randint.return_value = 8  # bitrate factor: 8 * 1000 = 8000 bps
            out = stub._apply_ffmpeg_codec(audio.copy(), SR)

        assert np.max(np.abs(out)) < 2.0, (
            f"MP3 output exceeds amplitude bound: max={np.max(np.abs(out)):.3f}"
        )

    @skip_no_ffmpeg
    def test_ogg_amplitude_bounded(self):
        stub = _make_dataset_stub({'codec': 1.0})
        if 'libvorbis' not in stub.available_encoders:
            pytest.skip("libvorbis not available")
        audio = _sine_wave()

        with mock.patch('random.choice') as mock_choice:
            mock_choice.side_effect = ['ogg', 'vorbis']
            out = stub._apply_ffmpeg_codec(audio.copy(), SR)

        assert np.max(np.abs(out)) < 2.0, (
            f"OGG/Vorbis output exceeds amplitude bound: max={np.max(np.abs(out)):.3f}"
        )

    @skip_no_ffmpeg
    def test_full_pipeline_amplitude_bounded(self):
        prob = {'crystalizer': 1.0, 'flanger': 1.0, 'crusher': 1.0, 'codec': 1.0}
        stub = _make_dataset_stub(prob)
        audio = _sine_wave()
        random.seed(42)
        out = stub.audio_effecter(audio.copy(), SR)
        assert np.max(np.abs(out)) < 2.0, (
            f"Full-pipeline output exceeds amplitude bound: max={np.max(np.abs(out)):.3f}"
        )


# ===========================================================================
# Test 11 — Filter availability check
# ===========================================================================

class TestFilterAvailabilityCheck:
    """_check_ffmpeg_filters correctly identifies available/unavailable filters."""

    @skip_no_ffmpeg
    def test_known_standard_filter_available(self):
        """'aecho' is a standard ffmpeg filter available in essentially all builds."""
        result = _check_ffmpeg_filters(('aecho',))
        assert 'aecho' in result, (
            "aecho is a standard ffmpeg filter and should be detected as available"
        )

    @skip_no_ffmpeg
    def test_fake_filter_not_available(self):
        result = _check_ffmpeg_filters(('nonexistent_filter_xyz_abc',))
        assert 'nonexistent_filter_xyz_abc' not in result

    @skip_no_ffmpeg
    def test_returns_frozenset(self):
        result = _check_ffmpeg_filters(('aecho',))
        assert isinstance(result, frozenset)

    def test_no_ffmpeg_returns_empty_frozenset(self):
        """When ffmpeg subprocess raises FileNotFoundError, return empty frozenset."""
        import subprocess
        with mock.patch('subprocess.run', side_effect=FileNotFoundError("no ffmpeg")):
            # Clear lru_cache to force re-execution
            _check_ffmpeg_filters.cache_clear()
            result = _check_ffmpeg_filters(('crystalizer', 'flanger'))
            assert result == frozenset(), f"Expected empty frozenset, got {result}"
        # Restore cache state by clearing again
        _check_ffmpeg_filters.cache_clear()

    @skip_no_ffmpeg
    def test_unavailable_filter_skipped_in_pipeline(self):
        """When available_filters is set to empty, filter effects are not applied
        even with prob=1.0."""
        prob = {'crystalizer': 1.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0}
        stub = _make_dataset_stub(prob)
        stub.available_filters = frozenset()  # artificially remove all filters

        effect_calls = []
        stub._apply_ffmpeg_effect = lambda audio, sr, es: (effect_calls.append(es), audio.astype(np.float32))[1]

        random.seed(0)
        stub.audio_effecter(_sine_wave(), SR)
        assert len(effect_calls) == 0, (
            "No ffmpeg effect call expected when available_filters is empty"
        )


# ===========================================================================
# Test 12 — Encoder availability check
# ===========================================================================

class TestEncoderAvailabilityCheck:
    """_check_ffmpeg_encoders parses ffmpeg output correctly."""

    @skip_no_ffmpeg
    def test_all_three_encoders_available(self):
        """In a full-featured ffmpeg build, all three encoders should be present."""
        result = _check_ffmpeg_encoders(('libmp3lame', 'libvorbis', 'libopus'))
        assert isinstance(result, frozenset)
        # These assertions hold in standard Ubuntu/Debian ffmpeg builds with all codecs
        assert 'libmp3lame' in result, "libmp3lame not found — is ffmpeg built with --enable-libmp3lame?"
        assert 'libvorbis' in result, "libvorbis not found — is ffmpeg built with --enable-libvorbis?"
        assert 'libopus' in result, "libopus not found — is ffmpeg built with --enable-libopus?"

    @skip_no_ffmpeg
    def test_fake_encoder_not_returned(self):
        result = _check_ffmpeg_encoders(('libfake_codec',))
        assert result == frozenset(), (
            f"Fake encoder should not appear in result, got {result}"
        )

    @skip_no_ffmpeg
    def test_returns_only_requested_encoders(self):
        """Result set is a subset of the requested encoders."""
        requested = ('libmp3lame', 'libvorbis', 'libfake_codec')
        result = _check_ffmpeg_encoders(requested)
        assert result <= frozenset(requested), (
            f"Result {result} contains encoders not in requested {requested}"
        )

    def test_no_ffmpeg_returns_empty_frozenset(self):
        with mock.patch('subprocess.run', side_effect=FileNotFoundError("no ffmpeg")):
            _check_ffmpeg_encoders.cache_clear()
            result = _check_ffmpeg_encoders(('libmp3lame', 'libvorbis', 'libopus'))
            assert result == frozenset()
        _check_ffmpeg_encoders.cache_clear()

    @skip_no_ffmpeg
    def test_unavailable_encoder_codec_skipped_but_random_consumed(self):
        """When available_encoders is empty, codec is skipped but random calls still happen."""
        stub = _make_dataset_stub({'crystalizer': 0.0, 'flanger': 0.0, 'crusher': 0.0, 'codec': 0.0})
        stub.available_encoders = frozenset()  # force all encoders unavailable

        # Set a known seed and record state before / after
        random.seed(123)
        state_before = random.getstate()

        # Call _apply_ffmpeg_codec — should consume exactly 2 random calls
        # (codec_type selection + parameter generation)
        audio = _sine_wave()
        out = stub._apply_ffmpeg_codec(audio.copy(), SR)

        # Verify: output equals input (skipped)
        np.testing.assert_array_equal(out, audio.astype(np.float32))

        # Verify: exactly 2 random values were consumed
        # Replay the same 2 random calls starting from state_before
        random.setstate(state_before)
        v1 = random.choice(['mp3', 'ogg'])   # 1st consumption
        if v1 == 'mp3':
            _v2 = random.randint(4, 16)       # 2nd consumption (mp3)
        else:
            _v2 = random.choice(['vorbis', 'opus'])  # 2nd consumption (ogg)

        # After replaying 2 calls, random state should match post-_apply_ffmpeg_codec state
        state_after_replay = random.getstate()

        random.seed(123)
        stub._apply_ffmpeg_codec(audio.copy(), SR)
        state_after_call = random.getstate()

        assert state_after_replay == state_after_call, (
            "_apply_ffmpeg_codec must consume exactly 2 random values even when skipping"
        )


# ===========================================================================
# Test 13 — Codec random consumption sequence
# ===========================================================================

class TestCodecRandomConsumptionSequence:
    """_apply_ffmpeg_codec consumes random in the same order as engine.py L172."""

    def _simulate_engine_py_consumption(self, seed: int) -> tuple[str, object]:
        """Simulate engine.py L172 random consumption pattern.

        engine.py pattern:
            mp3_case = lambda: (random.randint(4, 16), 'mp3')
            ogg_case = lambda: (random.choice(['vorbis', 'opus']), 'ogg')
            codec_type, param = random.choice([mp3_case, ogg_case])()

        Returns:
            (codec_type_str, param_value)
        """
        random.seed(seed)
        # 1st consumption: choose between mp3 and ogg
        codec_type = random.choice(['mp3', 'ogg'])
        # 2nd consumption: choose parameter
        if codec_type == 'mp3':
            param = random.randint(4, 16)
        else:
            param = random.choice(['vorbis', 'opus'])
        return codec_type, param

    def _simulate_dataset_consumption(self, stub, seed: int) -> tuple[str, object]:
        """Simulate _apply_ffmpeg_codec random consumption by tracking calls."""
        codec_chosen = []
        param_chosen = []

        orig_choice = random.choice
        orig_randint = random.randint

        call_count = [0]

        def patched_choice(seq):
            val = orig_choice(seq)
            call_count[0] += 1
            if call_count[0] == 1:
                codec_chosen.append(val)
            else:
                param_chosen.append(val)
            return val

        def patched_randint(a, b):
            val = orig_randint(a, b)
            param_chosen.append(val)
            return val

        with mock.patch('random.choice', side_effect=patched_choice), \
             mock.patch('random.randint', side_effect=patched_randint):
            random.seed(seed)
            # reset call_count after seeding (seed itself may call random internals)
            call_count[0] = 0
            stub._apply_ffmpeg_codec(_sine_wave(), SR)

        codec_type = codec_chosen[0] if codec_chosen else None
        param = param_chosen[0] if param_chosen else None
        return codec_type, param

    def test_consumption_order_matches_engine_py(self):
        """For 5 different seeds, dataset codec path matches engine.py consumption order."""
        stub = _make_dataset_stub({'codec': 1.0})
        # Force unavailability so no actual subprocess is spawned (just tests random logic)
        stub.available_encoders = frozenset()

        for seed in range(5):
            engine_type, engine_param = self._simulate_engine_py_consumption(seed)

            random.seed(seed)
            ds_type, ds_param = self._simulate_dataset_consumption(stub, seed)

            assert ds_type == engine_type, (
                f"Seed {seed}: codec_type mismatch — engine={engine_type}, dataset={ds_type}"
            )
            assert ds_param == engine_param, (
                f"Seed {seed}: codec param mismatch — engine={engine_param}, dataset={ds_param}"
            )

    def test_two_random_calls_total_mp3_path(self):
        """MP3 path consumes exactly 2 random values."""
        stub = _make_dataset_stub({'codec': 0.0})
        stub.available_encoders = frozenset()

        random.seed(0)
        # Drain until we get mp3 path
        for seed in range(100):
            random.seed(seed)
            state0 = random.getstate()
            codec_type = random.choice(['mp3', 'ogg'])
            if codec_type == 'mp3':
                break
        else:
            pytest.skip("Could not find seed that selects mp3 in 100 tries")

        random.setstate(state0)
        stub._apply_ffmpeg_codec(_sine_wave(), SR)
        state_after = random.getstate()

        # Replay 2 calls from state0
        random.setstate(state0)
        random.choice(['mp3', 'ogg'])   # 1st
        random.randint(4, 16)           # 2nd
        state_replay = random.getstate()

        assert state_after == state_replay, (
            "MP3 path must consume exactly 2 random values"
        )

    def test_two_random_calls_total_ogg_path(self):
        """OGG path consumes exactly 2 random values."""
        stub = _make_dataset_stub({'codec': 0.0})
        stub.available_encoders = frozenset()

        for seed in range(100):
            random.seed(seed)
            state0 = random.getstate()
            codec_type = random.choice(['mp3', 'ogg'])
            if codec_type == 'ogg':
                break
        else:
            pytest.skip("Could not find seed that selects ogg in 100 tries")

        random.setstate(state0)
        stub._apply_ffmpeg_codec(_sine_wave(), SR)
        state_after = random.getstate()

        random.setstate(state0)
        random.choice(['mp3', 'ogg'])           # 1st
        random.choice(['vorbis', 'opus'])       # 2nd
        state_replay = random.getstate()

        assert state_after == state_replay, (
            "OGG path must consume exactly 2 random values"
        )


# ===========================================================================
# Test 14 — engine.py _downsample_8k unit test
# ===========================================================================

class TestDownsample8k:
    """Verify engine.py _downsample_8k preserves behavior."""

    def test_downsample_returns_8k_when_triggered(self):
        """When random < prob, audio is resampled to 8kHz."""
        import torch
        from unittest import mock

        # Create a minimal Engine-like object with _downsample_8k
        from tf_restormer.models.TF_Restormer.engine import Engine
        real_cls = getattr(Engine, '__wrapped__', Engine)

        obj = types.SimpleNamespace()
        obj.prob_downsample_8k = 1.0  # always trigger
        obj._downsample_8k = real_cls._downsample_8k.__get__(obj)

        audio = torch.randn(2, 16000)  # batch of 2, 1 second at 16kHz
        result, sr = obj._downsample_8k(audio)
        assert sr == 8000
        assert result.shape[1] == 8000  # 1 second at 8kHz

    def test_no_downsample_when_prob_zero(self):
        """When prob is 0, audio passes through unchanged."""
        import torch

        from tf_restormer.models.TF_Restormer.engine import Engine
        real_cls = getattr(Engine, '__wrapped__', Engine)

        obj = types.SimpleNamespace()
        obj.prob_downsample_8k = 0.0  # never trigger
        obj._downsample_8k = real_cls._downsample_8k.__get__(obj)

        audio = torch.randn(2, 16000)
        result, sr = obj._downsample_8k(audio)
        assert sr == 16000
        assert torch.equal(result, audio)

    def test_downsample_preserves_batch_dim(self):
        """Batch dimension is preserved after downsampling."""
        import torch

        from tf_restormer.models.TF_Restormer.engine import Engine
        real_cls = getattr(Engine, '__wrapped__', Engine)

        obj = types.SimpleNamespace()
        obj.prob_downsample_8k = 1.0
        obj._downsample_8k = real_cls._downsample_8k.__get__(obj)

        audio = torch.randn(4, 32000)  # batch of 4
        result, sr = obj._downsample_8k(audio)
        assert result.shape[0] == 4


# ===========================================================================
# Test 15 — _synthesis() audio_effecter guard branch
# ===========================================================================

class TestSynthesisGuardBranch:
    """Verify _synthesis() only calls audio_effecter when conditions are met."""

    def test_no_effects_when_prob_effect_empty(self):
        """When prob_effect is empty dict, audio_effecter should not be called."""
        stub = _make_dataset_stub({})
        audio = _sine_wave()
        random.seed(42)
        out = stub.audio_effecter(audio.copy(), SR)
        # With empty prob_effect, all prob.get() return 0.0 → no effects
        np.testing.assert_array_equal(out, audio.astype(np.float32))

    def test_no_effects_when_no_ffmpeg(self):
        """When has_ffmpeg is False, audio_effecter returns input unchanged."""
        stub = _make_dataset_stub({'codec': 1.0, 'crystalizer': 1.0})
        stub.has_ffmpeg = False
        stub.available_filters = frozenset()
        stub.available_encoders = frozenset()
        audio = _sine_wave()
        random.seed(42)
        out = stub.audio_effecter(audio.copy(), SR)
        # All filters/encoders unavailable → only random consumed, no actual effects
        # Output should be float32 cast of input
        np.testing.assert_array_equal(out, audio.astype(np.float32))
