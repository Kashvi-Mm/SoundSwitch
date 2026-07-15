"""
True streaming DSP: overlap-add STFT processing on one ~32ms slice at a
time, instead of accumulating whole chunks before processing. Reuses the
exact same frequency-band and Chatter-ducking logic as the batch path
(suppress.py) -- see StreamingSuppressor.process_hop for how.
"""

import numpy as np
import librosa

from soundswitch.suppress import (
    N_FFT, HOP_LENGTH, SAMPLE_RATE, CHATTER_DUCK_DB,
    category_gain_vector, chatter_duck_gain, smooth_mask, db_to_linear,
)
from soundswitch.noise_reduction import apply_noise_reduction

# Sum of squared shifted periodic-Hann windows at 4x overlap (n_fft/hop_length),
# steady state -- verified numerically against librosa's own stft/istft
# round trip (RMSE ~1e-18, float64 machine precision), independent of the
# actual n_fft/hop_length values as long as their 4:1 ratio is preserved.
# Hann is applied on both analysis AND synthesis here (matching what
# librosa.istft itself does internally), which is why this is 1.5, not the
# single-Hann sum of 2.0.
_OLA_NORM = 1.5

# Live spectrogram: how many frequency bars the UI shows, and the display's
# dB floor/ceiling used to normalize magnitude to a 0-1 brightness value.
N_SPECTROGRAM_BINS = 48
SPECTROGRAM_DB_FLOOR = -70.0
SPECTROGRAM_DB_CEILING = -10.0


def _log_bin_edges(freqs, n_bins, fmin=20.0):
    """
    Frequency-bin edges (as indices into an rfft-length array), log-spaced
    like a real spectrogram/piano-key layout so low frequencies (where voice
    fundamentals and rhythm live) get proportionally more visual resolution
    than a linear bin split would give them.
    """
    fmax = freqs[-1]
    edges_hz = np.logspace(np.log10(fmin), np.log10(fmax), n_bins + 1)
    return np.searchsorted(freqs, edges_hz)


class StreamingSuppressor:
    """
    Call process_hop() once per hop_length-sized slice of incoming audio,
    in order, and it returns hop_length samples of suppressed output for
    that same slice. All state needed between calls lives on the instance.

    sr: the actual audio device sample rate this will run at -- NOT
    necessarily suppress.py's SAMPLE_RATE (16000, what our frequency bands/
    Chatter tuning were validated at). n_fft/hop_length are scaled
    proportionally from suppress.py's N_FFT/HOP_LENGTH so the actual
    analysis window duration (128ms) and hop duration (32ms) stay the same
    in real time regardless of sr -- this keeps frequency-bin resolution
    and all of Chatter's time constants (computed internally from
    hop_length/sr) correct at any sample rate. Running at the audio
    device's own native rate (rather than forcing 16000 and making
    CoreAudio/PortAudio silently resample) avoids resampling overhead that
    otherwise adds substantial latency on top of the DSP's own floor.
    """

    def __init__(
        self, muted_categories, sr=SAMPLE_RATE, noise_profile=None, debug_chatter=False,
        presence_classifier=None, feed_classifier=True, force_chatter_duck=False, input_gain=1.0,
    ):
        # Mutable so a caller (e.g. a web server handling browser toggle
        # requests) can add/remove categories live, from a different thread
        # than the one running process_hop -- simple set mutation is safe
        # enough here without extra locking, same reasoning already used
        # for PresenceClassifier.is_active.
        self.muted_categories = set(muted_categories)
        self.noise_profile = noise_profile
        self.debug_chatter = debug_chatter
        self._debug_hop_count = 0
        # PresenceClassifier (soundswitch/presence_classifier.py): periodic
        # (~1s) YAMNet check for whether Chatter/Traffic/Mechanical Hums are
        # actually currently present -- gates suppression so a muted category
        # only actually suppresses while it's genuinely detected, not for the
        # whole session just because it's muted. If None, every muted
        # category suppresses unconditionally (matches the original
        # always-on behavior, e.g. for live_demo.py runs without a
        # classifier attached).
        self.presence_classifier = presence_classifier
        # False when a caller feeds the classifier from a separate, independent
        # input stream instead (e.g. a room-facing mic distinct from the
        # low-latency duplex device) -- pushing this stream's audio into the
        # classifier's buffer too would interleave two different mics' audio
        # into one rolling buffer, corrupting it.
        self.feed_classifier = feed_classifier
        # Debug-only: bypass both the amplitude and classifier gates and just
        # apply CHATTER_DUCK_DB constantly whenever "Chatter" is muted -- lets
        # you A/B hear what the ducking amount itself sounds like, decoupled
        # from whether real-world audio happens to trip either detector.
        self.force_chatter_duck = force_chatter_duck
        # Uniform gain applied to incoming audio before anything else -- purely
        # to make quiet mic pickup audible enough to judge by ear. Does NOT
        # change the amplitude gate's decision (loudness vs. reference is a
        # ratio, scale-invariant to a constant multiplier applied to both), and
        # does NOT change YAMNet's classification of what's already a very
        # quiet signal. Only affects what you can actually hear.
        self.input_gain = input_gain
        scale = sr / SAMPLE_RATE
        self.n_fft = int(round(N_FFT * scale))
        self.hop_length = int(round(HOP_LENGTH * scale))
        self.sr = sr

        self.window = librosa.filters.get_window("hann", self.n_fft, fftbins=True)
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sr)

        self.history = np.zeros(self.n_fft, dtype=np.float32)  # input ring buffer
        self.ola = np.zeros(self.n_fft, dtype=np.float64)  # output overlap-add accumulator
        self.state = {}  # Chatter ducking state, same shape as apply_constant_suppression's

        # Live spectrogram snapshot -- read by a Flask request thread
        # (soundswitch nothing else touches it), written fresh each hop from
        # the audio thread. Always reassigned as a whole new array, never
        # mutated in place, so a concurrent read is always a complete valid
        # snapshot -- same atomic-swap reasoning as muted_categories.
        self._spectrogram_bin_idx = _log_bin_edges(self.freqs, N_SPECTROGRAM_BINS)
        self.latest_spectrum = [0.0] * N_SPECTROGRAM_BINS

    def seed_chatter_reference(self, reference_loudness):
        """
        Seed the Chatter amplitude gate's "recent loud peak" reference at
        startup, from a short calibration recording of the user's actual
        direct voice, instead of leaving it to cold-start from whatever
        ambient audio happens to arrive first. Without this, peak_envelope's
        reference just tracks the incoming signal itself -- if no distinctly
        loud direct-voice moment happens to occur organically during a
        session, the reference never rises above ambient level and ratio_db
        stays near 0 the whole time, so the amplitude gate can never fire,
        regardless of how quiet real background chatter actually is.
        """
        self.state["chatter"] = {"loudness": reference_loudness, "envelope": reference_loudness}

    def _compute_band_mask(self):
        """
        Frequency-band suppression (Traffic/Mechanical Hums) is memoryless,
        but muted_categories can now change live (e.g. from a UI toggle),
        so this is recomputed fresh each hop rather than cached once --
        cheap (a handful of tapered band vectors) relative to the FFT/IFFT
        already happening per hop.

        Gated on presence: a muted category only actually cuts its bands
        while presence_classifier currently detects it (e.g. Traffic only
        suppresses while traffic sound is actually happening) -- not for the
        whole session just because the switch is off. Defaults to always-on
        (no gating) if no presence_classifier is attached or a category
        isn't one it tracks, so behavior is unchanged for callers that don't
        set one up.
        """
        mask = np.ones_like(self.freqs)
        # list(...) takes a defensive snapshot before iterating -- if some
        # caller ever mutates muted_categories in place (.add()/.discard())
        # from another thread instead of reassigning a new set, iterating
        # the live set directly would risk "RuntimeError: Set changed size
        # during iteration".
        for category in list(self.muted_categories):
            if category == "Chatter":
                continue
            is_active = (
                self.presence_classifier is None
                or category not in self.presence_classifier.is_active
                or self.presence_classifier.is_active[category]
            )
            if is_active:
                mask *= category_gain_vector(self.freqs, category)
        return mask

    def _compute_spectrogram_snapshot(self, spec):
        """
        Post-suppression magnitude, binned log-spaced and dB-normalized to
        0-1 for the UI -- post-suppression (not the raw input spectrum) so
        muting a category visibly changes what's drawn, same as what you
        actually hear.
        """
        magnitude = np.abs(spec)
        bars = np.zeros(N_SPECTROGRAM_BINS)
        idx = self._spectrogram_bin_idx
        for i in range(N_SPECTROGRAM_BINS):
            lo, hi = idx[i], idx[i + 1]
            bars[i] = magnitude[lo:hi].mean() if hi > lo else 0.0
        db = 20 * np.log10(bars + 1e-8)
        normalized = (db - SPECTROGRAM_DB_FLOOR) / (SPECTROGRAM_DB_CEILING - SPECTROGRAM_DB_FLOOR)
        return np.clip(normalized, 0.0, 1.0).tolist()

    def process_hop(self, new_samples):
        """new_samples: exactly self.hop_length float32 samples in/out."""
        if self.input_gain != 1.0:
            new_samples = np.clip(new_samples * self.input_gain, -1.0, 1.0).astype(np.float32)
        self.history = np.concatenate([self.history[self.hop_length:], new_samples])

        frame = self.history * self.window
        spec = np.fft.rfft(frame)

        if self.noise_profile is not None:
            spec = apply_noise_reduction(spec, self.noise_profile)

        # Feed the classifier continuously, regardless of what's currently
        # muted -- so if the user toggles a category on mid-session (e.g.
        # from the web UI), its presence state is already warmed up and
        # reflects the current acoustic scene immediately, instead of
        # starting from an empty buffer and needing a fresh ~1-2s to catch up.
        if self.presence_classifier is not None and self.feed_classifier:
            self.presence_classifier.push_audio(new_samples)

        mask = self._compute_band_mask()
        if "Chatter" in self.muted_categories:
            # Reuse chatter_duck_gain/smooth_mask completely unchanged: called
            # with a single-column "spectrum," their internal per-array loops
            # simply don't execute, leaving exactly the single-hop recurrence
            # step seeded from the persisted state -- not a second, separately
            # written implementation of the same EMA/peak-envelope math.
            duck_gain, chatter_state = chatter_duck_gain(
                spec[:, np.newaxis], sr=self.sr, hop_length=self.hop_length,
                state=self.state.get("chatter"),
            )
            self.state["chatter"] = chatter_state

            # AND-gate: only actually duck if YAMNet's periodic check also
            # agrees this sounds chatter-like. Anything it reads as plain
            # "Speech" is protected regardless of how the amplitude signal
            # alone would have decided -- "direct speech always protected."
            classifier_says_chatter = (
                self.presence_classifier is None
                or self.presence_classifier.is_active.get("Chatter", True)
            )
            if self.force_chatter_duck:
                duck_gain = np.array([db_to_linear(CHATTER_DUCK_DB)])
            elif not classifier_says_chatter:
                duck_gain = np.array([1.0])

            if self.debug_chatter:
                self._debug_hop_count += 1
                if self._debug_hop_count % 15 == 0:  # ~every 480ms, avoid flooding the terminal
                    loudness = chatter_state["loudness"]
                    reference = chatter_state["envelope"]
                    ratio_db = 20 * np.log10((loudness + 1e-8) / (reference + 1e-8))
                    triggered = duck_gain[0] < 1.0
                    raw_score = (
                        self.presence_classifier.last_score.get("Chatter", float("nan"))
                        if self.presence_classifier is not None else float("nan")
                    )
                    print(
                        f"[chatter] loudness={loudness:.5f} reference={reference:.5f} "
                        f"ratio_db={ratio_db:+.2f} threshold={-0.5:+.2f} "
                        f"classifier_chatter={classifier_says_chatter} raw_score={raw_score:.3f} "
                        f"{'DUCKING' if triggered else ''}"
                    )

            mask = mask * duck_gain[0]
            mask = smooth_mask(
                mask[:, np.newaxis], sr=self.sr, hop_length=self.hop_length,
                initial=self.state.get("mask"),
            )[:, 0]
            self.state["mask"] = mask

        masked_spec = spec * mask
        self.latest_spectrum = self._compute_spectrogram_snapshot(masked_spec)

        synth = np.fft.irfft(masked_spec, n=self.n_fft) * self.window
        self.ola += synth

        out = (self.ola[:self.hop_length] / _OLA_NORM).astype(np.float32)
        self.ola = np.concatenate([self.ola[self.hop_length:], np.zeros(self.hop_length)])
        return out
