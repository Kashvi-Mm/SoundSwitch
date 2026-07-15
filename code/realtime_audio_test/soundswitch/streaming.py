"""
True streaming DSP: overlap-add STFT processing on one ~32ms slice at a
time, instead of accumulating whole chunks before processing.
"""

import numpy as np
import librosa

from soundswitch.suppress import N_FFT, HOP_LENGTH, SAMPLE_RATE, category_gain_vector
from soundswitch.noise_reduction import apply_noise_reduction

# Sum of squared shifted periodic-Hann windows at 4x overlap (n_fft/hop_length),
# steady state -- verified numerically against librosa's own stft/istft
# round trip (RMSE ~1e-18, float64 machine precision), independent of the
# actual n_fft/hop_length values as long as their 4:1 ratio is preserved.
# Hann is applied on both analysis AND synthesis here (matching what
# librosa.istft itself does internally), which is why this is 1.5, not the
# single-Hann sum of 2.0.
_OLA_NORM = 1.5

# Live spectrogram/visualizer: how many frequency bars to show, and the
# display's dB floor/ceiling used to normalize magnitude to a 0-1 value.
N_SPECTROGRAM_BINS = 48
SPECTROGRAM_DB_FLOOR = -70.0
SPECTROGRAM_DB_CEILING = -10.0


def _log_bin_edges(freqs, n_bins, fmin=20.0):
    """
    Frequency-bin edges (as indices into an rfft-length array), log-spaced
    like a real spectrogram/piano-key layout so low frequencies get
    proportionally more visual resolution than a linear bin split would.
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
    necessarily suppress.py's SAMPLE_RATE (16000, what the frequency bands
    were validated at). n_fft/hop_length are scaled proportionally from
    suppress.py's N_FFT/HOP_LENGTH so the actual analysis window duration
    (128ms) and hop duration (32ms) stay the same in real time regardless
    of sr. Running at the audio device's own native rate (rather than
    forcing 16000 and making CoreAudio/PortAudio silently resample) avoids
    resampling overhead that otherwise adds substantial latency on top of
    the DSP's own floor.
    """

    def __init__(self, muted_categories, sr=SAMPLE_RATE, noise_profile=None, input_gain=1.0):
        # Mutable so a caller (e.g. a web server handling browser toggle
        # requests) can add/remove categories live, from a different thread
        # than the one running process_hop -- simple set mutation is safe
        # enough here without extra locking.
        self.muted_categories = set(muted_categories)
        self.noise_profile = noise_profile
        # Uniform gain applied to incoming audio before anything else --
        # purely to make quiet mic pickup audible enough to judge by ear.
        self.input_gain = input_gain
        scale = sr / SAMPLE_RATE
        self.n_fft = int(round(N_FFT * scale))
        self.hop_length = int(round(HOP_LENGTH * scale))
        self.sr = sr

        self.window = librosa.filters.get_window("hann", self.n_fft, fftbins=True)
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sr)

        self.history = np.zeros(self.n_fft, dtype=np.float32)  # input ring buffer
        self.ola = np.zeros(self.n_fft, dtype=np.float64)  # output overlap-add accumulator

        # Live spectrogram/visualizer snapshot -- read by a UI-facing thread,
        # written fresh each hop from the audio thread. Always reassigned as
        # a whole new list, never mutated in place, so a concurrent read is
        # always a complete valid snapshot.
        self._spectrogram_bin_idx = _log_bin_edges(self.freqs, N_SPECTROGRAM_BINS)
        self.latest_spectrum = [0.0] * N_SPECTROGRAM_BINS

    def _compute_band_mask(self):
        """
        Frequency-band suppression is memoryless, but muted_categories can
        change live (e.g. from a UI toggle), so this is recomputed fresh
        each hop rather than cached once -- cheap (a handful of tapered
        band vectors) relative to the FFT/IFFT already happening per hop.
        """
        mask = np.ones_like(self.freqs)
        # list(...) takes a defensive snapshot before iterating -- if some
        # caller ever mutates muted_categories in place (.add()/.discard())
        # from another thread instead of reassigning a new set, iterating
        # the live set directly would risk "RuntimeError: Set changed size
        # during iteration".
        for category in list(self.muted_categories):
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

        mask = self._compute_band_mask()
        masked_spec = spec * mask
        self.latest_spectrum = self._compute_spectrogram_snapshot(masked_spec)

        synth = np.fft.irfft(masked_spec, n=self.n_fft) * self.window
        self.ola += synth

        out = (self.ola[:self.hop_length] / _OLA_NORM).astype(np.float32)
        self.ola = np.concatenate([self.ola[self.hop_length:], np.zeros(self.hop_length)])
        return out
