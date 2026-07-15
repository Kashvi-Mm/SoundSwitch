"""
General background-noise reduction via spectral subtraction: capture a
short "noise profile" (average magnitude spectrum) from a quiet moment,
then continuously subtract that profile from every subsequent frame's
magnitude, keeping phase unchanged. Unlike the category-based frequency
bands (Traffic/Mechanical Hums), this doesn't assume the noise lives in
any particular hand-picked range -- it targets whatever was actually
present during calibration (room hum, hiss, AC, electrical interference),
which is why it's a separate mechanism.
"""

import numpy as np
import librosa

# How aggressively to subtract the estimated noise (>1.0 subtracts more
# than the raw estimate, since real noise fluctuates around its average).
OVER_SUBTRACTION = 1.5

# Never reduce a bin's magnitude below this fraction of its original value.
# Naive spectral subtraction without a floor produces "musical noise"
# (isolated random-sounding tonal artifacts) in bins that get subtracted
# to near-zero inconsistently frame to frame.
SPECTRAL_FLOOR = 0.1


def estimate_noise_profile(calibration_audio, n_fft, hop_length, window):
    """
    calibration_audio: a short recording (e.g. ~1s) of just background
    noise, no wanted signal. Returns the average magnitude spectrum
    across all its frames -- one value per frequency bin.
    """
    stft = librosa.stft(calibration_audio, n_fft=n_fft, hop_length=hop_length, window=window)
    return np.abs(stft).mean(axis=1)


def apply_noise_reduction(spec, profile, over_subtraction=OVER_SUBTRACTION, floor=SPECTRAL_FLOOR):
    """
    spec: one hop's complex spectrum (from np.fft.rfft). Returns the
    noise-reduced complex spectrum -- magnitude reduced, phase untouched.
    """
    magnitude = np.abs(spec)
    phase = np.angle(spec)
    reduced_magnitude = np.maximum(magnitude - over_subtraction * profile, floor * magnitude)
    return reduced_magnitude * np.exp(1j * phase)
