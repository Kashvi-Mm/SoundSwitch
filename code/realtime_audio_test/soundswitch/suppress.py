"""
Core DSP: build a frequency-domain gain mask from category band configs.
"""

import numpy as np

from soundswitch.frequency_bands import FREQUENCY_BANDS, TAPER_WIDTH_HZ

N_FFT = 2048
HOP_LENGTH = 512
SAMPLE_RATE = 16000


def db_to_linear(db):
    return 10 ** (db / 20)


def band_gain_vector(freqs, low_hz, high_hz, gain_db, taper_hz=TAPER_WIDTH_HZ):
    """
    Real-valued gain per FFT frequency bin: 1.0 outside the band, gain_db
    (converted to linear) inside it, with a raised-cosine taper of width
    taper_hz at each edge so the transition is smooth rather than a hard
    cutoff (a hard edge causes ringing when inverted back to a waveform).
    """
    gain_linear = db_to_linear(gain_db)
    vec = np.ones_like(freqs)

    inside = (freqs >= low_hz) & (freqs <= high_hz)
    vec[inside] = gain_linear

    def taper(edge_center, rising_into_band):
        lo, hi = edge_center - taper_hz / 2, edge_center + taper_hz / 2
        in_taper = (freqs >= lo) & (freqs <= hi)
        if not np.any(in_taper):
            return
        t = (freqs[in_taper] - lo) / (hi - lo)  # 0..1 across the taper
        cosine = 0.5 * (1 - np.cos(np.pi * t))  # 0..1 raised-cosine ramp
        ramp = cosine if rising_into_band else (1 - cosine)
        vec[in_taper] = 1.0 + ramp * (gain_linear - 1.0)

    taper(low_hz, rising_into_band=True)
    taper(high_hz, rising_into_band=False)
    return vec


def category_gain_vector(freqs, category):
    """Combined gain vector for one category's full band list (bands multiply together)."""
    vec = np.ones_like(freqs)
    for low_hz, high_hz, gain_db in FREQUENCY_BANDS.get(category, []):
        vec *= band_gain_vector(freqs, low_hz, high_hz, gain_db)
    return vec
