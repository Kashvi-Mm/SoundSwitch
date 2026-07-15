"""
Maps each category to the frequency ranges to attenuate when that category
is muted, plus how much (in dB). These are starting guesses based on
typical spectral content -- expect to retune by ear.

Each band is (low_hz, high_hz, gain_db). gain_db is negative (attenuation).
"""

FREQUENCY_BANDS = {
    # Note: Traffic's frequency content (esp. honks, 1-2kHz) overlaps almost
    # entirely with speech. Cutting hard enough to fully silence traffic
    # also guts any concurrent speech, so these are deliberately mild
    # everywhere except the sub-250Hz rumble (which speech barely touches):
    # traffic gets noticeably reduced, not eliminated, when it overlaps speech.
    "Traffic": [
        (20, 250, -18),      # engine/road rumble -- below most speech content, safe to cut hard
        (250, 1000, -6),     # engine growl / body of honks -- real speech formants live here too
        (1000, 3500, -6),    # honk/whine peak range -- core speech intelligibility range, mild only
        (3500, 8000, -4),    # upper harmonics, very mild -- speech sibilance overlaps here
    ],
    "Mechanical Hums": [
        (50, 250, -24),     # mains hum + low harmonics
        (2000, 6000, -10),  # fan/motor whine
    ],
}

# Width (in Hz) of the raised-cosine taper applied at each band edge, so
# suppression fades in/out smoothly instead of cutting off sharply (a hard
# frequency-domain edge causes audible ringing in the reconstructed audio).
TAPER_WIDTH_HZ = 75
