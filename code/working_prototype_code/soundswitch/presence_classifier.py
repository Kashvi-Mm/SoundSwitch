"""
Runs YAMNet classification in its own background thread, decoupled from the
32ms audio hop cadence, to answer a coarse question roughly once a second:
which of a set of categories (Chatter, Traffic, Mechanical Hums) does the
recent audio actually sound like it contains? One YAMNet call per check
covers all tracked categories at once (not one call per category), so
tracking more categories doesn't multiply the classifier's CPU cost.

Used to gate suppression so muted categories only actually suppress while
YAMNet currently detects them -- e.g. Traffic only gets EQ-cut while traffic
sound is actually happening, not for the whole session just because it's
muted. Chatter's usage additionally combines this with an amplitude check
(see soundswitch/streaming.py) -- protecting anything YAMNet reads as plain
"Speech" regardless of loudness.
"""

import threading
import time

import numpy as np
import librosa

from soundswitch.yamnet_utils import load_yamnet, load_class_names, classify
from soundswitch.categories import build_label_to_category

CHECK_INTERVAL_SECONDS = 1.0
BUFFER_SECONDS = 1.5

# Starting guesses, tune by ear -- only Chatter's has been validated through
# extensive live testing this project. Traffic/Mechanical Hums are untested;
# expect to need the same kind of by-ear tuning Chatter went through.
CATEGORY_THRESHOLDS = {
    "Chatter": 0.12,
    "Traffic": 0.15,
    "Mechanical Hums": 0.15,
}

# Separate, more lenient thresholds used only for "is this worth showing the
# control for" (see is_worth_showing below) -- NOT for the actual suppression
# decision, which stays gated on CATEGORY_THRESHOLDS above. The two questions
# have very different costs when wrong: showing a switch that then doesn't
# duck anything is a minor inconvenience; ducking real direct speech by
# mistake is the thing this whole project is trying to avoid. So the UI can
# afford to be far more permissive than the suppression logic is.
VISIBILITY_THRESHOLDS = {
    "Chatter": 0.01,
    "Traffic": 0.03,
    "Mechanical Hums": 0.03,
}

# Chatter's own label score maxed out around 0.006 even during genuinely
# louder real background talking (measured live) -- YAMNet just doesn't read
# ordinary nearby speech as "Chatter/Hubbub/Crowd" regardless of volume, the
# same finding from earlier detection debugging. Lowering VISIBILITY_THRESHOLDS
# further would mean showing the card almost unconditionally (indistinguishable
# from the noise floor), which defeats the point. Instead, Chatter's card
# visibility ALSO lights up on plain "Speech" activity -- a much stronger,
# more reliable signal (0.5-0.98 confidence whenever anyone's actually
# talking) -- since "is there vocal activity nearby" is a far easier, more
# available question than "does it specifically sound like babble." Ducking
# itself is unaffected -- that still requires the strict Chatter-specific
# score in CATEGORY_THRESHOLDS.
SPEECH_VISIBILITY_THRESHOLD = 0.3

# A category only actually becomes "worth showing" once it's been
# continuously above its visibility threshold for at least this long -- a
# single momentary blip (or check-to-check noise) shouldn't pop a card in
# and immediately back out. Resets to 0 the instant a check comes back
# below threshold, so this only gates appearing, not disappearing.
REQUIRED_PRESENCE_SECONDS = 7.0

# Once a card is shown, it stays visible for at least this long since the
# LAST time it was actually detected -- not just a single instant. Without
# this, a category that comes and goes (e.g. chatter with natural pauses)
# flickers the card on/off every ~1s check instead of feeling stable.
MIN_VISIBLE_SECONDS = 30.0


class PresenceClassifier:
    def __init__(
        self, sr, categories, check_interval_seconds=CHECK_INTERVAL_SECONDS,
        buffer_seconds=BUFFER_SECONDS, thresholds=None, debug=False,
    ):
        self.sr = sr
        self.check_interval_seconds = check_interval_seconds
        self.categories = list(categories)
        self.thresholds = thresholds or CATEGORY_THRESHOLDS
        self.debug = debug

        self.yamnet_model = load_yamnet()
        class_names = load_class_names(self.yamnet_model)
        label_to_category = build_label_to_category(class_names)
        self.category_indices = {
            cat: [i for i, n in enumerate(class_names) if label_to_category[n] == cat]
            for cat in self.categories
        }
        # Tracked separately from `categories` -- only used to broaden
        # Chatter's visibility threshold, see SPEECH_VISIBILITY_THRESHOLD.
        self.speech_indices = [i for i, n in enumerate(class_names) if label_to_category[n] == "Speech"]

        self._buffer = np.zeros(int(buffer_seconds * sr), dtype=np.float32)
        self._lock = threading.Lock()
        # Read by the audio thread every hop, no lock needed -- always
        # reassigned as a whole new dict from _run(), never mutated in
        # place, so a concurrent read always sees one complete, consistent
        # snapshot (same atomic-swap pattern as muted_categories).
        self.is_active = {cat: False for cat in self.categories}
        # More lenient than is_active -- see VISIBILITY_THRESHOLDS above.
        # Only meant for "should the UI show this control," never for
        # suppression decisions.
        self.is_worth_showing = {cat: False for cat in self.categories}
        self.last_score = {cat: 0.0 for cat in self.categories}
        # Only ever touched by _run() on the background thread, so no lock
        # needed -- tracks how many consecutive seconds each category has
        # been above its visibility threshold, for REQUIRED_PRESENCE_SECONDS.
        self._presence_streak_seconds = {cat: 0.0 for cat in self.categories}
        # Wall-clock timestamp of the last time each category was actually
        # detected (pre-hold) -- drives MIN_VISIBLE_SECONDS below.
        self._last_true_time = {cat: None for cat in self.categories}
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_flag = True

    def push_audio(self, samples):
        """
        Called every hop from the real-time audio thread -- must stay fast.
        Lock hold time is just a short array concat, microseconds, an
        acceptable (not zero, but very low) risk to the audio callback's
        timing budget.
        """
        with self._lock:
            n = len(samples)
            self._buffer = np.concatenate([self._buffer[n:], samples])

    def _run(self):
        while not self._stop_flag:
            time.sleep(self.check_interval_seconds)
            with self._lock:
                snapshot = self._buffer.copy()

            audio_16k = librosa.resample(snapshot, orig_sr=self.sr, target_sr=16000)
            scores, _, _ = classify(self.yamnet_model, audio_16k)

            speech_score = float(scores[:, self.speech_indices].max()) if self.speech_indices else 0.0
            now = time.time()

            new_active, new_worth_showing, new_scores = {}, {}, {}
            for cat in self.categories:
                indices = self.category_indices[cat]
                score = float(scores[:, indices].max()) if indices else 0.0
                new_scores[cat] = score
                new_active[cat] = score > self.thresholds.get(cat, 0.12)

                if cat == "Chatter":
                    # Tied directly to Speech, immediately -- no streak delay.
                    # Chatter's own label score has repeatedly measured near
                    # zero even during genuinely audible real chatter (YAMNet
                    # reads it as plain Speech instead), so Speech is the
                    # actually-reliable signal here, not a fallback.
                    raw_worth_showing = speech_score > SPEECH_VISIBILITY_THRESHOLD
                else:
                    worth_showing_now = score > VISIBILITY_THRESHOLDS.get(cat, 0.03)
                    # Accumulate while present, decay (not hard-reset) while
                    # absent -- real conversation has natural pauses between
                    # sentences/turns, and a single below-threshold check
                    # shouldn't wipe out an otherwise genuinely sustained
                    # presence. A truly brief one-off blip still decays back
                    # to 0 before reaching REQUIRED_PRESENCE_SECONDS; only
                    # audio present more than it's absent actually accumulates.
                    if worth_showing_now:
                        self._presence_streak_seconds[cat] = min(
                            self._presence_streak_seconds[cat] + self.check_interval_seconds,
                            REQUIRED_PRESENCE_SECONDS,
                        )
                    else:
                        self._presence_streak_seconds[cat] = max(
                            self._presence_streak_seconds[cat] - self.check_interval_seconds,
                            0.0,
                        )
                    raw_worth_showing = self._presence_streak_seconds[cat] >= REQUIRED_PRESENCE_SECONDS

                # Minimum-hold: once shown, stay visible until MIN_VISIBLE_SECONDS
                # have passed since the LAST time it was actually detected --
                # smooths over natural gaps instead of flickering on/off.
                if raw_worth_showing:
                    self._last_true_time[cat] = now
                last_true = self._last_true_time[cat]
                new_worth_showing[cat] = last_true is not None and (now - last_true) < MIN_VISIBLE_SECONDS

                if self.debug:
                    print(
                        f"[presence] {cat}: score={score:.3f} speech_score={speech_score:.3f} "
                        f"streak={self._presence_streak_seconds[cat]:.1f}s raw={raw_worth_showing} "
                        f"visible={new_worth_showing[cat]}"
                    )
            self.last_score = new_scores
            self.is_active = new_active
            self.is_worth_showing = new_worth_showing
