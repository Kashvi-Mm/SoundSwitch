"""
Maps YAMNet's 521 fine-grained labels down to the small set of user-facing
perceptual categories. Labels are hand-picked as they're observed in test
clips -- anything not listed here defaults to "Other" and is never suppressed.

Note: "Chatter" and "Hubbub, speech noise, speech babble" are real, distinct
YAMNet labels (separate from "Speech"/"Conversation") -- split into their own
"Chatter" category. Since Chatter is spectrally identical to Speech (it's
still voices, just distant/overlapping ones), it isn't suppressed with a
frequency-band cut like Traffic/Mechanical Hums -- see suppress.py's
amplitude-based ducking, which uses relative loudness instead to tell
foreground speech from background chatter.
"""

CATEGORY_OVERRIDES = {
    # Speech
    "Speech": "Speech",
    "Child speech, kid speaking": "Speech",
    "Conversation": "Speech",
    "Babbling": "Speech",
    "Shout": "Speech",
    "Yell": "Speech",
    "Children shouting": "Speech",
    "Whispering": "Speech",

    # Chatter (background/distant voices, not suppressed by frequency --
    # see amplitude-based ducking in suppress.py)
    "Chatter": "Chatter",
    "Hubbub, speech noise, speech babble": "Chatter",
    "Crowd": "Chatter",

    # Traffic
    "Vehicle": "Traffic",
    "Motor vehicle (road)": "Traffic",
    "Car": "Traffic",
    "Vehicle horn, car horn, honking": "Traffic",
    "Car alarm": "Traffic",
    "Skidding": "Traffic",
    "Tire squeal": "Traffic",
    "Truck": "Traffic",
    "Bus": "Traffic",
    "Motorcycle": "Traffic",
    "Traffic noise, roadway noise": "Traffic",
    "Honk": "Traffic",
    "Accelerating, revving, vroom": "Traffic",

    # Mechanical Hums
    "Engine": "Mechanical Hums",
    "Light engine (high frequency)": "Mechanical Hums",
    "Medium engine (mid frequency)": "Mechanical Hums",
    "Heavy engine (low frequency)": "Mechanical Hums",
    "Engine knocking": "Mechanical Hums",
    "Engine starting": "Mechanical Hums",
    "Idling": "Mechanical Hums",
    "Mechanical fan": "Mechanical Hums",
    "Air conditioning": "Mechanical Hums",
    "Hum": "Mechanical Hums",
    "Mains hum": "Mechanical Hums",
    "Static": "Mechanical Hums",
    "White noise": "Mechanical Hums",
    "Vibration": "Mechanical Hums",
    "Humming": "Mechanical Hums",
}

CATEGORIES = ["Speech", "Chatter", "Traffic", "Mechanical Hums", "Other"]
DEFAULT_CATEGORY = "Other"


def build_label_to_category(class_names):
    """Every one of the 521 labels maps to something; unmapped -> 'Other' (never suppressed)."""
    return {name: CATEGORY_OVERRIDES.get(name, DEFAULT_CATEGORY) for name in class_names}
