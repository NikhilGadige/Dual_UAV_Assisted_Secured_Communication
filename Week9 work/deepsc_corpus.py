"""
Small synthetic text corpus used to train the DeepSC semantic communication
model (see deepsc_model.py). Built from template combinations so the
vocabulary stays compact while still exercising varied sentence structure.
"""

import random
from typing import List, Tuple

SUBJECTS = [
    "the base station", "the mobile user", "the uav jammer", "the target vehicle",
    "the ris platform", "the network controller", "the sensor node", "the ground station",
]
VERBS = [
    "transmits", "receives", "decodes", "encodes",
    "monitors", "tracks", "forwards", "processes",
]
OBJECTS = [
    "the semantic message", "the sensing signal", "the control command", "the channel estimate",
    "the encoded symbol", "the target location", "the jamming interference", "the data packet",
]
ADVERBIALS = [
    "over the wireless channel", "with high reliability", "under strong interference", "in real time",
    "across the network", "during the uplink phase", "at low power", "with minimal delay",
]


def build_corpus(num_sentences: int = 400, seed: int = 42) -> List[str]:
    """Generates a fixed, reproducible set of unique synthetic sentences."""
    rng = random.Random(seed)
    combos = set()
    sentences = []
    max_combos = len(SUBJECTS) * len(VERBS) * len(OBJECTS) * len(ADVERBIALS)
    num_sentences = min(num_sentences, max_combos)
    while len(sentences) < num_sentences:
        s = rng.choice(SUBJECTS)
        v = rng.choice(VERBS)
        o = rng.choice(OBJECTS)
        a = rng.choice(ADVERBIALS)
        key = (s, v, o, a)
        if key in combos:
            continue
        combos.add(key)
        sentences.append(f"{s} {v} {o} {a}")
    return sentences


def train_val_split(sentences: List[str], val_ratio: float = 0.2, seed: int = 7) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]
